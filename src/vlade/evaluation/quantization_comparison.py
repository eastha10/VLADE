"""Plan and run independent S0 quantization conditions in isolated processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Callable
import uuid

ROOT = Path(__file__).resolve().parents[3]
METHODS = {"bf16", "uniform_w4", "uniform_w8", "actquant", "qvla", "quant"}
SCOPES = {"language", "language_vision"}
REFERENCE_FIELDS = ("source", "hf_inputs", "checkpoint_seal_sha256", "data_manifest_sha256", "seed",
                    "calibration_queries", "eval_queries", "warmup", "rollout_count_requested", "benchmark_source",
                    "dtype", "batch_size", "gpu", "torch", "transformers", "cuda")


def atomic_json(path: Path, value: dict) -> None:
    """Replace a runtime report atomically; never emit JSON NaN/Infinity."""
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    paths = ["src", "scripts", "configs", "docs", "README.md", "STRUCTURE.md"]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--ignore-submodules=all", "--", *paths)
    return {"git_sha": git("rev-parse", "HEAD"), "source_dirty": bool(dirty)}


def registry_path(cfg: dict) -> Path:
    path = Path(cfg["registry"])
    return path if path.is_absolute() else ROOT / path


def validate_config(cfg: dict) -> dict:
    if cfg.get("schema_version") != 1 or cfg.get("phase") != "pilot":
        raise ValueError("This runner supports schema 1, pilot comparisons only")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg.get("run_group", "")):
        raise ValueError("Unsafe run_group")
    for key in ("registry", "output_root"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ValueError(f"Require {key}")
    for key in ("checkpoint", "data"):
        if not isinstance(cfg.get(key), str) or not cfg[key].startswith("hfasset://"):
            raise ValueError("Comparison inputs must use immutable HF asset keys")
    if cfg["data"] != "hfasset://prepared_d_seed42":
        raise ValueError("Use the existing prepared D/seed42 split")
    seeds = cfg.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Require distinct integer seeds in [0, 2**32)")
    for key, minimum in (("calibration_queries", 3), ("eval_queries", 1), ("warmup", 0), ("rollouts", 0)):
        if type(cfg.get(key)) is not int or cfg[key] < minimum:
            raise ValueError(f"Invalid {key}")
    if cfg["rollouts"] > 2402:
        raise ValueError("Spatial rollout count exceeds the available variants")
    cases = cfg.get("conditions")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Require independent conditions")
    ids, baseline = set(), 0
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "method", "scope", "target_bpw"}:
            raise ValueError("Each condition requires id, method, scope and target_bpw only")
        if not isinstance(case["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) or case["id"] in ids:
            raise ValueError("Unsafe/duplicate condition id")
        ids.add(case["id"])
        method, scope, target = case["method"], case["scope"], case["target_bpw"]
        if method not in METHODS or scope not in SCOPES:
            raise ValueError("Unsupported method/scope")
        if type(target) not in (float, int) or not math.isfinite(target) or not 2 <= target <= 16:
            raise ValueError("Invalid target weight budget")
        fixed = {"bf16": 16, "uniform_w4": 4, "uniform_w8": 8, "quant": 8}.get(method)
        if fixed is not None and target != fixed:
            raise ValueError("Fixed-precision condition has an incompatible bit budget")
        if method == "quant" and scope != "language":
            raise ValueError("QuantVLA pilot supports language-only W8; do not invent vision/W4 support")
        baseline += method == "bf16"
    if baseline != 1:
        raise ValueError("Require exactly one independent BF16 reference")
    return cfg


def load_config(path: Path) -> dict:
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def input_records(cfg: dict) -> dict:
    from vlade.storage.assets import asset_record
    records = {key: asset_record(cfg[key][len("hfasset://"):], registry_path(cfg)) for key in ("checkpoint", "data")}
    if cfg["rollouts"]:
        records["benchmark_assets"] = asset_record("libero_plus_assets", registry_path(cfg))
    model = records["checkpoint"]
    if (model["repo_id"] != "VLADE-konkuk/VLADE-checkpoints" or model["repo_type"] != "model"
            or model.get("format") != "directory" or "identity.json" not in model.get("files", {})
            or not any(p.startswith("model/") for p in model["files"])):
        raise ValueError("Require a complete registered QwenVLA checkpoint")
    data = records["data"]
    if (data["repo_id"] != "VLADE-konkuk/libero-plus-prepared" or data["repo_type"] != "dataset"
            or data.get("format") != "archive"
            or not {"manifest.json", "normalization.json", "splits.json"} <= data.get("identity_files", {}).keys()):
        raise ValueError("Require the registered prepared dataset with its original split/normalization identity")
    return records


def protect_output(output: Path, cfg: dict) -> None:
    """Never place runtime writes inside existing weight/data/cache inputs."""
    from vlade.storage.assets import cache_path
    from vlade.storage.local_inputs import registered_roots
    protected = [ROOT / "data", ROOT / "checkpoints"]
    protected.extend(registered_roots())
    for kind, record in input_records(cfg).items():
        key = "libero_plus_assets" if kind == "benchmark_assets" else cfg[kind][len("hfasset://"):]
        protected.append(cache_path(key, record).parent.parent.parent)
    for name in ("HF_HOME", "HF_HUB_CACHE", "VLADE_ASSET_CACHE"):
        if os.environ.get(name):
            protected.append(Path(os.environ[name]))
    if any(output.resolve().is_relative_to(path.resolve()) for path in protected):
        raise ValueError("Comparison output must stay outside dataset/checkpoint/HF caches")


def condition_table(cfg: dict) -> list[dict]:
    return [{**case,
             "comparison_group": "reference" if case["method"] == "bf16" else f"{case['scope']}-budget-{case['target_bpw']:g}",
             "precision": "BF16 native" if case["method"] == "bf16" else "weight-only fake quantization; BF16 activations",
             "implementation": "pilot; original-paper completeness not claimed",
             "independent_from_original_checkpoint": True} for case in cfg["conditions"]]


def plan(cfg: dict) -> dict:
    validate_config(cfg)
    records = input_records(cfg)
    return {"phase": cfg["phase"], "conditions": condition_table(cfg), "seeds": cfg["seeds"],
            "calibration": {"split": "train/is_calibration", "queries": cfg["calibration_queries"]},
            "evaluation": {"split": "validation", "queries": cfg["eval_queries"], "spatial_rollouts": cfg["rollouts"]},
            "inputs": {k: {n: r[n] for n in ("repo_id", "repo_type", "revision")} for k, r in records.items()},
            "selection_eligible": False, "automatic_winner": None,
            "gpu_execution": False, "downloads": False, "remote_hf_checked": False}


def paired_result(report: dict, reference: dict, case: dict) -> dict:
    """Accept only complete, finite, paired measurements with matching provenance."""
    run = report["runs"].get(case["method"])
    base = reference["runs"].get("bf16")
    if report.get("status") != "completed" or reference.get("status") != "completed" or not run or not base:
        raise ValueError("Incomplete condition/reference")
    for key in REFERENCE_FIELDS:
        if report.get(key) != reference.get(key):
            raise ValueError("Condition/reference identity mismatch: " + key)
    if run["samples"]["validation"] != base["samples"]["validation"]:
        raise ValueError("Conditions must evaluate the identical ordered validation queries")
    queries = run["samples"]["validation"]
    if len(queries) != report["eval_queries"] or len({tuple(q) for q in queries}) != len(queries):
        raise ValueError("Evaluation query count/identity mismatch")
    metrics = ("action_mae_vs_ground_truth", "action_mae_vs_bf16", "forward_p50_ms", "forward_p95_ms",
               "peak_allocated_bytes", "peak_reserved_bytes")
    if any(type(run.get(k)) not in (int, float) or not math.isfinite(run[k]) or run[k] < 0 for k in metrics):
        raise ValueError("Non-finite or invalid condition metrics")
    if case["method"] != "bf16":
        if case["method"] in {"actquant", "qvla", "quant"}:
            calibration = run["samples"].get("train", [])
            if len(calibration) != report["calibration_queries"] or len({tuple(q) for q in calibration}) != len(calibration):
                raise ValueError("Calibration query count/identity mismatch")
        if run.get("quantization_scope") != case["scope"] or run.get("target_bpw") != case["target_bpw"]:
            raise ValueError("Worker scope/bit budget differs from the condition table")
        actual = run["weight_bpw_excluding_metadata"]
        if not math.isfinite(actual) or not 2 <= actual <= case["target_bpw"] + 1e-9:
            raise ValueError("Worker exceeded its requested parameter-weighted budget")
        if set(run["target_modules"]) != set(run["bit_allocation"]):
            raise ValueError("Allocation does not cover the declared target modules")
        sizes = run["target_modules"]
        allocated = 0
        for name, size in sizes.items():
            if type(size) is not int or size <= 0:
                raise ValueError("Invalid target size")
            bits = run["bit_allocation"][name]
            if isinstance(bits, dict):
                if any(k not in {"2", "4", "8", "16"} or type(n) is not int or n < 0 for k, n in bits.items()):
                    raise ValueError("Invalid row allocation")
                rows = sum(bits.values())
                if rows < 1 or size % rows:
                    raise ValueError("Row allocation does not match target size")
                allocated += size // rows * sum(int(k) * n for k, n in bits.items())
            elif type(bits) is int and bits in {2, 4, 8, 16}:
                allocated += size * bits
            else:
                raise ValueError("Invalid tensor allocation")
        if not sizes or not math.isclose(allocated / sum(sizes.values()), actual, rel_tol=1e-8):
            raise ValueError("Reported bit average disagrees with its allocation")
    if report["rollout_count_requested"]:
        a, b = run.get("rollout", {}), base.get("rollout", {})
        if a.get("status") != "complete" or b.get("status") != "complete":
            return {"status": "rollout_unavailable", "metrics": {k: run[k] for k in metrics}}
        identity = lambda r: [(t["task_id"], t["init_sha256"], t["seed"]) for t in r["trials"]]
        if identity(a) != identity(b):
            raise ValueError("Rollout task/initial-state pairs differ")
    return {"status": "completed", "metrics": {k: run[k] for k in metrics},
            "mae_delta_vs_bf16": run["action_mae_vs_ground_truth"] - base["action_mae_vs_ground_truth"],
            "achieved_target_weight_bpw": run.get("weight_bpw_excluding_metadata"),
            "rollout": run.get("rollout")}


def invoke(command: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}).returncode


def run_comparison(cfg: dict, output: Path, *, resume: bool = False,
                   worker: Callable[[list[str], Path], int] = invoke) -> dict:
    proposal = plan(cfg)
    protect_output(output, cfg)
    source = source_identity()
    if source["source_dirty"]:
        raise ValueError("Commit the reviewed research source before GPU execution; dry-run remains available")
    binding = os.environ.get("VLADE_LOCAL_INPUTS")
    binding_hash = hashlib.sha256(Path(binding).read_bytes()).hexdigest() if binding else None
    identity = {"config": cfg, "source": source, "inputs": proposal["inputs"], "local_bindings_sha256": binding_hash}
    if resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Resume requires identical config, source commit and HF revisions")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": 1, "identity": identity, "plan": proposal, "status": "running", "jobs": {}}
        atomic_json(output / "config.json", cfg)
        atomic_json(output / "conditions.json", {"conditions": proposal["conditions"]})
    cases = sorted(cfg["conditions"], key=lambda c: c["method"] != "bf16")
    for seed in cfg["seeds"]:
        reference = None
        reference_path = None
        for case in cases:
            key = f"seed-{seed}/{case['id']}"
            previous = manifest["jobs"].get(key, {})
            if previous.get("status") == "completed":
                path = output / previous["report"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != previous["report_sha256"]:
                    raise ValueError("Completed report changed; preserve the original measurement")
                report = json.loads(path.read_text(encoding="utf-8"))
                paired_result(report, report if case["method"] == "bf16" else reference, case)
                if case["method"] == "bf16":
                    reference, reference_path = report, path
                continue
            if case["method"] != "bf16" and reference is None:
                manifest["jobs"][key] = {"status": "blocked", "reason": "BF16 reference unavailable"}
                atomic_json(output / "manifest.json", manifest)
                continue
            attempt = previous.get("attempt", 0) + 1
            destination = output / key / f"attempt-{attempt:03d}"
            while destination.exists():
                attempt += 1
                destination = output / key / f"attempt-{attempt:03d}"
            destination.mkdir(parents=True, exist_ok=False)
            report_path = destination / "report.json"
            job = {"status": "running", "attempt": attempt, "comparison_group": "reference" if case["method"] == "bf16"
                   else f"{case['scope']}-budget-{case['target_bpw']:g}", "report": report_path.relative_to(output).as_posix()}
            manifest["jobs"][key] = job
            manifest["status"] = "running"
            atomic_json(output / "manifest.json", manifest)
            command = [sys.executable, "-B", str(ROOT / "scripts/evaluate/run_s0_quantization.py"),
                       "--checkpoint", cfg["checkpoint"], "--data", cfg["data"], "--registry", str(registry_path(cfg)),
                       "--output", str(report_path), "--condition", case["method"], "--scope", case["scope"],
                       "--target-bpw", str(case["target_bpw"]), "--seed", str(seed),
                       "--calibration-queries", str(cfg["calibration_queries"]), "--eval-queries", str(cfg["eval_queries"]),
                       "--warmup", str(cfg["warmup"]), "--rollouts", str(cfg["rollouts"])]
            if reference_path is not None:
                command.extend(["--reference-report", str(reference_path)])
            try:
                print(f"Starting {key}, attempt {attempt}", flush=True)
                code = worker(command, destination / "worker.log")
                if code:
                    raise RuntimeError("Worker returned failure")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                job.update(paired_result(report, report if case["method"] == "bf16" else reference, case))
                job["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
                if case["method"] != "bf16":
                    for other_key, other in manifest["jobs"].items():
                        if other_key == key or not other_key.startswith(f"seed-{seed}/") or other.get("status") != "completed" or other.get("comparison_group") != job["comparison_group"]:
                            continue
                        peer = json.loads((output / other["report"]).read_text(encoding="utf-8"))
                        peer_run = next(iter(peer["runs"].values()))
                        if peer_run["target_modules"] != report["runs"][case["method"]]["target_modules"]:
                            raise ValueError("Methods in the same scope/budget group targeted different modules")
                        a = peer_run["samples"].get("train")
                        b = report["runs"][case["method"]]["samples"].get("train")
                        if a is not None and b is not None and a != b:
                            raise ValueError("Methods used different calibration queries")
                if case["method"] == "bf16" and job["status"] == "completed":
                    reference, reference_path = report, report_path
            except Exception as error:
                job.update(status="failed", error_type=type(error).__name__)
            print(f"{key}: {job['status']}", flush=True)
            atomic_json(output / "manifest.json", manifest)
            atomic_json(output / "summary.json", {"jobs": manifest["jobs"], "selection_eligible": False,
                "automatic_winner": None, "note": "Compare only within the same declared scope and budget; fake quantization is not low-bit deployment."})
    manifest["status"] = "completed" if all(j["status"] == "completed" for j in manifest["jobs"].values()) else "incomplete"
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "summary.json", {"status": manifest["status"], "jobs": manifest["jobs"],
                                          "selection_eligible": False, "automatic_winner": None})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true", help="Explicitly allow model/data resolution and GPU evaluation")
    parser.add_argument("--condition", action="append", help="Condition id; BF16 reference is always included")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output and args.resume:
        parser.error("Choose a new output or a resume directory")
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if args.condition:
        ids = {c["id"] for c in cfg["conditions"]}
        if set(args.condition) - ids:
            parser.error("Unknown condition id")
        cfg["conditions"] = [c for c in cfg["conditions"] if c["id"] in args.condition or c["method"] == "bf16"]
    if not args.run:
        print(json.dumps(plan(cfg), indent=2, allow_nan=False))
        return
    output = args.resume or args.output or Path(cfg["output_root"]) / (
        cfg["run_group"] + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    result = run_comparison(cfg, output.resolve(), resume=args.resume is not None)
    print(json.dumps({"output": str(output), "status": result["status"]}))
    if result["status"] != "completed":
        raise SystemExit(1)
