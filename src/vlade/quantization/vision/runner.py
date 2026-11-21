"""Configuration-driven vision PTQ, with download-free preflight."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import re
import subprocess
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
METHODS = {"fp", "w8", "w4", "mixed", "random", "visual", "ava", "combined"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported PTQ configuration")
    for key in ("alpha", "beta", "target_avg_bits"):
        if type(cfg.get(key)) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError(f"Invalid {key}")
    if min(cfg["alpha"], cfg["beta"]) < 0 or not 4 <= cfg["target_avg_bits"] <= 8:
        raise ValueError("Invalid sensitivity weights or bit budget")
    for key in ("calibration_episodes", "max_queries_per_episode", "batch_size"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    for key in ("quantize_siglip", "quantize_dinov2", "quantize_patch_embed", "upload"):
        if type(cfg.get(key)) is not bool:
            raise ValueError(f"{key} must be boolean")
    if not (cfg["quantize_siglip"] or cfg["quantize_dinov2"]):
        raise ValueError("At least one vision backbone must be selected")
    if cfg.get("method") not in METHODS:
        raise ValueError("Unknown method")
    if not isinstance(cfg.get("device"), str) or not isinstance(cfg.get("output_root"), str) or not cfg["output_root"]:
        raise ValueError("Require device and output_root")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg.get("run_group", "")):
        raise ValueError("Unsafe run group")
    for key in ("checkpoint", "calibration_data"):
        if not isinstance(cfg.get(key), str) or not cfg[key].startswith("hfasset://"):
            raise ValueError("Inputs must use pinned hfasset:// keys")
    if cfg["calibration_data"] != "hfasset://prepared_d_seed42":
        raise ValueError("Only the existing D/seed42 calibration subset is supported")
    ablations = cfg.get("ablations")
    if not isinstance(ablations, dict) or not ablations:
        raise ValueError("Require ablation definitions")
    for name, entry in ablations.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or entry.get("method") not in METHODS:
            raise ValueError("Invalid ablation")
        if set(entry) - {"method", "alpha", "beta"}:
            raise ValueError("Ablations share one bit budget and scope")
        for key in ("alpha", "beta"):
            v = entry.get(key, cfg[key])
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError("Invalid ablation sensitivity coefficient")
    return cfg


def registry_path(cfg: dict) -> Path:
    path = Path(cfg.get("registry", "configs/storage/hf_assets.lock.json"))
    return path if path.is_absolute() else ROOT / path


def input_records(cfg: dict) -> dict:
    from vlade.storage.assets import asset_record
    records = {key: asset_record(cfg[key][len("hfasset://"):], registry_path(cfg))
               for key in ("checkpoint", "calibration_data")}
    checkpoint = records["checkpoint"]
    if (checkpoint["repo_id"] != "VLADE-konkuk/VLADE-checkpoints"
            or checkpoint.get("format") != "directory" or not checkpoint.get("files")
            or "t0" in checkpoint.get("prefix", "").lower()):
        raise ValueError("Require a sealed T1 checkpoint asset, never T0")
    return records


def inspect_hf(records: dict) -> dict:
    """Check remote revisions and file identities using metadata only."""
    from huggingface_hub import HfApi
    from vlade.storage.assets import configure_http
    configure_http()
    api = HfApi()
    report = {}
    for key, record in records.items():
        info = api.repo_info(record["repo_id"], repo_type=record["repo_type"],
                             revision=record["revision"], files_metadata=True)
        if info.sha != record["revision"]:
            raise ValueError("HF revision mismatch")
        files = {item.rfilename: item for item in info.siblings}
        expected = ({record["prefix"].rstrip("/") + "/files/" + name: value
                     for name, value in record["files"].items()}
                    if record.get("format") == "directory" else
                    {record["archive_path"]: {"sha256": record["archive_sha256"], "bytes": record["archive_bytes"]}})
        verified = 0
        for name, value in expected.items():
            item = files.get(name)
            if item is None or ("bytes" in value and item.size != value["bytes"]):
                raise ValueError(f"Pinned HF file missing/size mismatch: {name}")
            lfs = getattr(item, "lfs", None)
            digest = getattr(lfs, "sha256", None) if lfs else None
            if digest is not None:
                if digest != value["sha256"]:
                    raise ValueError(f"Pinned HF SHA256 mismatch: {name}")
                verified += 1
        report[key] = {"revision": info.sha, "files_present": len(expected),
                       "lfs_sha256_verified": verified,
                       "payload_sha256": "verified by resolve_input before execution"}
    return report


def remote_json(record: dict, relative: str) -> dict:
    """Read a small, hash-verified metadata file without downloading weights."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import get_session, build_hf_headers
    expected = record["files"][relative]
    if expected["bytes"] > 2 * 1024 * 1024:
        raise ValueError("Metadata exceeds preflight size limit")
    url = hf_hub_url(record["repo_id"], record["prefix"] + "/files/" + relative,
                     revision=record["revision"], repo_type=record["repo_type"])
    response = get_session().get(url, headers=build_hf_headers(), timeout=30)
    response.raise_for_status()
    payload = response.content
    if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise ValueError("HF metadata SHA256 mismatch: " + relative)
    return json.loads(payload)


def backbone_roles(model_config: dict) -> dict[str, str]:
    ids = model_config.get("timm_model_ids", [])
    if not 1 <= len(ids) <= 2:
        raise ValueError("Unsupported Prismatic vision configuration")
    result = {}
    for prefix, name in zip(("featurizer", "fused_featurizer"), ids):
        role = "siglip" if "siglip" in name.lower() else "dinov2" if "dinov2" in name.lower() else None
        if role is None or role in result.values():
            raise ValueError("Cannot identify SigLIP/DINOv2 backbone order")
        result[prefix] = role
    return result


def preflight(cfg: dict, *, remote: bool = True) -> dict:
    records = input_records(cfg)
    result = {"status": "configuration_valid", "inputs": {
        key: {k: r[k] for k in ("repo_id", "revision", "format")} for key, r in records.items()},
              "target_scope": "vision_backbone.{featurizer,fused_featurizer}.blocks.*.{attn,mlp} Linear",
              "patch_embedding": cfg["quantize_patch_embed"],
              "module_inventory": "deferred until verified T1 is loaded; no model instantiated",
              "fake_quantization": True, "weight_or_dataset_downloads": False, "gpu_execution": False}
    if remote:
        result["hf"] = inspect_hf(records)
        record = records["checkpoint"]
        config = remote_json(record, "config.json")
        metadata = remote_json(record, "vlade_fft.json")
        seal = remote_json(record, "checkpoint_manifest.json")
        identity_files = records["calibration_data"]["identity_files"]
        expected_identity = {k: identity_files[f]["sha256"] for k, f in (
            ("manifest_sha256", "manifest.json"), ("normalization_sha256", "normalization.json"),
            ("split_sha256", "splits.json"))}
        if (metadata.get("model") != "ava_vla" or metadata.get("quantized", False) is not False
                or metadata.get("step") != seal.get("step")
                or metadata.get("prepared_data_identity") != expected_identity):
            raise ValueError("T1 role, step or calibration identity mismatch")
        expected_files = {k: v for k, v in record["files"].items()
                          if k not in {"COMPLETE", "checkpoint_manifest.json"}}
        if seal.get("schema_version") != 1 or seal.get("files") != expected_files:
            raise ValueError("T1 checkpoint seal disagrees with asset registry")
        roles = backbone_roles(config)
        index = remote_json(record, "model.safetensors.index.json")
        names = []
        for key in index["weight_map"]:
            for prefix, role in roles.items():
                if not cfg["quantize_" + role]:
                    continue
                base = "vision_backbone." + prefix + "."
                name = key.removeprefix(base)
                if key.startswith(base) and key.endswith(".weight") and (
                    (name.startswith("blocks.") and any(p in name.split(".") for p in ("attn", "mlp")))
                    or (cfg["quantize_patch_embed"] and name == "patch_embed.proj.weight")):
                    names.append(key)
        if not names:
            raise ValueError("No requested vision modules in checkpoint index")
        result["module_inventory"] = sorted(names)
        result["backbone_roles"] = roles
        result["metadata_sha256"] = "config and weight index verified; payload deferred to resolver"
    else:
        result["hf"] = "remote metadata not checked (offline preflight)"
    return result


def calibration_queries(cfg: dict, checkpoint: Path, data: object, signature: dict) -> tuple:
    """Capture FP causal contexts once; probes never advance recurrent state."""
    import torch
    from vlade.training.fixed_cache import SequentialTeacher
    selected = sorted(e["episode_id"] for e in data.entries
                      if e["split"] == "train" and e.get("is_calibration") is True)
    random.Random(cfg["seed"]).shuffle(selected)
    selected = selected[:cfg["calibration_episodes"]]
    if not selected:
        raise ValueError("No existing Train calibration episodes")
    policy = SequentialTeacher(checkpoint, device=cfg["device"],
                               temporal_feature_layer=signature["temporal_feature_layer"])
    queries = []
    def capture_vision(module, args):
        current["pixels"] = args[0].detach().cpu().clone()
    def capture_context(module, args, kwargs):
        current["action"] = None if args[1] is None else args[1].detach().cpu().clone()
        current["text"] = args[2].detach().cpu().clone()
        current["kwargs"] = copy.deepcopy(kwargs)
    handles = [policy.model.vision_backbone.register_forward_pre_hook(capture_vision),
               policy.generator.register_forward_pre_hook(capture_context, with_kwargs=True)]
    try:
        with torch.no_grad():
            for entry, samples in data.episodes("train", order=selected):
                policy.reset()
                for sample in samples[:cfg["max_queries_per_episode"]]:
                    current = {"episode_id": entry["episode_id"], "timestep": sample["timestep"]}
                    # Existing policy computes the FP temporal state. No action metrics are produced.
                    policy.predict(sample, data)
                    if not {"pixels", "text", "action", "kwargs"} <= current.keys():
                        raise RuntimeError("Teacher capture interface mismatch")
                    queries.append(current)
    finally:
        for handle in handles:
            handle.remove()
    device = next(policy.model.parameters()).device
    def evaluate(query):
        def move(value):
            return None if value is None else value.to(device).clone()
        vision = policy.model.vision_backbone(move(query["pixels"]))
        projected = policy.model.projector(vision)
        _, weights = policy.generator(projected, move(query["action"]), move(query["text"]),
                                      **copy.deepcopy(query["kwargs"]))
        return vision, weights[0]
    return policy, queries, evaluate


def execute(cfg: dict, *, pilot: bool = False) -> list[Path]:
    """Select the compatible registry only for this invocation."""
    previous = os.environ.get("VLADE_ASSET_REGISTRY")
    os.environ["VLADE_ASSET_REGISTRY"] = str(registry_path(cfg))
    try:
        return _execute(cfg, pilot=pilot)
    finally:
        if previous is None:
            os.environ.pop("VLADE_ASSET_REGISTRY", None)
        else:
            os.environ["VLADE_ASSET_REGISTRY"] = previous


def _execute(cfg: dict, *, pilot: bool = False) -> list[Path]:
    import torch
    from vlade.storage.assets import resolve_input
    from vlade.data.episode_data import EpisodeData
    from vlade.training.teacher_inference import teacher_identity
    from .quantizer import VisionQuantizer
    from .sensitivity import measure
    from .allocation import allocate
    records = input_records(cfg)
    inspect_hf(records)
    # Existing resolver verifies immutable HF origin and complete file SHA256.
    data = EpisodeData(resolve_input(cfg["calibration_data"], registry=registry_path(cfg)))
    checkpoint, signature = teacher_identity(cfg["checkpoint"], data)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    if pilot:
        cfg = {**cfg, "calibration_episodes": min(2, cfg["calibration_episodes"]),
               "max_queries_per_episode": min(2, cfg["max_queries_per_episode"])}
    policy, queries, evaluate = calibration_queries(cfg, checkpoint, data, signature)
    roles = backbone_roles(json.loads((checkpoint / "config.json").read_text()))
    options = {k: cfg[k] for k in ("quantize_siglip", "quantize_dinov2", "quantize_patch_embed")}
    # A single-backbone model may intentionally have no DINOv2 branch.
    for role in ("siglip", "dinov2"):
        if options["quantize_" + role] and role not in roles.values():
            raise ValueError(f"Requested {role} is absent from checkpoint")
    quantizer = VisionQuantizer(policy.model.vision_backbone, backbone_roles=roles, **options)
    method = "combined" if cfg["method"] == "mixed" else cfg["method"]
    alpha, beta = cfg["alpha"], cfg["beta"]
    if method == "visual":
        alpha, beta = 1., 0.
    elif method == "ava":
        alpha, beta = 0., 1.
    scores = measure(quantizer, queries, evaluate, alpha=alpha, beta=beta,
                     batch_size=cfg["batch_size"]) if method in {"visual", "ava", "combined"} else {}
    if method == "fp":
        allocation = {}
    elif method in {"w4", "w8"}:
        allocation = dict.fromkeys(quantizer.groups, int(method[1:]))
    else:
        allocation = allocate(quantizer.sizes, scores, cfg["target_avg_bits"],
                              random_seed=cfg["seed"] if method == "random" else None)
    group = cfg["run_group"] + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(cfg["output_root"]) / group
    output.mkdir(parents=True, exist_ok=False)
    used = sum(quantizer.sizes[n] * b for n, b in allocation.items())
    manifest = {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "hf_revision": records["checkpoint"]["revision"],
                "dataset_revision": records["calibration_data"]["revision"],
                "dataset_manifest_revision": data.manifest.get("dataset_revision"),
                "seed": cfg["seed"], "calibration_samples": len(queries),
                "queries": [{k: q[k] for k in ("episode_id", "timestep")} for q in queries],
                "targets": quantizer.sizes, "backbone_roles": roles, "allocation": allocation,
                "checkpoint_identity": signature, "pilot": pilot,
                "split_sha256": data.manifest["split_sha256"],
                "normalization_sha256": data.manifest["normalization_sha256"],
                "torch": str(torch.__version__), "device": cfg["device"],
                "hardware": torch.cuda.get_device_name(cfg["device"]) if str(cfg["device"]).startswith("cuda")
                    else platform.processor(),
                "platform": platform.platform(), "cuda": torch.version.cuda,
                "dtype": str(next(policy.model.parameters()).dtype),
                "fake_quantization": True, "alpha": alpha, "beta": beta}
    try:
        quantizer.apply(allocation)
        write_json(output / "config.json", {**cfg, "run_group": group})
        write_json(output / "manifest.json", manifest)
        write_json(output / "sensitivity.json", scores)
        write_json(output / "bit_allocation.json", allocation)
        write_json(output / "quantization_summary.json", {
            "method": method, "status": "completed", "selected_parameters": sum(quantizer.sizes.values()),
            "allocated_bits": used, "effective_avg_bits": used / sum(quantizer.sizes.values()) if allocation else None,
            "target_avg_bits": cfg["target_avg_bits"], "fp": method == "fp",
            "weights_saved": False, "reconstruction": "apply allocation to the same verified FP checkpoint",
            "integer_acceleration": False, "global_optimum_claimed": False})
    finally:
        quantizer.restore()
    if cfg["upload"]:
        from huggingface_hub import HfApi
        api = HfApi()
        if api.model_info("VLADE-konkuk/VLADE-checkpoints").private is not True:
            raise ValueError("PTQ upload requires private checkpoint repository")
        api.upload_folder(repo_id="VLADE-konkuk/VLADE-checkpoints", folder_path=str(output),
                          path_in_repo="runs/quantization/" + group,
                          allow_patterns=["*.json"], commit_message="Add vision PTQ metadata " + group)
    return [output]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Dry-run only: skip remote HF metadata")
    parser.add_argument("--pilot", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--method", choices=sorted(METHODS))
    parser.add_argument("--run-group")
    args = parser.parse_args()
    if args.offline and not args.dry_run:
        parser.error("--offline requires --dry-run")
    cfg = load_config(args.config)
    if args.run_group:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_group):
            parser.error("Unsafe run group")
        cfg["run_group"] = args.run_group
    runs = [{**cfg, **entry, "run_group": cfg["run_group"] + "-" + name}
            for name, entry in cfg["ablations"].items()] if args.all else [{**cfg, "method": args.method or cfg["method"]}]
    for run in runs:
        if args.dry_run:
            print(json.dumps(preflight(run, remote=not args.offline), indent=2))
        else:
            print(json.dumps([str(p) for p in execute(run, pilot=args.pilot)]))
