"""Plan-supervised AVA FFT client around the existing trainer and private Hub."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from vlade.runtime.environment import locate, prepare, runtime_environment
from vlade.storage.assets import REGISTRY_REPO, asset_record, read_registry, sha256
from vlade.utils.source_identity import source_identity

ROOT = Path(__file__).resolve().parents[5]
CLI = ROOT / "src/vlade/legacy/plan_cot/scripts/train/run_teacher_plan.py"
ACTIONS = ("plan", "setup", "template", "publish-annotations", "preflight", "pilot", "train", "retry", "export-pin")


def load_settings(path: Path, overrides: dict | None = None) -> dict:
    """Validate the portable settings without accessing remote inputs or a GPU."""
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, item in (overrides or {}).items():
        target = value
        *parts, name = key.split(".")
        for part in parts:
            target = target.get(part)
            if not isinstance(target, dict):
                raise ValueError("Unknown config section: " + key)
        if name not in target:
            raise ValueError("Unknown config key: " + key)
        target[name] = copy.deepcopy(item)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported Teacher Plan workflow")
    base_path = (ROOT / value["base_config"]).resolve()
    if not base_path.is_relative_to((ROOT / "configs").resolve()):
        raise ValueError("base_config must remain inside project configs")
    revision = value.get("registry_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("registry_revision must be an immutable HF commit")
    for key in ("initial_checkpoint", "data", "annotations"):
        if not re.fullmatch(r"hfasset://[A-Za-z0-9][A-Za-z0-9_.-]*", value[key]) or ".." in value[key]:
            raise ValueError(key + " must select a pinned HF asset")
    weight = value["plan_loss_weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Plan CE weight must be finite and positive")
    # Reuse the existing FFT configuration validation for training controls.
    from vlade.training.fft import FFTConfig

    raw = build_fft_config(value, Path("T1_plan"), pilot=False)
    allowed = set(FFTConfig.__dataclass_fields__) - {"model", "config_path"}
    if set(raw["fft_defaults"]) - allowed or set(raw["fft_runs"]["ava_vla"]) - allowed:
        raise ValueError("Unknown FFT setting")
    merged = {**raw["fft_defaults"], **raw["fft_runs"]["ava_vla"], "model": "ava_vla"}
    for key in ("checkpoint", "data_root", "output_dir", "prepared_data_root", "plan_annotations"):
        merged[key] = Path(merged[key])
    for key in ("max_steps", "grad_accumulation_steps", "save_steps", "eval_steps", "max_eval_batches"):
        if type(merged[key]) is not int or merged[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if merged["save_steps"] > merged["max_steps"]:
        raise ValueError("save_steps cannot exceed max_steps")
    FFTConfig(**merged).validate()
    return value


def build_fft_config(settings: dict, output: Path, *, pilot: bool) -> dict:
    """Keep the original FFT loss, optimizer and K4 contract; use T1 as new initialization."""
    base = json.loads((ROOT / settings["base_config"]).read_text(encoding="utf-8"))
    base["fft_defaults"].update(settings["train"], prepared_data_root=settings["data"],
        plan_annotations=settings["annotations"], plan_loss_weight=settings["plan_loss_weight"],
        image_aug=False, save_optimizer_state=True)
    base["fft_runs"] = {"ava_vla": {**base["fft_runs"]["ava_vla"],
                                    "checkpoint": settings["initial_checkpoint"], "output_dir": str(output)}}
    if pilot:
        base["fft_defaults"].update(max_steps=1, save_steps=1, eval_steps=1, warmup_steps=0)
    return base


def job(config: str | Path, *, workspace: str | Path, run_group: str, action: str = "plan",
        overrides: dict | None = None, resume_revision: str | None = None,
        review_source: str | None = None, colab: bool = False) -> None:
    """Launch the same CLI from notebooks; only the child receives a Colab secret."""
    command = [sys.executable, "-B", str(CLI), "--config", str(Path(config).resolve()),
        "--workspace", str(Path(workspace).resolve()), "--run-group", run_group, "--action", action,
        "--overrides", json.dumps(overrides or {})]
    if resume_revision:
        command += ["--resume-revision", resume_revision]
    if review_source:
        command += ["--review-source", review_source]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    if colab and action != "plan":
        from google.colab import userdata
        env["HF_TOKEN"] = userdata.get("HF_TOKEN")
    if action == "plan":
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    logs = Path(workspace) / "runs" / run_group / "launch-logs"
    logs.mkdir(parents=True, exist_ok=True)
    import uuid
    log = logs / (action + "-" + uuid.uuid4().hex[:8] + ".log")
    with log.open("x", encoding="utf-8") as stream, subprocess.Popen(command, cwd=ROOT, env=env,
            start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8") as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            if process.wait():
                raise RuntimeError(f"Teacher Plan {action} failed; see the original error above and {log}")
        except KeyboardInterrupt:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            raise


def _write_immutable(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Settings or source changed; use a new run group: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_inputs(settings: dict, work: Path, run: Path, *, annotations: bool) -> Path:
    """Pin the runtime registry and reject unavailable annotations before model downloads."""
    from vlade.research.registry import prepare_registry
    from vlade.storage.checkpoint_inputs import pin_checkpoint_inputs

    source = prepare_registry(ROOT, work, settings["registry_revision"])
    registry = read_registry(source)
    if annotations and settings["annotations"][10:] not in registry["assets"]:
        raise ValueError("Reviewed Plan annotations are not registered at this revision. "
                         "Prepare/review/publish the template and set its registry_revision first.")
    path = run / ("inputs.json" if annotations else "annotation-inputs.json")
    pins = settings["checkpoint_pins"] if annotations else {}
    pin_checkpoint_inputs(registry, pins, path)
    return path


def export_pin(output: Path, group: str, target: int) -> dict:
    """Verify a final supervised local seal and pinned remote file identities before S2 handoff."""
    from huggingface_hub import HfApi, hf_hub_download
    from vlade.legacy.plan_cot.distillation.teacher_plan import require_plan_supervision
    from vlade.storage.checkpoint_inputs import checkpoint_record
    from vlade.training.run_control import verify_checkpoint

    checkpoint = output / f"step-{target:07d}"
    if not checkpoint.is_dir():
        raise RuntimeError("Target step is not complete; preserve the last checkpoint and resume this run")
    seal = verify_checkpoint(checkpoint)
    metadata = json.loads((checkpoint / "vlade_fft.json").read_text(encoding="utf-8"))
    require_plan_supervision(metadata)
    if metadata["model"] != "ava_vla" or metadata["step"] != target or seal["step"] != target:
        raise ValueError("A target-step AVA Teacher checkpoint is required")
    if not (checkpoint / "trainer_state.pt").is_file():
        raise ValueError("The supervised Teacher must contain resumable training state")
    pending = output / "hub_pending.json"
    if pending.exists() and json.loads(pending.read_text(encoding="utf-8"))["queue"]:
        raise RuntimeError("Hub upload pending; retry before exporting the Teacher pin")
    info = HfApi().model_info(REGISTRY_REPO)
    if info.private is not True:
        raise ValueError("Teacher checkpoints must remain private")
    base = f"runs/{group}/{output.name}"
    pointer = json.loads(Path(hf_hub_download(REGISTRY_REPO, base + "/final.json", revision=info.sha)).read_text())
    prefix = base + "/checkpoints/" + checkpoint.name
    if pointer["checkpoint"] != prefix or pointer["step"] != target:
        raise ValueError("Remote final pointer does not match the target checkpoint")
    record = checkpoint_record(REGISTRY_REPO, info.sha, prefix)
    if record["files"]["checkpoint_manifest.json"]["sha256"] != sha256(checkpoint / "checkpoint_manifest.json"):
        raise ValueError("Remote and local Teacher seals differ")
    pin = {"repo_id": REGISTRY_REPO, "revision": info.sha, "prefix": prefix}
    report = {"TEACHER_CHECKPOINT_PIN": pin, "step": target, "pending": False,
              "verification": "local seal and remote seal/file identities; no full weight redownload",
              "plan_semantics_and_rollout": "not_verified"}
    (output.parent / (output.name + "-handoff.json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def _worker(settings: dict, work: Path, run: Path, group: str, action: str,
            revision: str | None, review_source: str | None) -> None:
    from vlade.training.hub import HubCheckpoints, download_checkpoint
    from vlade.storage.assets import resolve_input

    source = source_identity(ROOT, require_clean=True)
    _write_immutable(run / "run-contract.json", {"settings": settings, "group": group, **source})
    pilot = action == "pilot"
    selected_group = group + "-pilot" if pilot else group
    output = run / ("T1_plan_pilot" if pilot else "T1_plan")
    os.environ.update(VLADE_HUB_REPO=REGISTRY_REPO, VLADE_HUB_RUN_ID=selected_group)
    if action == "retry":
        # Retry both outputs without conflating the pilot run with the main run.
        for folder, hub_group in ((run / "T1_plan_pilot", group + "-pilot"), (output, group)):
            if (folder / "hub_pending.json").exists():
                os.environ["VLADE_HUB_RUN_ID"] = hub_group
                if not HubCheckpoints(folder).flush():
                    raise RuntimeError("Hub upload remains pending")
        return
    if action == "export-pin":
        export_pin(output, group, settings["train"]["max_steps"])
        return
    need_annotations = action not in {"template", "publish-annotations"}
    registry = prepare_inputs(settings, work, run, annotations=need_annotations)
    os.environ["VLADE_ASSET_REGISTRY"] = str(registry)
    if not need_annotations:
        from vlade.legacy.plan_cot.data.teacher_plan_annotations import export_template, publish_reviewed
        draft = work / "annotations" / group
        if action == "template":
            print(json.dumps(export_template(settings["data"], draft), indent=2))
        else:
            key = settings["annotations"][10:]
            publish_reviewed(settings["data"], draft, review_source, key)
            from huggingface_hub import HfApi
            from vlade.storage.assets import refresh_registry
            revision = HfApi().model_info(REGISTRY_REPO).sha
            refresh_registry(run / "published-annotation-registry.json", revision=revision)
            asset_record(key, run / "published-annotation-registry.json")
            print("ANNOTATIONS_REGISTRY_REVISION:", revision)
            print("Use a NEW training run group with this registry revision; retain the annotation draft.")
        return
    # All external inputs, including cached paths, use the existing HF verifier.
    for reference in (settings["data"], settings["annotations"]):
        resolve_input(reference)
    from vlade.legacy.plan_cot.data.teacher_plan_annotations import validate_reviewed
    annotations_root = resolve_input(settings["annotations"])
    review = json.loads((annotations_root / "manifest.json").read_text(encoding="utf-8"))["annotation_source"]
    validate_reviewed(settings["data"], annotations_root, review)
    resolve_input(settings["initial_checkpoint"])
    from vlade.training.fft import FFTConfig, preflight, train
    from vlade.training.teacher_inference import teacher_identity
    from vlade.data.episode_data import EpisodeData

    teacher_identity(settings["initial_checkpoint"], EpisodeData(resolve_input(settings["data"])))
    raw = build_fft_config(settings, output, pilot=pilot)
    initial = asset_record(settings["initial_checkpoint"][10:], registry)
    raw["initial_checkpoints"]["ava_vla"].update(repository=initial["repo_id"], revision=initial["revision"],
                                                               local_directory=settings["initial_checkpoint"])
    path = run / ("pilot-fft.json" if pilot else "main-fft.json")
    _write_immutable(path, raw)
    cfg = FFTConfig.from_json(path, "ava_vla")
    checkpoints = sorted(output.glob("step-*"))
    if revision and not checkpoints:
        download_checkpoint(REGISTRY_REPO, f"{selected_group}/{output.name}", output, alias="last", revision=revision)
        checkpoints = sorted(output.glob("step-*"))
    if checkpoints:
        from dataclasses import replace
        cfg = replace(cfg, resume=checkpoints[-1])
    if action == "preflight":
        preflight(cfg)
        HubCheckpoints(output)
        print("Plan annotation coverage, T1 initialization and private Hub preflight passed.")
        return
    if action == "train" and not checkpoints:
        pilot_output = run / "T1_plan_pilot"
        if not (pilot_output / "step-0000001" / "COMPLETE").exists():
            raise RuntimeError("Complete the separate one-step pilot before main training")
        export_pin(pilot_output, group + "-pilot", 1)
    if not checkpoints or int(checkpoints[-1].name.removeprefix("step-")) < cfg.max_steps:
        train(cfg)
    else:
        preflight(cfg)
    export_pin(output, selected_group, cfg.max_steps)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "src/vlade/legacy/plan_cot/configs/experiments/teacher_plan_training.json")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--action", choices=ACTIONS, default="plan")
    parser.add_argument("--overrides", type=json.loads, default={})
    parser.add_argument("--resume-revision")
    parser.add_argument("--review-source", choices=("human_reviewed", "externally_reviewed"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,73}", args.run_group):
        parser.error("Use a safe run group of at most 74 characters")
    if args.resume_revision and not re.fullmatch(r"[0-9a-f]{40}", args.resume_revision):
        parser.error("Resume requires an immutable HF commit")
    if args.resume_revision and args.action == "pilot":
        parser.error("resume-revision restores the main T1_plan run; do not apply it to the pilot")
    settings = load_settings(args.config, args.overrides)
    work = args.workspace.resolve()
    run = work / "runs" / args.run_group
    if args.action == "plan":
        print(json.dumps({"settings": settings, "output": str(run), "hub_repo": REGISTRY_REPO,
            "pilot": {"steps": 1, "hub_run": args.run_group + "-pilot/T1_plan_pilot"},
            "main": {"hub_run": args.run_group + "/T1_plan", "initializes_from": settings["initial_checkpoint"]},
            "required": "reviewed HF Plan annotations; this command does not resolve or download inputs"}, indent=2))
        return
    if args.action == "setup":
        prepare(ROOT, work, "ava")
        return
    if not args.worker:
        env = runtime_environment(work)
        env["PYTHONPATH"] = str(ROOT / "src")
        # An explicitly injected token takes precedence over any runtime token cache.
        command = [str(locate(ROOT, work, "ava")), "-B", str(CLI), *(argv if argv is not None else sys.argv[1:]), "--worker"]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    _worker(settings, work, run, args.run_group, args.action, args.resume_revision, args.review_source)


if __name__ == "__main__":
    main()
