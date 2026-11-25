"""Run native QwenVLA distillation from the same config in Colab and the CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid

from vlade.distillation.teacher_plan import plan_token_budget
from vlade.runtime.environment import locate, prepare, runtime_environment
from vlade.storage.assets import REGISTRY_REPO, asset_record, read_registry
from vlade.utils.source_identity import source_identity
from vlade.training.memory import validate_memory_profile
from vlade.training.residency import validate_teacher_residency

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "scripts/train/run_distillation.py"
HUB = ROOT / "scripts/train/hub_checkpoints.py"
ACTIONS = ("plan", "setup", "preflight", "probe", "pilot", "train", "retry")


def new_run_group(prefix: str) -> str:
    """Keep independently launched notebook runs from sharing a Hub destination."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def _arguments(config: Path, workspace: Path, run_group: str, action: str, overrides: dict,
               resume_revisions: dict) -> list[str]:
    return [str(CLI), "--config", str(config.resolve()), "--workspace", str(workspace.resolve()),
            "--run-group", run_group, "--action", action, "--overrides", json.dumps(overrides),
            "--resume-revisions", json.dumps(resume_revisions)]


def job(config: str | Path, *, workspace: str | Path, run_group: str, action: str = "plan",
        overrides: dict | None = None, resume_revisions: dict | None = None, colab: bool = False) -> None:
    """Notebook client. Only the launched process receives a Colab secret."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if colab and action != "plan":
        from google.colab import userdata

        env["HF_TOKEN"] = userdata.get("HF_TOKEN")
    args = _arguments(Path(config), Path(workspace), run_group, action, overrides or {}, resume_revisions or {})
    command = [sys.executable, "-B", *args]
    if not colab:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    env["PYTHONIOENCODING"] = "utf-8"
    with subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8") as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            raise


def load_settings(config: Path, overrides: dict) -> tuple[dict, dict[str, dict]]:
    """Apply dotted config overrides, then expand the requested experiment variants."""
    settings = json.loads(config.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        target = settings
        *sections, name = key.split(".")
        for section in sections:
            target = target.get(section)
            if not isinstance(target, dict):
                raise ValueError(f"Unknown config section: {key}")
        if name not in target:
            raise ValueError(f"Unknown config key: {key}")
        target[name] = value
    workflow = settings.pop("workflow")
    variants = workflow["variant"]
    if variants not in ("gt_only", "gt_kd", "both"):
        raise ValueError("workflow.variant must be gt_only, gt_kd or both")
    initialization = settings.get("initialization")
    if initialization not in {"student_checkpoint", "pretrained_backbone"}:
        raise ValueError("This workflow expects native S0 or pretrained backbone initialization")
    if initialization == "pretrained_backbone" and variants != "gt_only":
        raise ValueError("Action KD requires a complete native S0 checkpoint")
    config_path = ROOT / settings["model_config"]
    if json.loads(config_path.read_text(encoding="utf-8"))["student"]["schema_version"] != 2:
        raise ValueError("This workflow requires a schema 2 QwenVLA model config")
    for key in ("max_steps", "accumulation_steps", "save_steps", "validation_queries"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["save_steps"] > settings["max_steps"]:
        raise ValueError("save_steps cannot exceed max_steps")
    for key in ("learning_rate", "weight_decay"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if settings["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    revision = workflow.get("registry_revision")
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("registry_revision must be an immutable HF commit")
    kd = settings.get("distillation", {})
    if variants != "gt_only":
        tokens = plan_token_budget(kd)
        if kd["teacher_mode"] not in ("inference", "cache"):
            raise ValueError("teacher_mode must be inference or cache")
        for key in ("task_weight", "action_kd_weight"):
            value = kd[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        if kd["action_kd_weight"] == 0:
            raise ValueError("action_kd_weight must be positive")
        if tokens and variants == "both":
            raise ValueError("Use separate runs for CoT and GT-only; the input prompts are different")
    # Validate only selected variants: GT-only carries no online Teacher.
    validation_settings = settings if variants != "gt_only" else {k: v for k, v in settings.items() if k != "distillation"}
    validate_memory_profile(validation_settings)
    if variants != "gt_only":
        validate_teacher_residency(kd)
    selected = {}
    if variants in ("gt_only", "both"):
        selected["gt_only"] = {key: value for key, value in settings.items() if key != "distillation"}
    if variants in ("gt_kd", "both"):
        selected["gt_kd"] = settings
    return workflow, selected


def prepare_inputs(workspace: Path, run_root: Path, workflow: dict, variants: dict) -> Path:
    """Materialize just the registry. The trainer verifies and resolves every input."""
    from vlade.research.registry import prepare_registry
    from vlade.storage.checkpoint_inputs import pin_checkpoint_inputs

    revision = workflow["registry_revision"]
    source = (prepare_registry(ROOT, workspace, revision) if revision else
              Path(os.environ.get("VLADE_ASSET_REGISTRY", ROOT / "configs/storage/hf_assets.lock.json")))
    references = set()
    for settings in variants.values():
        references.update((settings["data"], settings["initial_checkpoint"]))
        kd = settings.get("distillation")
        if kd:
            key = "teacher_checkpoint" if kd["teacher_mode"] == "inference" else "teacher_cache"
            references.add(kd[key])
    keys = {reference.removeprefix("hfasset://") for reference in references if reference.startswith("hfasset://")}
    pins = {key: value for key, value in workflow["checkpoint_pins"].items() if key in keys}
    registry = run_root / "inputs.json"
    pin_checkpoint_inputs(read_registry(source), pins, registry)
    for key in sorted(keys):
        record = asset_record(key, registry)
        print(f"{key} | {record['repo_id']} | {record['revision']}", flush=True)
    return registry


def _checkpoint_location(name: str, settings: dict, run_group: str) -> tuple[str, str]:
    if name == "gt_kd":
        return run_group, "S2" if plan_token_budget(settings["distillation"]) else "S1"
    if settings.get("initialization") == "pretrained_backbone":
        return run_group, "S0"
    return f"{run_group}-{name}", name


def _run_variant(name: str, settings: dict, run_root: Path, run_group: str, action: str,
                 teacher_python: Path | None, revision: str | None) -> None:
    if action == "probe":
        from vlade.training.teacher_inference import probe_teacher_plans

        if teacher_python is None or not plan_token_budget(settings.get("distillation", {})):
            raise ValueError("The plan probe requires an online CoT configuration")
        probe_teacher_plans({**settings["distillation"], "teacher_python": str(teacher_python)},
                            settings["data"], run_root)
        return
    group, model_name = _checkpoint_location(name, settings, run_group)
    output = run_root / model_name
    env = {**os.environ, "VLADE_HUB_REPO": REGISTRY_REPO, "VLADE_HUB_RUN_ID": group}
    hub = [sys.executable, "-B", str(HUB)]

    def invoke(command: list[str]) -> None:
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    if action == "retry":
        invoke([*hub, "retry", "--output", str(output)])
        return
    checkpoints = sorted(output.glob("step-*"))
    if revision and not checkpoints:
        invoke([*hub, "download", "--repo", REGISTRY_REPO, "--run", f"{group}/{model_name}",
                "--revision", revision, "--alias", "last", "--output", str(output)])
        checkpoints = sorted(output.glob("step-*"))
        if not checkpoints:
            raise RuntimeError("HF restore produced no checkpoint")
    config = run_root / f"{name}.json"
    config.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-B", "-m", "vlade.training.qwen_vla_task",
               "--config", str(config), "--output", str(output)]
    if teacher_python is not None and settings.get("distillation", {}).get("teacher_mode") == "inference":
        command += ["--teacher-python", str(teacher_python)]
    previous = checkpoints[-1] if checkpoints else None
    if previous:
        command += ["--resume", str(previous)]
    stop = 1 if action == "pilot" else settings["max_steps"]
    if action == "preflight" or (previous and int(previous.name.removeprefix("step-")) >= stop):
        invoke([*hub, "check", "--output", str(output)])
        invoke([*command, "--preflight"])
        if action != "preflight":
            print(f"{model_name}: checkpoint already reached step {stop}", flush=True)
    else:
        if action == "train" and previous is None:
            raise RuntimeError("Run the one-step pilot before starting full training")
        invoke([*command, "--stop-after", str(stop)])
    pending_path = output / "hub_pending.json"
    if action != "preflight" and pending_path.is_file():
        if json.loads(pending_path.read_text(encoding="utf-8"))["queue"]:
            raise RuntimeError(f"{model_name}: checkpoint saved locally; use --action retry to complete the Hub upload")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--action", choices=ACTIONS, default="plan")
    parser.add_argument("--overrides", type=json.loads, default={})
    parser.add_argument("--resume-revisions", type=json.loads, default={})
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_group):
        parser.error("Run group must use letters, numbers, underscores or hyphens")
    workflow, variants = load_settings(args.config, args.overrides)
    if args.action == "probe" and (list(variants) != ["gt_kd"] or
                                   not plan_token_budget(variants["gt_kd"]["distillation"])):
        parser.error("The plan probe requires a single online CoT configuration")
    locations = {name: _checkpoint_location(name, settings, args.run_group) for name, settings in variants.items()}
    aliases = {label: name for name, (_, label) in locations.items()}
    resumes = {}
    for label, revision in args.resume_revisions.items():
        name = aliases.get(label, label)
        if name not in variants or name in resumes or not re.fullmatch(r"[0-9a-f]{40}", revision):
            parser.error("Resume revisions must map selected models to immutable HF commits")
        resumes[name] = revision
    work = args.workspace.resolve()
    run_root = work / "runs" / args.run_group
    contract = {"workflow": workflow, "variants": variants, "run_group": args.run_group}
    if args.action == "plan":
        hub_runs = {label: f"runs/{group}/{label}" for group, label in locations.values()}
        print(json.dumps({**contract, "output": str(run_root), "hub_repo": REGISTRY_REPO,
                          "hub_runs": hub_runs}, indent=2))
        return
    source = source_identity(ROOT, require_clean=True)
    needs_teacher = variants.get("gt_kd", {}).get("distillation", {}).get("teacher_mode") == "inference"
    if args.action == "setup":
        prepare(ROOT, work, "student")
        if needs_teacher:
            prepare(ROOT, work, "ava")
        return
    if not args.worker:
        student = locate(ROOT, work, "student")
        env = runtime_environment(work)
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        hf_home = Path(os.environ.get("HF_HOME", cache_home / "huggingface"))
        env["HF_TOKEN_PATH"] = os.environ.get("HF_TOKEN_PATH", str(hf_home.expanduser() / "token"))
        env["PYTHONPATH"] = str(ROOT / "src")
        child = _arguments(args.config, work, args.run_group, args.action, args.overrides, args.resume_revisions)
        subprocess.run([str(student), "-B", *child, "--worker"], cwd=ROOT, env=env, check=True)
        return
    run_root.mkdir(parents=True, exist_ok=True)
    contract_path = run_root / "run-contract.json"
    contract.update(source)
    try:
        with contract_path.open("x", encoding="utf-8") as stream:
            json.dump(contract, stream, indent=2)
    except FileExistsError:
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Source, inputs or settings changed; choose a new run group")
    teacher = None
    if args.action != "retry":
        teacher = locate(ROOT, work, "ava") if needs_teacher else None
        os.environ["VLADE_ASSET_REGISTRY"] = str(prepare_inputs(work, run_root, workflow, variants))
    for name, settings in variants.items():
        _run_variant(name, settings, run_root, args.run_group, args.action, teacher, resumes.get(name))
