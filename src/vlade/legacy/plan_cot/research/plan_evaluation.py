"""Config-driven entry point for the two LIBERO+ Teacher-plan evaluations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from vlade.runtime.environment import locate, prepare, runtime_environment
from vlade.storage.assets import asset_record, read_registry
from vlade.utils.source_identity import source_identity

ROOT = Path(__file__).resolve().parents[5]
ACTIONS = ("plan", "setup", "run")


def load_settings(config: Path, overrides: dict) -> dict:
    settings = json.loads(config.read_text(encoding="utf-8"))
    unknown = overrides.keys() - settings.keys()
    if unknown:
        raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}")
    settings.update(overrides)
    if settings["mode"] not in ("alignment", "student"):
        raise ValueError("Expected mode alignment or student")
    if not re.fullmatch(r"[0-9a-f]{40}", settings["registry_revision"]):
        raise ValueError("Expected an immutable HF registry revision")
    for key, maximum in (("trials", 2402), ("max_steps", 600)):
        if type(settings[key]) is not int or not 1 <= settings[key] <= maximum:
            raise ValueError(f"{key} must be between 1 and {maximum}")
    limits = ("capture_every_queries", "captures_per_task", "plan_tokens") if settings["mode"] == "alignment" else ("lag_queries",)
    if any(type(settings[key]) is not int or not 1 <= settings[key] <= 128 for key in limits):
        raise ValueError(f"Expected positive bounded settings: {limits}")
    return settings


def output_path(workspace: Path, run_name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_name):
        raise ValueError("Use letters, digits, underscores or hyphens for the run name")
    return workspace.resolve() / "evaluations" / run_name


def _references(settings: dict) -> list[str]:
    keys = ["teacher", "data", "benchmark_assets"]
    if settings["mode"] == "student":
        keys.extend(("cot_checkpoint", "action_checkpoint"))
    return [settings[key] for key in keys]


def run_worker(settings: dict, workspace: Path, teacher_python: Path, output: Path) -> None:
    from vlade.legacy.plan_cot.evaluation.plan_rollout import run
    from vlade.research.registry import prepare_registry
    from vlade.storage.checkpoint_inputs import pin_checkpoint_inputs

    source = source_identity(ROOT, require_clean=True)
    base = read_registry(prepare_registry(ROOT, workspace, settings["registry_revision"]))
    references = _references(settings)
    keys = {ref.removeprefix("hfasset://") for ref in references if ref.startswith("hfasset://")}
    selections = settings["checkpoint_pins"]
    if (unused := selections.keys() - keys):
        raise ValueError(f"Checkpoint pin key is unused: {sorted(unused)}")
    registry = workspace / "registries" / f"{output.name}-evaluation.json"
    pin_checkpoint_inputs(base, selections, registry)
    os.environ["VLADE_ASSET_REGISTRY"] = str(registry)
    assets = {}
    for key in sorted(keys):
        record = asset_record(key, registry)
        assets[key] = {field: record.get(field) for field in ("repo_id", "revision", "prefix")}
    source["input_assets"] = assets
    report = run(settings, workspace, teacher_python, output, source)
    print(f"Evaluation {report['status']}: {output / 'results.json'}", flush=True)


def job(config: str | Path, *, workspace: str | Path, run_name: str, action: str,
        overrides: dict | None = None, colab: bool = False) -> Path:
    """Plan, prepare the repository runtimes, or launch the simulator worker."""
    work = Path(workspace).resolve()
    output = output_path(work, run_name)
    settings = load_settings(Path(config), overrides or {})
    if action == "plan":
        print(json.dumps({"settings": settings, "output": str(output)}, indent=2))
        return output
    if action == "setup":
        prepare(ROOT, work, "student_eval")
        prepare(ROOT, work, "ava")
        return output
    if action != "run":
        raise ValueError(f"Unknown action: {action}")
    if output.exists():
        raise FileExistsError(f"Choose an unused run name: {output}")
    source_identity(ROOT, require_clean=True)
    student = locate(ROOT, work, "student_eval")
    teacher = locate(ROOT, work, "ava")
    env = runtime_environment(work)
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    hf_home = Path(os.environ.get("HF_HOME", cache_home / "huggingface"))
    env["HF_TOKEN_PATH"] = os.environ.get("HF_TOKEN_PATH", str(hf_home.expanduser() / "token"))
    if colab:
        from google.colab import userdata

        env["HF_TOKEN"] = userdata.get("HF_TOKEN")
    subprocess.run([str(student), "-B", "-m", "vlade.legacy.plan_cot.research.plan_evaluation",
                    "--worker", "--settings", json.dumps(settings), "--workspace", str(work),
                    "--teacher-python", str(teacher), "--output", str(output)],
                   cwd=ROOT, env=env, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--settings", type=json.loads)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--teacher-python", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--action", choices=ACTIONS, default="plan")
    parser.add_argument("--overrides", type=json.loads, default={})
    args = parser.parse_args()
    if args.worker:
        run_worker(args.settings, args.workspace, args.teacher_python, args.output)
    elif args.config and args.run_name:
        job(args.config, workspace=args.workspace, run_name=args.run_name, action=args.action,
            overrides=args.overrides)
    else:
        parser.error("--config and --run-name are required")


if __name__ == "__main__":
    main()
