"""Research task adapters around the existing training and data implementations."""

from __future__ import annotations
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys


def dispatch(config: dict, code: Path, work: Path, output: Path) -> None:
    task = config["task"]
    if task == "prepare":
        from vlade.storage.assets import resolve_input

        for key in config["assets"]:
            print(json.dumps({"asset": key, "verified_cache": str(resolve_input(key))}), flush=True)
        return
    if task == "fft":
        from vlade.training.fft import FFTConfig, train
        from vlade.data.episode_data import write_json

        base = json.loads(
            (
                code / config.get("base_config", "configs/experiments/libero_plus_fft.json")
            ).read_text()
        )
        updates = {
            "grad_accumulation_steps" if key == "accumulation_steps" else key: value
            for key, value in config["train"].items()
        }
        base["fft_defaults"].update(
            updates, seed=config["seed"], save_optimizer_state=True, image_aug=False
        )
        base["fft_runs"][config["model"]]["output_dir"] = str(output)
        if config.get("data"):
            base["fft_defaults"]["prepared_data_root"] = config["data"]
        if config.get("checkpoint"):
            base["fft_runs"][config["model"]]["checkpoint"] = config["checkpoint"]
        path = output.parent / (output.name + "-fft-config.json")
        if path.is_file() and json.loads(path.read_text()) != base:
            raise ValueError("Existing FFT config differs; choose a new run_group")
        write_json(path, base)
        cfg = FFTConfig.from_json(path, config["model"])
        if config.get("resume"):
            from vlade.storage.assets import resolve_path

            cfg = replace(cfg, resume=resolve_path(config["resume"]))
        cfg.validate()
        train(cfg)
        if config.get("reload_check", False):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            checkpoint = output / f"step-{cfg.max_steps:07d}"
            from vlade.storage.assets import resolve_input
            from vlade.training.fft import resolve_config_assets

            resolved = resolve_config_assets(cfg, download=True)
            metrics = sorted(output.glob("metrics-*.jsonl"))
            manifests = sorted(output.glob("run-*.json"))
            if len(metrics) != 1 or len(manifests) != 1:
                raise ValueError("Bounded reload check requires one isolated invocation")
            subprocess.run(
                [
                    sys.executable,
                    str(code / "scripts/train/check_fft_checkpoint.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--data",
                    str(resolved.prepared_data_root),
                    "--config",
                    str(path),
                    "--metrics",
                    str(metrics[0]),
                    "--run-manifest",
                    str(manifests[0]),
                    "--report",
                    str(output / "reload_report.json"),
                ],
                cwd=code,
                check=True,
            )
        return
    if task == "student":
        from vlade.training.student_task import run

        student = json.loads(
            (
                code / config.get("model_config", "configs/models/student/qwen3_5_0_8b.json")
            ).read_text()
        )["student"]
        settings = {
            "method": "task_only",
            "data": config["data"],
            "initial_checkpoint": config["checkpoint"],
            "student": student,
            "output": str(output),
            "seed": config["seed"],
            **config["train"],
        }
        settings.update(config.get("distillation", {}))
        from .student import run_with_pilot

        run_with_pilot(settings, config, code)
        return
    if task == "paper":
        from vlade.training.paper_ava import train

        settings = json.loads((code / config["base_config"]).read_text())
        settings.update(config["train"], seed=config["seed"], output=str(output))
        train(
            settings,
            resume=Path(config["resume"]) if config.get("resume") else None,
            pilot=config.get("pilot", False),
        )
        return
    if task == "suite":
        from vlade.training.colab_suite_fft import execute

        base = json.loads((code / config["base_config"]).read_text())
        values = {
            "grad_accumulation_steps" if k == "accumulation_steps" else k: v
            for k, v in config["train"].items()
        }
        base["fft_defaults"].update(
            values, seed=config["seed"], image_aug=False, save_optimizer_state=True
        )
        from .suite import prepare_phase

        phase, suite_map = prepare_phase(config, output)
        phase_output = output / "pilot4" if phase == "pilot4" else output
        base["fft_runs"]["ava_vla"]["output_dir"] = str(phase_output / "checkpoints")
        path = output.parent / (output.name + "-" + phase + "-fft-config.json")
        from vlade.data.episode_data import write_json

        if path.is_file() and json.loads(path.read_text()) != base:
            raise ValueError("Existing suite config differs; choose a new run_group")
        write_json(path, base)
        settings = {
            **config["settings"],
            "suite_map": config["settings"]["suite_map"],
            "config": str(path),
            "run_root": str(output),
            "authorized": True,
            "deadline_unix": float(os.environ["VLADE_DEADLINE_UNIX"]),
            "stop_step": config["train"]["max_steps"],
            "resume": config.get("resume"),
        }
        execute(settings)
        return
    if task == "cache":
        from vlade.data.episode_data import EpisodeData
        from vlade.training.fixed_cache import validate_selection, SequentialTeacher, generate
        from vlade.storage.assets import resolve_input

        data = EpisodeData(resolve_input(config["data"]))
        selection = json.loads(Path(config["selection"]).read_text(encoding="utf-8"))
        checkpoint = validate_selection(selection, data)
        result = generate(
            data,
            SequentialTeacher(checkpoint),
            selection,
            output,
            hidden=config.get("save_hidden", False),
            max_episodes=config.get("max_episodes", 0),
        )
        if config.get("publish", False):
            if result.get("scope") != "full_train_split" or result.get("status") != "completed":
                raise ValueError("Only a completed full train cache may be published")
            from vlade.storage.migrate import publish_files
            from huggingface_hub import HfApi

            sources = {
                p.relative_to(output).as_posix(): p for p in output.rglob("*") if p.is_file()
            }
            publish_files(
                HfApi(),
                config["asset_key"],
                sources,
                repo="VLADE-konkuk/vlade-teacher-cache",
                kind="dataset",
                local_name=output.name,
            )
        return
    if task == "evaluate":
        from vlade.evaluation.rollouts import run

        run(config, output, work)
        return
    if task == "inference":
        from vlade.evaluation.inference import run

        run(config, output)
        return
    if task == "script":
        entry = (code / config["entrypoint"]).resolve()
        if not entry.is_relative_to((code / "scripts").resolve()) or not entry.is_file():
            raise ValueError("Research script must be an existing project entry point")
        args = [sys.executable, str(entry), *map(str, config.get("arguments", []))]
        subprocess.run(args, cwd=code, check=True)
        return
    raise ValueError(f"Unsupported research task: {task}")
