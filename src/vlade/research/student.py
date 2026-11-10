"""Keep real-query reload and overfit gates in the short Student research flow."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def run_with_pilot(settings: dict, config: dict, code: Path) -> None:
    """Run bounded pilots, verify their current contexts, then invoke the original loop."""
    from vlade.data.episode_data import write_json
    from vlade.storage.assets import resolve_input
    from vlade.training.student_task import run
    from vlade.training.run_control import pilot_context, require_reports, overfit_report

    output = Path(settings["output"])
    root = output.parent / (output.name + "-pilot")
    data = resolve_input(settings["data"])
    reports, contexts = [], []
    controls = config["pilot"]
    for phase, steps in (("reload", 1), ("overfit", controls["overfit_steps"])):
        directory = root / phase
        pilot = {
            **settings,
            "output": str(directory),
            "max_steps": steps,
            "warmup_steps": 0,
            "accumulation_steps": 1,
            "episode_limit": 1,
            "save_steps": steps,
            "eval_steps": steps,
        }
        path = root / (phase + "-config.json")
        if path.is_file() and json.loads(path.read_text()) != pilot:
            raise ValueError("Student pilot config changed; use a new run_group")
        write_json(path, pilot)
        checkpoint = directory / f"step-{steps:07d}"
        report_path = root / (phase + "-report.json")
        if not report_path.is_file():
            if checkpoint.exists():
                raise ValueError("Unverified interrupted pilot exists; use a new run_group")
            run(pilot)
            if phase == "reload":
                subprocess.run(
                    [
                        sys.executable,
                        str(code / "scripts/train/check_student_checkpoint.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--data",
                        str(data),
                        "--config",
                        str(path),
                        "--report",
                        str(report_path),
                    ],
                    check=True,
                    cwd=code,
                )
            else:
                rows = [
                    json.loads(line)
                    for log in directory.glob("metrics-*.jsonl")
                    for line in log.read_text().splitlines()
                ]
                report = overfit_report(rows, max_ratio=controls["max_ratio"])
                report["context"] = pilot_context(path, checkpoint, data, "student")
                write_json(report_path, report)
        reports.append(report_path)
        contexts.append(pilot_context(path, checkpoint, data, "student"))
    require_reports(reports, contexts)
    if controls.get("only", False):
        return
    run(settings, resume=Path(config["resume"]) if config.get("resume") else None)
