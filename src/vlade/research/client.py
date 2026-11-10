"""Small Colab-facing command wrapper; experiment logic lives in project modules."""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import os
import threading
import time

_RELEASE_TIMERS = {}


def job(
    config: str | Path,
    *,
    action: str = "plan",
    workspace: str | Path,
    overrides: dict | None = None,
) -> None:
    root = Path(__file__).resolve().parents[3]
    if action in {"setup", "run"}:
        from .configuration import load

        assignments = [key + "=" + json.dumps(value) for key, value in (overrides or {}).items()]
        settings = load(Path(config), assignments)
        budget = settings.get("budget", {})
        if budget.get("release_colab_on_deadline") and settings["run_group"] not in _RELEASE_TIMERS:
            try:
                from google.colab import runtime
            except ImportError:
                pass
            else:
                session = Path(workspace) / "sessions" / (settings["run_group"] + ".json")
                if session.is_file():
                    deadline = json.loads(session.read_text())["deadline_unix"]
                else:
                    deadline = (
                        float(os.environ.get("VLADE_SESSION_STARTED_UNIX", time.time()))
                        + budget["hours"] * 3600
                    )
                timer = threading.Timer(max(0, deadline - time.time()), runtime.unassign)
                timer.daemon = True
                timer.start()
                _RELEASE_TIMERS[settings["run_group"]] = timer
    command = [
        sys.executable,
        str(root / "scripts/research/run.py"),
        "--config",
        str(config),
        "--action",
        action,
        "--workspace",
        str(workspace),
    ]
    for key, value in (overrides or {}).items():
        command.extend(["--set", key + "=" + json.dumps(value)])
    subprocess.run(command, cwd=root, check=True)
