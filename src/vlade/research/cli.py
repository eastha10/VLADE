"""One config-driven entry point for setup, planning and explicit research runs."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .configuration import ROOT, expand, load


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--action", choices=("plan", "setup", "run"), default="plan")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    raw = load(args.config, args.set)
    work = args.workspace.resolve()
    runs = work / "runs"
    config = expand(raw, code=ROOT, work=work, runs=runs, group=raw["run_group"])
    output = runs / config["run_group"] / config.get("output_name", config["task"])
    if not output.resolve().is_relative_to(runs):
        raise ValueError("Output escapes the research run root")
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "config": config,
                    "output": str(output),
                    "execution": "not_started",
                    "third_party_required": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    from vlade.utils.source_identity import source_identity

    identity = source_identity(ROOT, require_clean=True)
    from vlade.runtime.environment import runtime_environment, prepare, locate

    env = runtime_environment(work)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    session_path = work / "sessions" / (config["run_group"] + ".json")
    session_path.parent.mkdir(parents=True, exist_ok=True)
    if session_path.is_file():
        session = json.loads(session_path.read_text())
    else:
        started = float(os.environ.get("VLADE_SESSION_STARTED_UNIX", time.time()))
        session = {
            "started_unix": started,
            "deadline_unix": started + config.get("budget", {}).get("hours", 8) * 3600,
        }
        session_path.write_text(json.dumps(session), encoding="utf-8")
    env.update(
        VLADE_DEADLINE_UNIX=str(session["deadline_unix"]),
        VLADE_TRAIN_DEADLINE_UNIX=str(session["deadline_unix"]),
        VLADE_SAVE_RESERVE_SECONDS=str(config.get("budget", {}).get("save_reserve_seconds", 300)),
    )
    if time.time() + float(env["VLADE_SAVE_RESERVE_SECONDS"]) >= session["deadline_unix"]:
        raise RuntimeError("This research session budget has expired")
    hub = config.get("hub", {})
    if config["task"] in {"fft", "student", "paper", "suite"}:
        if hub.get("repo") != "VLADE-konkuk/VLADE-checkpoints" or hub.get("enabled") is not True:
            raise ValueError("Training requires the configured private VLADE checkpoint Hub")
        env.update(VLADE_HUB_REPO=hub["repo"], VLADE_HUB_RUN_ID=config["run_group"])
    os.environ.update(env)
    if config.get("registry_revision") and not args.worker:
        from .registry import prepare_registry

        registry = prepare_registry(ROOT, work, config["registry_revision"])
        env["VLADE_ASSET_REGISTRY"] = str(registry)
        os.environ["VLADE_ASSET_REGISTRY"] = str(registry)
    if args.action == "setup":
        python = prepare(ROOT, work, config["runtime"])
        print(json.dumps({"runtime": str(python), "source": identity, "status": "ready"}))
        return 0
    if args.worker:
        from vlade.data.episode_data import write_json
        from .execution import dispatch

        output.parent.mkdir(parents=True, exist_ok=True)
        phase = (
            ("pilot" if config.get("pilot") else "main")
            if config["task"] == "paper"
            else config.get("settings", {}).get("phase", "run")
        )
        settings_path = output.parent / (output.name + "-" + phase + "-settings.json")
        if settings_path.is_file() and json.loads(settings_path.read_text()) != config:
            raise ValueError("Existing run config differs; choose a new run_group")
        write_json(settings_path, config)
        dispatch(config, ROOT, work, output)
        return 0
    python = locate(ROOT, work, config["runtime"])
    command = [
        str(python),
        str(ROOT / "scripts/research/run.py"),
        "--config",
        str(args.config.resolve()),
        "--workspace",
        str(work),
        "--action",
        "run",
        "--worker",
    ]
    for override in args.set:
        command.extend(("--set", override))
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    timers = []
    for seconds, callback in [
        (
            max(0, session["deadline_unix"] - time.time() - 60),
            lambda: process.send_signal(signal.SIGINT),
        ),
        (max(0, session["deadline_unix"] - time.time()), process.kill),
    ]:
        timer = threading.Timer(seconds, callback)
        timer.daemon = True
        timer.start()
        timers.append(timer)
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        for timer in timers:
            timer.cancel()
    return 0
