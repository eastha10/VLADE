"""Measured, conservative wall-time budgets for single-GPU FFT runs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median


class BudgetExhausted(Exception):
    """A completed optimizer boundary was saved before the session deadline."""


def pilot_profile(output: Path, *, expected_steps: int = 4) -> dict:
    """Read only the latest completed pilot, excluding the first warm-up update."""
    runs = sorted(output.glob("run-*.json"), key=lambda p: p.stat().st_mtime_ns)
    if not runs:
        raise ValueError("Run the measured pilot first")
    run = json.loads(runs[-1].read_text(encoding="utf-8"))
    if run.get("status") != "completed" or run.get("step") != expected_steps:
        raise ValueError("A fully completed timing pilot is required")
    rows = [json.loads(line) for line in (output / f"metrics-{run['run_id']}.jsonl").read_text().splitlines()]
    times = [r["optimizer_step_seconds"] for r in rows if r["step"] > 1]
    saves = [r["checkpoint_seconds"] for r in rows if "checkpoint_seconds" in r]
    evaluations = [r["validation_seconds"] for r in rows if "validation_seconds" in r]
    values = times + saves + evaluations + [run.get("startup_seconds", 0)]
    if (len(times) != expected_steps - 1 or not saves or not evaluations or
            any(not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError("Pilot timing data are incomplete or invalid")
    return {"step_seconds": max(median(times), max(times)),
            "checkpoint_seconds": max(saves), "validation_seconds": max(evaluations),
            "startup_seconds": run["startup_seconds"], "pilot_run_id": run["run_id"],
            "gpu": run["hardware"]["gpu"], "vram_bytes": run["hardware"]["vram_bytes"]}


def plan_fft(train_queries: int, profile: dict, remaining_seconds: float, *,
             accumulation: int = 8, max_epochs: int = 1, safety_factor: float = 1.3,
             downstream_reserve_seconds: float = 3600, shutdown_reserve_seconds: float = 300) -> dict:
    """Cap training at one pass or the measured time budget, whichever is smaller.

    Budget two checkpoints and two validations. Reserves include further reload,
    validation selection/cache work; they do not promise that full rollout/cache fits.
    """
    if any(type(v) is not int or v < 1 for v in (train_queries, accumulation, max_epochs)):
        raise ValueError("Queries, accumulation and epochs must be positive integers")
    values = [remaining_seconds, safety_factor, downstream_reserve_seconds, shutdown_reserve_seconds,
              *(profile[key] for key in ("step_seconds", "checkpoint_seconds", "validation_seconds", "startup_seconds"))]
    if any(not math.isfinite(v) or v <= 0 for v in values) or safety_factor < 1:
        raise ValueError("Invalid time budget/profile")
    overhead = safety_factor * (profile["startup_seconds"] + 2 * profile["checkpoint_seconds"] +
                                2 * profile["validation_seconds"])
    available = remaining_seconds - downstream_reserve_seconds - shutdown_reserve_seconds - overhead
    affordable = math.floor(available / (profile["step_seconds"] * safety_factor))
    target = math.ceil(train_queries * max_epochs / accumulation)
    steps = min(target, affordable)
    if steps < 2:
        raise ValueError("Insufficient measured budget for FFT; do not start the main run")
    interval = math.ceil(steps / 2)
    return {"optimizer_steps": steps, "save_steps": interval, "eval_steps": interval,
            "warmup_steps": min(steps - 1, max(1, math.ceil(steps * .1))),
            "train_queries_per_epoch": train_queries, "target_epochs": max_epochs,
            "effective_epochs": steps * accumulation / train_queries,
            "budget_limited": steps < target, "safety_factor": safety_factor,
            "remaining_seconds_at_plan": remaining_seconds,
            "downstream_reserve_seconds": downstream_reserve_seconds,
            "save_reserve_seconds": max(shutdown_reserve_seconds, profile["checkpoint_seconds"] * safety_factor),
            "estimated_main_seconds": overhead + steps * profile["step_seconds"] * safety_factor,
            "profile": profile}


def should_stop(remaining_seconds: float, save_reserve_seconds: float,
                next_step_seconds: float) -> bool:
    """Leave room for one conservative save before starting another update."""
    return remaining_seconds <= save_reserve_seconds + next_step_seconds * 1.3
