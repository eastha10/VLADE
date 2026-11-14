"""Validated, editable controls for training, cache generation and rollouts."""

from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from typing import Any

TASKS = {"prepare", "fft", "student", "paper", "suite", "cache", "evaluate", "inference", "script"}
ROOT = Path(__file__).resolve().parents[3]


def load(path: Path, overrides: list[str] = ()) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    value = copy.deepcopy(value)
    for override in overrides:
        key, raw = override.split("=", 1)
        parts = key.split(".")
        target = value
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown config section: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown config key: {key}")
        target[parts[-1]] = json.loads(raw)
    validate(value)
    return value


def validate(value: dict) -> None:
    if value.get("schema_version") != 1 or value.get("task") not in TASKS:
        raise ValueError("Unsupported research config")
    if not isinstance(value.get("runtime"), str):
        raise ValueError("A runtime specification is required")
    group = value.get("run_group", "")
    if (
        not isinstance(group, str)
        or not group
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in group
        )
    ):
        raise ValueError("run_group must be a nonempty safe identifier")
    train = value.get("train", {})
    for key in (
        "max_steps",
        "batch_size",
        "accumulation_steps",
        "save_steps",
        "validation_queries",
    ):
        if key in train and (type(train[key]) is not int or train[key] < 1):
            raise ValueError(f"train.{key} must be a positive integer")
    for key in ("eval_steps", "warmup_steps", "episode_limit"):
        if key in train and (type(train[key]) is not int or train[key] < 0):
            raise ValueError(f"train.{key} must be a nonnegative integer")
    for key in ("learning_rate", "weight_decay"):
        if key in train and (
            not isinstance(train[key], (int, float))
            or not math.isfinite(train[key])
            or train[key] < 0
            or (key == "learning_rate" and train[key] == 0)
        ):
            raise ValueError(f"Invalid train.{key}")
    if train.get("warmup_steps", 0) >= train.get("max_steps", 1):
        raise ValueError("warmup_steps must be smaller than max_steps")
    evaluation = value.get("evaluation", {})
    for key in ("rollouts_per_task", "max_steps", "open_loop_steps", "resolution"):
        if (
            key in evaluation
            and evaluation[key] is not None
            and (type(evaluation[key]) is not int or evaluation[key] < 1)
        ):
            raise ValueError(f"evaluation.{key} must be positive")
    if evaluation.get("open_loop_steps", 8) > 8:
        raise ValueError("evaluation.open_loop_steps cannot exceed the eight-action chunk")
    if "settle_steps" in evaluation and (
        type(evaluation["settle_steps"]) is not int or evaluation["settle_steps"] < 0
    ):
        raise ValueError("evaluation.settle_steps must be nonnegative")
    if evaluation.get("tasks_per_suite") is not None and (
        type(evaluation["tasks_per_suite"]) is not int or evaluation["tasks_per_suite"] < 1
    ):
        raise ValueError("tasks_per_suite must be null (all tasks) or positive")
    if type(value.get("seed", 42)) is not int:
        raise ValueError("seed must be an integer")
    if value["task"] == "student":
        pilot = value.get("pilot", {})
        if type(pilot.get("overfit_steps")) is not int or pilot["overfit_steps"] < 10:
            raise ValueError("Student pilot requires at least ten overfit steps")
        if type(pilot.get("max_ratio")) not in (int, float) or not 0 < pilot["max_ratio"] < 1:
            raise ValueError("Student pilot max_ratio must be between zero and one")
        if train.get("batch_size") != 1 or train.get("eval_steps", 0) < 1:
            raise ValueError("Student requires batch_size=1 and periodic validation")
    budget = value.get("budget", {})
    for key in ("hours", "save_reserve_seconds"):
        number = budget.get(key, 1)
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"budget.{key} must be finite and positive")
    if budget.get("save_reserve_seconds", 300) >= budget.get("hours", 8) * 3600:
        raise ValueError("Save reserve must be smaller than the session budget")


def expand(value: Any, *, code: Path, work: Path, runs: Path, group: str) -> Any:
    """Expand only declared runtime paths; no Python expressions or shell evaluation."""
    if isinstance(value, dict):
        return {
            key: expand(item, code=code, work=work, runs=runs, group=group)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand(item, code=code, work=work, runs=runs, group=group) for item in value]
    if isinstance(value, str):
        for token, replacement in {
            "{code}": code,
            "{work}": work,
            "{runs}": runs,
            "{run_group}": group,
        }.items():
            value = value.replace(token, str(replacement))
    return value
