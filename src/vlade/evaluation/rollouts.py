"""Config-driven LIBERO(+) rollouts and paired checkpoint comparisons."""

from __future__ import annotations
from collections import deque
import json
from pathlib import Path
import random
import time
import sys
import numpy as np

HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def episode(env, policy, instruction: str, settings: dict, initial_state, suite: str) -> dict:
    """Reset temporal state for every rollout and fail on simulator/policy errors."""
    from benchmarks.libero.observation import prepare_observation
    from benchmarks.libero.action import to_environment

    env.reset()
    observation = env.set_init_state(initial_state)
    policy.reset(suite)
    wait = settings.get("settle_steps", 10)
    for _ in range(wait):
        observation, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    queue = deque()
    horizon = settings.get("max_steps") or HORIZONS[suite]
    for step in range(horizon):
        if not queue:
            actions = np.asarray(
                policy.predict(prepare_observation(observation), instruction, step)
            )
            if actions.shape != (8, 7) or not np.isfinite(actions).all():
                raise ValueError("Policy must return a finite 8x7 action chunk")
            queue.extend(actions[: settings.get("open_loop_steps", 8)])
        observation, _, done, _ = env.step(to_environment(queue.popleft()).tolist())
        if done:
            return {"success": True, "environment_steps": step + 1}
    return {"success": False, "environment_steps": horizon}


def run(config: dict, output: Path, work: Path) -> dict:
    from vlade.runtime.benchmark import prepare
    from vlade.data.episode_data import write_json
    from vlade.evaluation.policies import load_policy
    from vlade.utils.source_identity import source_identity
    import torch

    settings = config["evaluation"]
    if settings.get("benchmark_revision"):
        record = json.loads((Path(sys.prefix) / "benchmark-source.json").read_text())
        if record["pin"]["commit"] != settings["benchmark_revision"]:
            raise ValueError("Evaluation benchmark revision differs from the configured protocol")
    if not config.get("checkpoints") or not settings.get("task_suites"):
        raise ValueError("Evaluation requires checkpoints and task suites")
    for key in ("rollouts_per_task", "open_loop_steps"):
        value = settings.get(key, 8 if key == "open_loop_steps" else None)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid evaluation {key}")
    if settings.get("open_loop_steps", 8) > 8:
        raise ValueError("Open-loop execution cannot exceed the eight-action chunk")
    prepare(work, settings.get("assets", "hfasset://libero_plus_assets"))
    from libero.libero import benchmark
    from benchmarks.libero.environment import create_environment

    suites = benchmark.get_benchmark_dict()
    rng = random.Random(config["seed"])
    selected = {}
    for name in settings["task_suites"]:
        suite = suites[name]()
        expected_tasks = settings.get("expected_task_counts", {}).get(name)
        if expected_tasks is not None and suite.n_tasks != expected_tasks:
            raise ValueError(
                f"Installed benchmark task count differs from the pinned {name} protocol"
            )
        count = settings.get("tasks_per_suite") or suite.n_tasks
        if count > suite.n_tasks:
            raise ValueError(f"tasks_per_suite exceeds {name} task count")
        # Reinitialize the seeded selection for each suite, matching the paired reference notebook.
        ids = sorted(random.Random(config["seed"]).sample(range(suite.n_tasks), count))
        selected[name] = (suite, ids)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "rollouts.json"
    identity = {
        "config": config,
        "source": source_identity(Path(__file__).resolve().parents[3]),
        "task_ids": {name: ids for name, (_, ids) in selected.items()},
    }
    result = {"identity": identity, "complete": False, "status": "running", "records": []}
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError("Evaluation output belongs to a different config/source")
        if result.get("complete"):
            return result
    completed = {(r["model"], r["suite"], r["task_id"], r["rollout"]) for r in result["records"]}
    expected = (
        len(config["checkpoints"])
        * sum(len(ids) for _, ids in selected.values())
        * settings["rollouts_per_task"]
    )
    started = time.monotonic()
    write_json(path, result)
    try:
        for label, checkpoint in config["checkpoints"].items():
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            policy = load_policy(config["model"], checkpoint, settings)
            for suite_name, (suite, ids) in selected.items():
                for task_id in ids:
                    task = suite.get_task(task_id)
                    states = suite.get_task_init_states(task_id)
                    trials = settings["rollouts_per_task"]
                    if trials > len(states) and not settings.get("repeat_initial_states", False):
                        raise ValueError(
                            "Requested rollouts exceed unique initial states; configure repeat_initial_states explicitly"
                        )
                    env = create_environment(
                        task, settings.get("resolution", 256), settings.get("environment_seed", 0)
                    )
                    try:
                        for trial in range(trials):
                            if (label, suite_name, task_id, trial) in completed:
                                continue
                            state_id = trial % len(states)
                            row = episode(
                                env, policy, task.language, settings, states[state_id], suite_name
                            )
                            result["records"].append(
                                {
                                    "model": label,
                                    "suite": suite_name,
                                    "task_id": task_id,
                                    "rollout": trial,
                                    "initial_state_index": state_id,
                                    **row,
                                }
                            )
                            write_json(path, result)
                    finally:
                        env.close()
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if len(result["records"]) != expected:
            raise ValueError("Incomplete rollout count")
        result.update(
            complete=True,
            status="completed",
            rollouts=expected,
            successes=sum(int(r["success"]) for r in result["records"]),
        )
        result["success_rate"] = result["successes"] / expected
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        result["invocation_seconds"] = time.monotonic() - started
        write_json(path, result)
    return result
