"""LIBERO+ rollout evaluation for the experimental S0 fake-quantization pilot."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.core.multiarray import _reconstruct
import torch
from PIL import Image

from vlade.data.episode_data import normalize

SUITE = "libero_spatial"
VARIANT_COUNT = 2402
MAX_STEPS = 600
SETTLE_STEPS = 30


@dataclass(frozen=True)
class RolloutPlan:
    suite: Any
    env_factory: Callable
    quat_to_axisangle: Callable
    task_ids: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class RolloutQuery:
    task_id: int
    timestep: int
    instruction: str
    primary: Image.Image
    wrist: Image.Image
    proprio: np.ndarray


def prepare_rollouts(count: int, seed: int) -> RolloutPlan:
    """Load the installed simulator and select paired Spatial variants."""
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.transform_utils import quat2axisangle
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"LIBERO+ 환경을 불러올 수 없습니다. {exc}") from exc
    suite = benchmark.get_benchmark_dict()[SUITE]()
    if suite.n_tasks != VARIANT_COUNT:
        raise RuntimeError(f"LIBERO+ Spatial 변형 {VARIANT_COUNT}개가 필요합니다. 감지된 수 {suite.n_tasks}개.")
    return RolloutPlan(suite, OffScreenRenderEnv, quat2axisangle,
                       tuple(sorted(random.Random(seed).sample(range(suite.n_tasks), count))), seed)


def _action_chunk(normalized: np.ndarray, stats: dict) -> np.ndarray:
    if normalized.shape != (8, 7) or not np.isfinite(normalized).all():
        raise ValueError("Expected a finite [8, 7] action chunk")
    lo, hi, mask = (np.asarray(stats[key]) for key in ("q01", "q99", "mask"))
    physical = np.where(mask, (np.clip(normalized, -1, 1) + 1) * (hi - lo + 1e-8) / 2 + lo,
                        normalized)
    physical[:, 6] = np.where(physical[:, 6] >= 0.5, -1, 1)
    return np.clip(physical, -1, 1)


def _camera_images(observation: dict) -> tuple[Image.Image, Image.Image]:
    return tuple(Image.fromarray(np.ascontiguousarray(observation[key][::-1, ::-1]))
                 for key in ("agentview_image", "robot0_eye_in_hand_image"))


def _trace_frame(observation: dict) -> dict:
    return {
        "eef_xyz": np.asarray(observation["robot0_eef_pos"]).tolist(),
        "gripper_qpos": np.asarray(observation["robot0_gripper_qpos"]).tolist(),
        "object_xyz": {name: np.asarray(value).tolist() for name, value in observation.items()
                       if name.endswith("_pos") and not name.startswith("robot") and np.shape(value) == (3,)},
    }


def run_query_rollouts(plan: RolloutPlan, predict: Callable, stats: dict, *,
                       max_steps: int = MAX_STEPS, capture: Callable | None = None) -> dict:
    """Run paired trials with an optional observer for the executed chunk."""
    trials = []
    for index, task_id in enumerate(plan.task_ids, 1):
        print(f"Rollout {index}/{len(plan.task_ids)} (task {task_id})", flush=True)
        task_seed = plan.seed + task_id
        random.seed(task_seed)
        np.random.seed(task_seed % 2**32)
        try:
            # Official LIBERO+ initial states include NumPy arrays serialized by older PyTorch.
            numpy_types = [np.ndarray, np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType,
                           (_reconstruct, "numpy.core.multiarray._reconstruct")]
            with torch.serialization.safe_globals(numpy_types):
                initial = plan.suite.get_task_init_states(task_id)[0]
            initial_digest = hashlib.sha256(np.asarray(initial).tobytes()).hexdigest()
            env = plan.env_factory(bddl_file_name=plan.suite.get_task_bddl_file_path(task_id),
                                   camera_heights=256, camera_widths=256, horizon=max_steps + SETTLE_STEPS)
            try:
                env.seed(task_seed)
                env.reset()
                observation = env.set_init_state(initial)
                for _ in range(SETTLE_STEPS):
                    observation, _, _, _ = env.step([0.] * 6 + [-1.])
                success, terminal, steps, state = False, False, 0, None
                while steps < max_steps and not (success or terminal):
                    primary, wrist = _camera_images(observation)
                    proprio = np.concatenate((
                        observation["robot0_eef_pos"],
                        plan.quat_to_axisangle(observation["robot0_eef_quat"].copy()),
                        observation["robot0_gripper_qpos"],
                    ))
                    query = RolloutQuery(task_id, steps, plan.suite.get_task(task_id).language, primary, wrist,
                                         normalize(proprio, stats["proprio"]))
                    actions, state = predict(query, state)
                    observer = capture(query, actions) if capture is not None else None
                    if observer is not None:
                        trace = [_trace_frame(observation)]
                        frames = [(primary, wrist)]
                        executed = []
                    for action in _action_chunk(actions, stats["action"])[:max_steps - steps]:
                        observation, _, terminal, _ = env.step(action.tolist())
                        steps += 1
                        success = bool(env.check_success())
                        if observer is not None:
                            executed.append(action.tolist())
                            trace.append(_trace_frame(observation))
                            frames.append(_camera_images(observation))
                        if success or terminal:
                            break
                    if observer is not None:
                        observer(trace, frames, executed)
            finally:
                env.close()
        except Exception as exc:
            return {"status": "error", "suite": SUITE, "reason": f"task {task_id}: {exc}",
                    "trials": trials, "requested": len(plan.task_ids)}
        trials.append({"task_id": task_id, "init_index": 0, "init_sha256": initial_digest,
                       "seed": task_seed, "steps": steps, "success": success})
    successes = sum(trial["success"] for trial in trials)
    return {"status": "complete", "suite": SUITE, "max_steps": max_steps, "settle_steps": SETTLE_STEPS,
            "successes": successes, "episodes": len(trials), "success_rate": successes / len(trials),
            "trials": trials}


def run_rollouts(plan: RolloutPlan, predict: Callable, stats: dict) -> dict:
    """Keep the action-only callback for existing rollout callers."""
    def predict_query(query: RolloutQuery, state):
        return predict(query.instruction, query.primary, query.wrist, query.proprio, state)

    return run_query_rollouts(plan, predict_query, stats)
