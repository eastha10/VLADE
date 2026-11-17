"""LIBERO+ rollout evaluation for the experimental S0 fake-quantization pilot."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
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


def run_rollouts(plan: RolloutPlan, predict: Callable, stats: dict) -> dict:
    """Run one trial per variant and count environment-reported task success."""
    trials = []
    for index, task_id in enumerate(plan.task_ids, 1):
        print(f"Rollout {index}/{len(plan.task_ids)} (task {task_id})", flush=True)
        task_seed = plan.seed + task_id
        random.seed(task_seed)
        np.random.seed(task_seed % 2**32)
        try:
            # Official LIBERO+ initial states include NumPy arrays serialized by older PyTorch.
            numpy_types = [np.ndarray, np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType,
                           (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct")]
            with torch.serialization.safe_globals(numpy_types):
                initial = plan.suite.get_task_init_states(task_id)[0]
            initial_digest = hashlib.sha256(np.asarray(initial).tobytes()).hexdigest()
            env = plan.env_factory(bddl_file_name=plan.suite.get_task_bddl_file_path(task_id),
                                   camera_heights=256, camera_widths=256, horizon=MAX_STEPS + SETTLE_STEPS)
            try:
                env.seed(task_seed)
                env.reset()
                observation = env.set_init_state(initial)
                for _ in range(SETTLE_STEPS):
                    observation, _, _, _ = env.step([0.] * 6 + [-1.])
                success, terminal, steps, state = False, False, 0, None
                while steps < MAX_STEPS and not (success or terminal):
                    primary = Image.fromarray(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]))
                    wrist = Image.fromarray(np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1]))
                    proprio = np.concatenate((
                        observation["robot0_eef_pos"],
                        plan.quat_to_axisangle(observation["robot0_eef_quat"].copy()),
                        observation["robot0_gripper_qpos"],
                    ))
                    actions, state = predict(plan.suite.get_task(task_id).language, primary, wrist,
                                             normalize(proprio, stats["proprio"]), state)
                    for action in _action_chunk(actions, stats["action"])[:MAX_STEPS - steps]:
                        observation, _, terminal, _ = env.step(action.tolist())
                        steps += 1
                        success = bool(env.check_success())
                        if success or terminal:
                            break
            finally:
                env.close()
        except Exception as exc:
            return {"status": "error", "suite": SUITE, "reason": f"task {task_id}: {exc}",
                    "trials": trials, "requested": len(plan.task_ids)}
        trials.append({"task_id": task_id, "init_index": 0, "init_sha256": initial_digest,
                       "seed": task_seed, "steps": steps, "success": success})
    successes = sum(trial["success"] for trial in trials)
    return {"status": "complete", "suite": SUITE, "max_steps": MAX_STEPS, "settle_steps": SETTLE_STEPS,
            "successes": successes, "episodes": len(trials), "success_rate": successes / len(trials),
            "trials": trials}
