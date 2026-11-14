"""Single real-weight inference, separate from rollout or training claims."""

from __future__ import annotations
from pathlib import Path
import numpy as np


def run(config: dict, output: Path) -> dict:
    from vlade.evaluation.policies import load_policy
    from vlade.data.episode_data import write_json
    import torch

    torch.manual_seed(config["seed"])
    settings = config.get("evaluation", {})
    records = {}
    for label, checkpoint in config["checkpoints"].items():
        policy = load_policy(config["model"], checkpoint, settings)
        policy.reset(settings.get("task_suite", "libero_spatial"))
        observation = {
            "full_image": np.full((224, 224, 3), 128, dtype=np.uint8),
            "wrist_image": np.full((224, 224, 3), 64, dtype=np.uint8),
            "state": np.zeros(8, dtype=np.float32),
        }
        actions = np.asarray(
            policy.predict(observation, config.get("instruction", "pick up the object"), 0)
        )
        if actions.shape != (8, 7) or not np.isfinite(actions).all():
            raise ValueError("Single inference did not produce a finite 8x7 action chunk")
        records[label] = {
            "action_shape": list(actions.shape),
            "checkpoint": checkpoint,
            "finite": True,
        }
        del policy
    result = {"status": "single_inference_completed", "rollout_measured": False, "records": records}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inference.json", result)
    return result
