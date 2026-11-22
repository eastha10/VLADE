"""Select lambda by held-out GT L1 from matched, sealed CoT+Action KD runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vlade.distillation.action import validate_weights
from vlade.distillation.teacher_plan import plan_token_budget
from vlade.training.run_control import verify_checkpoint


def select(checkpoints: list[Path]) -> dict:
    """Compare already produced training outputs at their common final step."""
    if len(checkpoints) < 2:
        raise ValueError("Supply at least two completed lambda runs")
    candidates = {}
    baseline = None
    for path in checkpoints:
        seal = verify_checkpoint(path)
        identity = json.loads((path / "identity.json").read_text(encoding="utf-8"))
        settings = identity["settings"]
        kd = settings.get("distillation", settings)
        if not identity.get("teacher_inference") or not plan_token_budget(kd):
            raise ValueError(f"Expected a Teacher plan KD checkpoint: {path}")
        weight = kd.pop("action_kd_weight")
        validate_weights(kd["task_weight"], weight)
        if weight in candidates:
            raise ValueError("Supply a single final checkpoint for each lambda")
        step = settings["max_steps"]
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        score = metrics["validation_l1"]
        if (seal["step"] != step or metrics.get("step") != step
                or seal["microsteps"] != step * settings["accumulation_steps"]
                or type(score) not in (int, float) or not math.isfinite(score) or score < 0):
            raise ValueError(f"Expected final-step, finite validation GT L1: {path}")
        settings.pop("output", None)
        if baseline is not None and identity != baseline:
            raise ValueError("Candidate identities differ beyond lambda and output location")
        baseline = identity
        candidates[weight] = {"lambda": weight, "validation_gt_l1": score,
                              "step": step, "checkpoint": str(path)}
    ranked = sorted(candidates.values(), key=lambda row: (row["validation_gt_l1"], row["lambda"]))
    return {"metric": "validation_gt_l1_with_teacher_plan", "selected": ranked[0], "candidates": ranked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", metavar="STEP_DIR", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(select(args.checkpoints), indent=2))


if __name__ == "__main__":
    main()
