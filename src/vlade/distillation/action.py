"""Continuous action distillation in the shared normalized action space."""
from __future__ import annotations

import math

import torch

from vlade.models.contracts import PolicyOutput, masked_action_l1


def validate_weights(task_weight: float, action_weight: float) -> None:
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (task_weight, action_weight))
            or task_weight < 0 or action_weight <= 0):
        raise ValueError("Action KD requires finite task_weight >= 0 and action_kd_weight > 0")


def action_distillation_loss(output: PolicyOutput, target: torch.Tensor, teacher: torch.Tensor,
                             valid: torch.Tensor, *, task_weight: float = 1.0,
                             action_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Mask episode padding, detach Teacher targets, and report both L1 terms."""
    validate_weights(task_weight, action_weight)
    task = masked_action_l1(output, target, valid)
    kd = masked_action_l1(output, teacher.detach(), valid)
    return task_weight * task + action_weight * kd, {
        "task_l1": float(task.detach()), "action_kd_l1": float(kd.detach())}
