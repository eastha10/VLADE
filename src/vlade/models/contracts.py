"""Shared continuous-action boundary; token IDs are intentionally not shared."""

from dataclasses import dataclass

import torch


@dataclass
class PolicyOutput:
    actions: torch.Tensor  # normalized [B, 8, 7]; environment adapter denormalizes
    action_features: torch.Tensor  # semantic per-horizon features [B, 8, D]

    def validate(self) -> None:
        if self.actions.ndim != 3 or self.actions.shape[1:] != (8, 7):
            raise ValueError("actions must have shape [B,8,7]")
        if self.action_features.ndim != 3 or self.action_features.shape[:2] != self.actions.shape[:2]:
            raise ValueError("action_features must have shape [B,8,D]")
        for value in (self.actions, self.action_features):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("Policy outputs must be finite floating point")


def masked_action_l1(output: PolicyOutput, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Same objective for BC and Teacher actions; caller supplies normalization."""
    output.validate()
    if target.shape != output.actions.shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError("Target and bool validity mask must match [B,8,7]")
    if not valid.any() or not torch.isfinite(target[valid]).all():
        raise ValueError("Need at least one finite valid action target")
    return (output.actions[valid].float() - target[valid].float()).abs().mean()
