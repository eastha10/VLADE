"""Action-slot representation KD across independently tokenized VLA backbones."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from vlade.distillation.representation_config import representation_contract, representation_settings


class ActionRepresentationKD(nn.Module):
    """Training-only projection; match (horizon, action dimension) slots, not text IDs.

    Each vector is layer-normalized without affine parameters before MSE. This
    removes the unrelated hidden-state scale of Llama2 and Qwen. Teacher is detached.
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        if min(student_dim, teacher_dim) < 1:
            raise ValueError("Representation dimensions must be positive")
        # Adding this module must not advance the Student's training RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        batch = student.shape[0]
        if (student.shape != (batch, 56, self.projection.in_features)
                or teacher.shape != (batch, 56, self.projection.out_features)
                or valid.shape != (batch, 8, 7) or valid.dtype != torch.bool
                or not valid.any()):
            raise ValueError("Expected aligned hidden [B,56,D] and nonempty bool mask [B,8,7]")
        mask = valid.reshape(batch, 56)
        if not torch.isfinite(student[mask]).all() or not torch.isfinite(teacher[mask]).all():
            raise ValueError("Representation targets must be finite at valid action slots")
        # Select valid slots first: padding cannot contaminate a projection gradient.
        with torch.autocast(device_type=student.device.type, enabled=False):
            projected = self.projection(student[mask].float())
            target = teacher.detach()[mask].float()
            return F.mse_loss(F.layer_norm(projected, (projected.shape[-1],)),
                              F.layer_norm(target, (target.shape[-1],)))


def save_representation(module: ActionRepresentationKD, path: Path, contract: dict) -> None:
    """Save alongside the policy before sealing; the deployment policy stays unchanged."""
    torch.save({"contract": contract, "state_dict": module.state_dict()}, path)


def load_representation(module: ActionRepresentationKD, path: Path, contract: dict) -> None:
    """Reject incompatible/missing projection state on resume."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["contract"] != contract:
        raise ValueError("Checkpoint representation contract differs from this run")
    module.load_state_dict(payload["state_dict"], strict=True)
