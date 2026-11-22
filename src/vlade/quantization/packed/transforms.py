"""Invertible DuQuant-style dual input rotations and zigzag redistribution.

Independent implementation of QuantVLA v4 Appendix B, Eq. 22–25. The input
diagonal is represented as a divisor, so Lambda in Eq. 22 is input_scale**-1.
"""
from __future__ import annotations
import math
import torch


def block_rotate(x: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    size = rotations.shape[-1]
    if rotations.shape != (shape[-1] // size, size, size) or shape[-1] % size:
        raise ValueError('Block rotation does not cover the input')
    blocks = x.float().reshape(-1, shape[-1] // size, size)
    with torch.autocast(x.device.type, enabled=False):
        return torch.einsum('mgk,gkj->mgj', blocks, rotations.float()).reshape(shape)


def dual_transform(weight: torch.Tensor, activation_max: torch.Tensor, *, block_size: int = 64,
                   alpha: float = .15) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if weight.ndim != 2 or activation_max.shape != (weight.shape[1],) or block_size < 1 or not 0 <= alpha <= 1:
        raise ValueError('Invalid DuQuant transform parameters')
    w = weight.detach().float()
    if not torch.isfinite(w).all() or not torch.isfinite(activation_max).all() or (activation_max < 0).any():
        raise ValueError('Non-finite transform statistics')
    size = math.gcd(weight.shape[1], block_size)
    scale = (activation_max.to(w.device).clamp_min(1e-8).pow(alpha) /
             w.abs().amax(0).clamp_min(1e-8).pow(1-alpha)).clamp(1e-4, 1e4)
    w = w * scale

    def rotations(value):
        blocks = value.reshape(value.shape[0], -1, size).permute(1, 2, 0)
        # SVD supplies a full orthonormal input basis, as in the official preprocessing.
        # Only U is needed. Avoid allocating an out_features**2 Vh for wide layers.
        u, _, _ = torch.linalg.svd(blocks, full_matrices=blocks.shape[-1] < size)
        return u

    first = rotations(w)
    w = block_rotate(w, first)
    order = w.square().mean(0).argsort(descending=True, stable=True)
    # Alternating high/low energies spreads outliers before the second rotation.
    zigzag = torch.empty_like(order)
    zigzag[::2] = order[:(len(order)+1)//2]
    zigzag[1::2] = order[(len(order)+1)//2:].flip(0)
    w = w[:, zigzag]
    second = rotations(w)
    w = block_rotate(w, second)
    return w, {'input_scale': scale, 'rotation1': first,
               'permutation': zigzag, 'rotation2': second}
