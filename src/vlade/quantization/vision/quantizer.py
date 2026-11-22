"""Reversible vision-only weight fake quantization; no integer kernels."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
import torch
from torch import nn


def fake_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel round/clamp in float32, preserving dtype."""
    if bits not in (4, 8) or weight.ndim < 2 or not torch.isfinite(weight).all():
        raise ValueError("Require finite matrix/convolution weights and 4 or 8 bits")
    value = weight.float()
    limit = 2 ** (bits - 1) - 1
    scale = value.abs().flatten(1).amax(1).clamp_min(torch.finfo(torch.float32).tiny) / limit
    scale = scale.reshape((-1,) + (1,) * (value.ndim - 1))
    return (torch.round(value / scale).clamp(-limit, limit) * scale).to(weight.dtype)


class VisionQuantizer:
    """Only explicitly identified Prismatic featurizers are eligible."""

    def __init__(self, vision: nn.Module, *, quantize_siglip: bool = True,
                 quantize_dinov2: bool = True, quantize_patch_embed: bool = False,
                 backbone_roles: dict[str, str] | None = None) -> None:
        self.groups: dict[str, nn.Module] = {}
        roles = backbone_roles or {"featurizer": "dinov2", "fused_featurizer": "siglip"}
        for prefix, role in roles.items():
            if prefix not in {"featurizer", "fused_featurizer"} or role not in {"siglip", "dinov2"}:
                raise ValueError("Unsupported vision backbone mapping")
            if not (quantize_siglip if role == "siglip" else quantize_dinov2):
                continue
            backbone = getattr(vision, prefix, None)
            if backbone is None:
                raise ValueError(f"Requested backbone missing: {prefix}")
            for name, module in backbone.named_modules():
                linear = isinstance(module, nn.Linear) and name.startswith("blocks.") and any(
                    part in name.split(".") for part in ("attn", "mlp"))
                patch = quantize_patch_embed and name == "patch_embed.proj" and isinstance(module, nn.Conv2d)
                if linear or patch:
                    self.groups[f"{prefix}.{name}"] = module
        if not self.groups:
            raise ValueError("No eligible vision groups")
        pointers = [m.weight.data_ptr() for m in self.groups.values()]
        if len(set(pointers)) != len(pointers):
            raise ValueError("Shared quantization weights are unsupported")
        self.original = {name: m.weight.detach().cpu().clone() for name, m in self.groups.items()}

    @property
    def sizes(self) -> dict[str, int]:
        return {name: m.weight.numel() for name, m in self.groups.items()}

    @torch.no_grad()
    def restore(self) -> None:
        for name, module in self.groups.items():
            module.weight.copy_(self.original[name])

    @torch.no_grad()
    def apply(self, allocation: dict[str, int]) -> None:
        if set(allocation) - self.groups.keys() or any(b not in (4, 8) for b in allocation.values()):
            raise ValueError("Invalid vision allocation")
        self.restore()
        try:
            for name, bits in allocation.items():
                module = self.groups[name]
                module.weight.copy_(fake_quantize(self.original[name].to(module.weight.device), bits))
        except BaseException:
            self.restore()
            raise

    @contextmanager
    def temporary(self, allocation: dict[str, int]) -> Iterator[None]:
        try:
            self.apply(allocation)
            yield
        finally:
            self.restore()
