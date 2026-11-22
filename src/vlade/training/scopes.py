"""Explicit trainable-parameter selection for native QwenVLA experiments."""
from __future__ import annotations

from typing import Any


def configure_trainable_scope(model: Any, settings: dict, *, from_s0: bool) -> dict:
    """Select full, policy-only, or top decoder layers without changing schema.

    Missing options retain the historical policy-only S0 continuation and full
    pretrained initialization. Explicit settings participate in run identity via
    the trainer's unchanged settings contract, so incompatible resumes fail.
    """
    scope = settings.get("trainable_scope", "policy" if from_s0 else "full")
    count = settings.get("top_layer_count", 0)
    if scope not in ("full", "policy", "top_layers"):
        raise ValueError("trainable_scope must be full, policy, or top_layers")
    if type(count) is not int or count < 0:
        raise ValueError("top_layer_count must be a nonnegative integer")
    if scope == "top_layers":
        if not 1 <= count <= len(model.language_model.layers):
            raise ValueError("top_layers requires a valid positive top_layer_count")
    elif count:
        raise ValueError("top_layer_count is only valid with top_layers")

    model.requires_grad_(True)
    if scope != "full":
        model.backbone.requires_grad_(False)
    if scope == "top_layers":
        for layer in model.language_model.layers[-count:]:
            layer.requires_grad_(True)
        model.language_model.norm.requires_grad_(True)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    return {"trainable_scope": scope, "top_layer_count": count,
            "total_parameters": total, "trainable_parameters": trainable,
            "backbone_trainable_parameters": backbone}
