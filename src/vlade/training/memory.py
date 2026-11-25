"""Opt-in Colab memory controls without changing model or optimizer precision."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def validate_memory_profile(settings: dict) -> dict | None:
    """Validate the single-query L4 profile and its original effective batch."""
    profile = settings.get("memory_profile")
    if profile is None:
        return None
    expected = {"name", "parameter_dtype", "autocast_dtype", "attention_backend",
                "gradient_checkpointing", "microbatch_size", "effective_batch_size", "reserve_gib"}
    if not isinstance(profile, dict) or set(profile) != expected:
        raise ValueError("memory_profile must declare all supported memory controls")
    if (profile["name"] != "colab_l4_24gb" or profile["parameter_dtype"] != "float32"
            or profile["autocast_dtype"] != "bfloat16"
            or profile["attention_backend"] not in {"eager", "sdpa"}
            or profile["gradient_checkpointing"] is not True):
        raise ValueError("L4 requires FP32 parameters, BF16 autocast, checkpointing and eager/sdpa")
    if (type(profile["microbatch_size"]) is not int or profile["microbatch_size"] != 1
            or type(profile["effective_batch_size"]) is not int
            or profile["effective_batch_size"] != settings["accumulation_steps"]):
        raise ValueError("L4 effective batch must equal microbatch 1 times accumulation_steps")
    reserve = profile["reserve_gib"]
    if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("memory_profile.reserve_gib must be finite and nonnegative")
    kd = settings.get("distillation", {})
    if kd.get("teacher_mode") == "inference" and kd.get("teacher_residency") != "exclusive":
        raise ValueError("L4 online Teacher requires exclusive residency; use verified cache otherwise")
    return profile


def adamw_storage(model: Any) -> dict[str, int]:
    """Conservative FP32 AdamW weights + gradients + two moments, no activations."""
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameter_bytes": weights, "gradient_bytes": trainable * 4,
            "optimizer_bytes": trainable * 8, "minimum_training_bytes": weights + trainable * 12}


def cuda_memory(device: Any) -> dict[str, int]:
    """Allocator and device-wide counters; device use includes other processes."""
    import torch

    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_free_bytes": free, "device_total_bytes": total,
            "device_used_bytes": total - free}


def record_memory(path: Path, device: Any, phase: str, **context: Any) -> dict:
    """Append measurements, including optimizer, validation and offload boundaries."""
    from datetime import datetime, timezone

    row = {"phase": phase, "at_utc": datetime.now(timezone.utc).isoformat(),
           **context, **cuda_memory(device)}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def configure_qwen_memory(model: Any, settings: dict) -> dict:
    """Use the supported public HF backend switch, retaining AVA additive masks."""
    profile = validate_memory_profile(settings)
    backend = profile["attention_backend"] if profile else "eager"
    if profile:
        model.backbone.set_attn_implementation(backend)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.language_model.config.use_cache = False
    return {"attention_backend": backend, "gradient_checkpointing": True,
            "checkpoint_use_reentrant": False, "microbatch_size": 1,
            "effective_batch_size": settings["accumulation_steps"],
            "parameter_dtype": "float32", "autocast_dtype": "bfloat16", "adamw_foreach": False}
