"""Dependency-free validation and provenance for action-slot representation KD."""
from __future__ import annotations

import json
import math
from pathlib import Path


def representation_settings(settings: dict) -> dict | None:
    """Validate opt-in online, plan-free S2 before resolving any external inputs."""
    value = settings.get("representation")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"weight", "loss", "alignment", "student_source"}:
        raise ValueError("Invalid representation configuration")
    weight = value["weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Representation weight must be finite and positive")
    if (value["loss"] != "normalized_mse" or value["alignment"] != "action_slots_8x7"
            or value["student_source"] != "temporal_hidden"):
        raise ValueError("Unsupported representation loss, token alignment or Student source")
    if settings.get("teacher_mode") != "inference" or settings.get("teacher_plan_max_new_tokens", 0) != 0:
        raise ValueError("Representation KD requires online T1 and no text Plan")
    return value


def representation_contract(checkpoint: Path, *, student_dim: int, student_layer: int,
                            teacher_layer: int) -> dict:
    """Read dimensions from the verified AVA config, never from a Qwen tokenizer."""
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    teacher_dim = text.get("hidden_size")
    layers = text.get("num_hidden_layers")
    if (config.get("llm_backbone_id") != "llama2-7b-pure"
            or type(teacher_dim) is not int or teacher_dim < 1
            or type(layers) is not int or layers < 1):
        raise ValueError("Representation KD requires an explicit AVA Llama2 text config")
    index = layers + 1 + teacher_layer if teacher_layer < 0 else teacher_layer
    if not 0 <= index <= layers:
        raise ValueError("Teacher representation layer is outside its text backbone")
    return {"student_dim": student_dim, "teacher_dim": teacher_dim,
            "student_layer": student_layer, "teacher_layer": teacher_layer,
            "layout": "horizon_major_action_dimension_minor", "tokens": 56,
            "source": "temporal_hidden", "projection": "linear_student_to_teacher_no_bias"}
