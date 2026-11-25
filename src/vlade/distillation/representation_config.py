"""Dependency-free validation and provenance for action-slot representation KD."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys


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
                            teacher_layer: int, teacher_python: Path | None = None) -> dict:
    """Read the effective AVA config, including HF's omitted serialization defaults.

    The trainer supplies its existing AVA interpreter so resolution uses the same
    Transformers configuration class as T1 inference. No weights are loaded.
    Standalone CPU callers may use the current interpreter.
    """
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config")
    if (config.get("llm_backbone_id") != "llama2-7b-pure"
            or config.get("model_type", "openvla") not in {"openvla", "prismatic"}
            or not isinstance(text, dict)
            or text.get("model_type", "llama") != "llama"):
        raise ValueError("Representation KD requires an AVA Llama2 text config")
    for key in ("hidden_size", "num_hidden_layers"):
        if key in text and (type(text[key]) is not int or text[key] < 1):
            raise ValueError(f"Invalid AVA Llama2 text config: {key}")
    if any(key not in text for key in ("hidden_size", "num_hidden_layers")):
        # Use project-owned configuration code; never execute a checkpoint's auto_map.
        command = (
            "import json, sys; "
            "from vlade.models.teachers.configuration_ava_vla import OpenVLAConfig; "
            "c=OpenVLAConfig.from_pretrained(sys.argv[1], local_files_only=True); "
            "print(json.dumps({'hidden_size':c.text_config.hidden_size, "
            "'num_hidden_layers':c.text_config.num_hidden_layers}))"
        )
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        result = subprocess.run([str(teacher_python or sys.executable), "-B", "-c", command,
                                 str(checkpoint.resolve())], env=env, stdout=subprocess.PIPE,
                                text=True, check=True, timeout=60)
        text = json.loads(result.stdout)
    teacher_dim = text.get("hidden_size")
    layers = text.get("num_hidden_layers")
    if (type(teacher_dim) is not int or teacher_dim < 1
            or type(layers) is not int or layers < 1):
        raise ValueError("Invalid effective AVA Llama2 text config")
    if type(student_dim) is not int or student_dim < 1 or type(student_layer) is not int:
        raise ValueError("Invalid Student representation dimension or layer")
    if type(teacher_layer) is not int:
        raise ValueError("Invalid Teacher representation layer")
    index = layers + 1 + teacher_layer if teacher_layer < 0 else teacher_layer
    if not 0 <= index <= layers:
        raise ValueError("Teacher representation layer is outside its text backbone")
    return {"student_dim": student_dim, "teacher_dim": teacher_dim,
            "student_layer": student_layer, "teacher_layer": teacher_layer,
            "layout": "horizon_major_action_dimension_minor", "tokens": 56,
            "source": "temporal_hidden", "projection": "linear_student_to_teacher_no_bias"}
