"""Load the native AVA LoRA/components format for inference and rollouts."""

from __future__ import annotations

import json
from pathlib import Path
import torch


def load_components(checkpoint: Path, device: torch.device) -> tuple:
    """Restore sealed adapters over their verified, pinned OpenVLA initial weights."""
    from vlade.storage.assets import resolve_input
    from vlade.training.paper_ava import _model, validate_recipe
    from vlade.training.run_control import verify_checkpoint

    verify_checkpoint(checkpoint)
    metadata = json.loads((checkpoint / "paper_ava.json").read_text())
    if metadata.get("format") != "native_lora_components_not_merged_fft":
        raise ValueError("Unsupported paper checkpoint format")
    config = metadata["identity"]["configuration"]
    validate_recipe(config)
    initial = resolve_input(config["initial_checkpoint"])
    manifest = json.loads((initial / "download_manifest.json").read_text())
    if (manifest["repository"], manifest["revision"]) != (
        config["initial_repository"],
        config["initial_revision"],
    ):
        raise ValueError("Paper initial model identity differs from its training recipe")
    components = _model({**config, "gradient_checkpointing": False}, initial, device)
    model, _, head, proprio, generator = components
    payload = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if set(parameters) != set(payload["model_adapters"]):
        raise ValueError("Paper LoRA parameter set mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            saved = payload["model_adapters"][name]
            if saved.shape != parameter.shape or not torch.isfinite(saved).all():
                raise ValueError("Invalid saved LoRA parameter: " + name)
            parameter.copy_(saved)
    for module, key in ((head, "action_head"), (proprio, "proprio"), (generator, "generator")):
        module.load_state_dict(payload[key], strict=True)
    data = resolve_input(config["data"])
    model.norm_stats["libero_plus_mixdata"] = json.loads((data / "normalization.json").read_text())
    return components
