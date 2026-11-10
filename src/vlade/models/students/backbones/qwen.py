"""Native Qwen3.5 vision, merger, language loading and observation encoding."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from torch import nn
from vlade.models.students.configuration_qwen_vla import QwenVLAConfig

def verified_initial(config: QwenVLAConfig, directory: str | Path) -> Path:
    from vlade.storage.assets import resolve_input
    config.validate()
    path = resolve_input(directory)  # same HF integrity check for URI and local cache
    manifest = json.loads((path / "download_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("repository"), manifest.get("revision")) != (config.model_id, config.revision):
        raise ValueError("Initial checkpoint repository/revision differs from Student config")
    return path


def validate_backbone(backbone: nn.Module, config: QwenVLAConfig) -> None:
    config.validate()
    if (backbone.config.model_type != "qwen3_5"
            or backbone.config.text_config.hidden_size != config.hidden_size):
        raise ValueError("Expected the complete Qwen3.5 VLM backbone with matching hidden size")
    vision = backbone.config.vision_config
    if (vision.out_hidden_size != config.hidden_size
            or vision.patch_size*vision.spatial_merge_size*config.patch_grid != config.image_size):
        raise ValueError("Native merger output or image grid differs from Student config")
    if vision.deepstack_visual_indexes:
        raise ValueError("This pinned 0.8B path expects no additional deepstack vision injections")
    text = backbone.config.text_config
    if not {"linear_attention", "full_attention"}.issuperset(text.layer_types) or "full_attention" not in text.layer_types:
        raise ValueError("AVA requires the native Qwen hybrid layout with full-attention layers")


def encode_observation(backbone: nn.Module, config: QwenVLAConfig, inputs: dict[str, torch.Tensor]) -> tuple:
    """Native vision, native merger, placeholder scatter and native M-RoPE."""
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if required - inputs.keys() or set(inputs) - required - {"mm_token_type_ids"}:
        raise ValueError("Expected current native Qwen image/text inputs; no labels/cache/position overrides")
    ids, mask, pixels, grid = (inputs[key] for key in
                              ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"))
    cfg, vision = config, backbone.config.vision_config
    if (ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype != torch.long
            or mask.shape != ids.shape or not (mask == 1).all()):
        raise ValueError("Use unpadded equal-length prompt batches with all-one attention_mask")
    batch = ids.shape[0]
    if ids.min() < 0 or ids.max() >= backbone.language_model.config.vocab_size:
        raise ValueError("Token IDs are outside the Qwen vocabulary")
    if ids.shape[1] + 57 > cfg.max_sequence_length:
        raise ValueError("Native image/text/proprio/action sequence exceeds configured budget")
    edge = cfg.patch_grid*vision.spatial_merge_size
    if (grid.shape != (2*batch, 3) or grid.dtype != torch.long
            or not (grid == grid.new_tensor([1, edge, edge])).all()):
        raise ValueError("Expected exactly two current square camera grids per sample")
    patch_size = vision.in_channels*vision.temporal_patch_size*vision.patch_size**2
    if pixels.shape != (batch*2*edge**2, patch_size) or not torch.isfinite(pixels).all():
        raise ValueError("Packed image patches differ from the native Qwen image grids")
    image_mask = ids == backbone.config.image_token_id
    prior = nn.functional.pad(image_mask[:, :-1], (1, 0), value=False)
    if not ((image_mask & ~prior).sum(-1) == 2).all():
        raise ValueError("Need exactly two separate native image placeholder spans per sample")
    # Each of the two runs must have the configured number of merged camera tokens.
    for row in image_mask:
        starts = torch.where(row & ~nn.functional.pad(row[:-1], (1, 0), value=False))[0]
        ends = torch.where(row & ~nn.functional.pad(row[1:], (0, 1), value=False))[0] + 1
        if not ((ends-starts) == cfg.patch_grid**2).all():
            raise ValueError("Native image placeholder run length differs from camera grid")
    types = image_mask.long()
    if "mm_token_type_ids" in inputs and (inputs["mm_token_type_ids"].shape != ids.shape
                                          or not torch.equal(inputs["mm_token_type_ids"], types)):
        raise ValueError("Native modality IDs disagree with image placeholders")
    image_outputs = backbone.get_image_features(pixels, grid, return_dict=True)
    features = torch.cat(image_outputs.pooler_output, dim=0)
    language = backbone.get_input_embeddings()(ids)
    features = features.to(device=language.device, dtype=language.dtype)
    placeholder, _ = backbone.get_placeholder_mask(ids, inputs_embeds=language, image_features=features)
    embeddings = language.masked_scatter(placeholder, features)
    positions, _ = backbone.get_rope_index(ids, mm_token_type_ids=types,
                                                image_grid_thw=grid, attention_mask=mask)
    # No backbone.rope_deltas mutation or recurrent/KV cache across episodes.
    visual = features.reshape(batch, 2*cfg.patch_grid**2, cfg.hidden_size)
    instruction = language[~image_mask].reshape(batch, -1, cfg.hidden_size)
    return embeddings, visual, instruction, image_mask, positions



def load_processor(config: QwenVLAConfig, *, qwen_initial: str | Path = "hfasset://qwen_initial"):
    from transformers import AutoProcessor
    checkpoint = verified_initial(config, qwen_initial)
    return AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False,
        use_fast=False, size={"shortest_edge": config.image_size**2, "longest_edge": config.image_size**2})


def prepare_observation(processor: Any, config: QwenVLAConfig, *, instruction: str,
                        primary: Any, wrist: Any) -> dict[str, torch.Tensor]:
    """Two current camera images, native Qwen processor; history is the AVA state."""
    config.validate()
    if not instruction.strip():
        raise ValueError("A nonempty instruction is required")
    content = [{"type": "text", "text": f"Instruction: {instruction.strip()}. Primary camera:"},
               {"type": "image", "image": primary.convert("RGB")},
               {"type": "text", "text": "Wrist camera:"},
               {"type": "image", "image": wrist.convert("RGB")},
               {"type": "text", "text": "Predict the next eight normalized robot actions."}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=False, return_dict=True, return_tensors="pt", enable_thinking=False)
    if inputs["input_ids"].shape[1] + 57 > config.max_sequence_length:
        raise ValueError("Image/text/action token budget exceeded; do not truncate image spans")
    return dict(inputs)
