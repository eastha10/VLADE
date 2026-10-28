"""Separate native Qwen3.5 VLM + continuous action queries, without Teacher forks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vlade.models.contracts import PolicyOutput


@dataclass(frozen=True)
class StudentConfig:
    model_id: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "2fc06364715b967f1860aea9cf38778875588b17"
    hidden_size: int = 1024
    head_size: int = 256
    head_heads: int = 4
    history_queries: int = 2
    query_interval_steps: int = 8
    max_sequence_length: int = 8192
    max_pixels_per_image: int = 256 * 256
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1 or self.model_id != "Qwen/Qwen3.5-0.8B":
            raise ValueError("Unsupported Student configuration")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("Student revision must be an immutable commit")
        values = (self.hidden_size, self.head_size, self.head_heads, self.history_queries,
                  self.query_interval_steps, self.max_sequence_length, self.max_pixels_per_image)
        if any(type(value) is not int or value < 1 for value in values) or self.head_size % self.head_heads:
            raise ValueError("Invalid dimensions or input budget")

    @classmethod
    def from_json(cls, path: Path) -> "StudentConfig":
        config = cls(**json.loads(path.read_text(encoding="utf-8"))["student"])
        config.validate()
        return config


class ContinuousActionHead(nn.Module):
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.context = nn.Linear(config.hidden_size, config.head_size)
        self.proprio = nn.Linear(8, config.head_size)
        self.queries = nn.Parameter(torch.randn(8, config.head_size) * 0.02)
        self.attention = nn.MultiheadAttention(config.head_size, config.head_heads, batch_first=True, dropout=0)
        self.norm = nn.LayerNorm(config.head_size)
        self.action = nn.Sequential(nn.Linear(config.head_size, config.head_size), nn.GELU(),
                                    nn.Linear(config.head_size, 7))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, proprio: torch.Tensor) -> PolicyOutput:
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Expected [B,L,D] hidden and binary [B,L] attention mask")
        if not mask.bool().any(dim=1).all():
            raise ValueError("Every sample needs valid context")
        if proprio.shape != (hidden.shape[0], 8) or not torch.isfinite(proprio).all():
            raise ValueError("Expected finite proprio [B,8]")
        # Mask before projection too: padded NaNs must not poison attention logits.
        memory = self.context(hidden.masked_fill(~mask.bool().unsqueeze(-1), 0))
        memory = torch.cat((memory, self.proprio(proprio).unsqueeze(1)), dim=1)
        padding = torch.cat((~mask.bool(), torch.zeros((hidden.shape[0], 1), dtype=torch.bool, device=mask.device)), dim=1)
        queries = self.queries.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        features, _ = self.attention(queries, memory, memory, key_padding_mask=padding, need_weights=False)
        features = self.norm(features + queries)
        output = PolicyOutput(self.action(features), features)
        output.validate()
        return output


class QwenStudent(nn.Module):
    """Backbone returns fused context; head owns independent horizon queries.

    No language generation, GT action tokens, or past_key_values on this path.
    Rebuild each causal history window; no state can leak across episodes.
    """

    def __init__(self, backbone: nn.Module, config: StudentConfig):
        super().__init__()
        config.validate()
        if backbone.config.text_config.hidden_size != config.hidden_size:
            raise ValueError("Backbone hidden dimension differs from Student contract")
        self.backbone, self.student_config = backbone, config
        self.head = ContinuousActionHead(config)

    @classmethod
    def from_pretrained(cls, config: StudentConfig, *, local_files_only: bool = True,
                        dtype: torch.dtype = torch.float32, directory: Path | None = None) -> "QwenStudent":
        from transformers import Qwen3_5Model

        config.validate()
        backbone = Qwen3_5Model.from_pretrained(str(directory) if directory else config.model_id, revision=config.revision,
                                               local_files_only=local_files_only, dtype=dtype,
                                               attn_implementation="sdpa", trust_remote_code=False)
        return cls(backbone, config).to(dtype=dtype)

    def forward(self, inputs: dict[str, torch.Tensor], proprio: torch.Tensor) -> PolicyOutput:
        if {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"} - inputs.keys():
            raise ValueError("Native Qwen processor image/text inputs are required")
        if set(inputs) - {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}:
            raise ValueError("Unexpected input: GT action labels/cache/position override are not accepted")
        if inputs["input_ids"].shape[1] > self.student_config.max_sequence_length:
            raise ValueError("Input budget exceeded; do not truncate image token spans")
        if inputs["image_grid_thw"].shape != (inputs["input_ids"].shape[0] * self.student_config.history_queries * 2, 3):
            raise ValueError("Expected two cameras per history query for each sample")
        output = self.backbone(**inputs, use_cache=False, return_dict=True)
        return self.head(output.last_hidden_state, inputs["attention_mask"], proprio)

    def save(self, directory: Path) -> None:
        """Self-contained native backbone and strict head/config; refuses overwrite."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.backbone.save_pretrained(directory / "backbone", safe_serialization=True)
        torch.save(self.head.state_dict(), directory / "action_head.pt")
        (directory / "student.json").write_text(json.dumps(asdict(self.student_config), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "QwenStudent":
        from transformers import Qwen3_5Model

        directory = Path(directory)
        config = StudentConfig(**json.loads((directory / "student.json").read_text(encoding="utf-8")))
        backbone = Qwen3_5Model.from_pretrained(directory / "backbone", local_files_only=True,
                                               attn_implementation="sdpa", trust_remote_code=False)
        model = cls(backbone, config)
        model.head.load_state_dict(torch.load(directory / "action_head.pt", map_location="cpu", weights_only=True), strict=True)
        return model


@dataclass
class HistoryFrame:
    timestep: int
    primary: Any  # PIL image, already oriented by the dataset/environment adapter
    wrist: Any


def prepare_inputs(processor: Any, config: StudentConfig, *, instruction: str,
                   frames: list[HistoryFrame], current_timestep: int) -> dict[str, torch.Tensor]:
    """Frames belong to one episode; causal query history, oldest first.

    Episode start repeats its earliest frame until the window is full. Camera/time
    markers distinguish both views and repeated history. No future observation is used.
    """
    config.validate()
    if not instruction.strip() or not frames or type(current_timestep) is not int or current_timestep < 0:
        raise ValueError("Instruction and nonempty causal history are required")
    times = [frame.timestep for frame in frames]
    if any(type(time) is not int or time < 0 for time in times) or times != sorted(set(times)) or times[-1] != current_timestep:
        raise ValueError("History must be strictly chronological and end at current timestep")
    expected = list(range(max(0, current_timestep - (config.history_queries - 1) * config.query_interval_steps),
                          current_timestep + 1, config.query_interval_steps))
    window = frames[-config.history_queries:]
    if [frame.timestep for frame in window] != expected:
        raise ValueError("History does not match query interval/window; reset at episode start")
    window = [window[0]] * (config.history_queries - len(window)) + window
    content = [{"type": "text", "text": f"Instruction: {instruction}"}]
    for index, frame in enumerate(window):
        for camera in ("primary", "wrist"):
            content.extend([{"type": "text", "text": f"History {index}; t={frame.timestep}; camera={camera}."},
                            {"type": "image", "image": getattr(frame, camera)}])
    content.append({"type": "text", "text": "Predict the next eight normalized robot actions."})
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                          add_generation_prompt=False, return_dict=True,
                                          return_tensors="pt", enable_thinking=False)
    if inputs["input_ids"].shape[1] > config.max_sequence_length:
        raise ValueError("History exceeds token budget; reduce image budget explicitly")
    return dict(inputs)


def load_processor(config: StudentConfig, *, local_files_only: bool = True, directory: Path | None = None):
    from transformers import AutoProcessor

    config.validate()
    return AutoProcessor.from_pretrained(str(directory) if directory else config.model_id, revision=config.revision,
                                         local_files_only=local_files_only, trust_remote_code=False,
                                         size={"shortest_edge": min(65536, config.max_pixels_per_image),
                                               "longest_edge": config.max_pixels_per_image})
