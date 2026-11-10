"""Configuration for the native Qwen VLA Student."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class QwenVLAConfig:
    schema_version: int = 2
    model_id: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "2fc06364715b967f1860aea9cf38778875588b17"
    hidden_size: int = 1024
    image_size: int = 256
    patch_grid: int = 8  # per-camera spatial grid AFTER the native merger
    ava_embed_dim: int = 1024
    action_head_hidden_dim: int = 4096
    temporal_feature_layer: int = -2
    max_sequence_length: int = 2048
    attention_max_steps: int = 100000

    def validate(self) -> None:
        if self.schema_version != 2 or self.model_id != "Qwen/Qwen3.5-0.8B":
            raise ValueError("Expected native-Qwen vision Student schema 2; old external-vision checkpoints are incompatible")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("Initial weights require an immutable revision")
        values = (self.hidden_size, self.image_size, self.patch_grid, self.ava_embed_dim,
                  self.action_head_hidden_dim, self.max_sequence_length, self.attention_max_steps)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("Student dimensions and budgets must be positive integers")
        if self.ava_embed_dim % 128 or self.max_sequence_length <= 2*self.patch_grid**2 + 57:
            raise ValueError("AVA needs head dimension 128 and room for visual/action tokens")
        if type(self.temporal_feature_layer) is not int:
            raise ValueError("Temporal feature layer must be an integer")

    @classmethod
    def from_json(cls, path: Path) -> "QwenVLAConfig":
        config = cls(**json.loads(Path(path).read_text(encoding="utf-8"))["student"])
        config.validate()
        return config
