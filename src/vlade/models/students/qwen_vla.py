"""Qwen VLA Student: native VLM + AVA recurrence, proprio and L1 action head."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from vlade.models.contracts import PolicyOutput
from vlade.models.students.configuration_qwen_vla import QwenVLAConfig
from vlade.models.students.backbones.qwen import encode_observation, validate_backbone, verified_initial
from vlade.models.students.components.action_heads import L1RegressionActionHead
from vlade.models.students.components.projectors import ProprioProjector
from vlade.models.students.components.attn_weight_generator import AttentionWeightGenerator
from vlade.models.students.components.qwen_attention import language_forward

@dataclass
class QwenVLAOutput(PolicyOutput):
    recurrent_state: torch.Tensor  # [B,56,D]; explicitly owned by the episode caller
    visual_weights: torch.Tensor  # [B,2*patch_grid**2,1], camera order primary then wrist




class QwenVLA(nn.Module):
    """Current Qwen camera tokens + proprio + 56 recurrent action placeholders.

    Pass state=None at episode/sequence start. Carry returned state only within
    that episode; detach at the caller's TBPTT boundary. Targets never enter forward.
    """
    def __init__(self, backbone: nn.Module, config: QwenVLAConfig):
        super().__init__()
        validate_backbone(backbone, config)
        self.student_config, self.backbone = config, backbone
        self.language_model.config.use_cache = False
        self.language_model.config._attn_implementation = "eager"
        self.proprio_projector = ProprioProjector(config.hidden_size, 8)
        self.action_head = L1RegressionActionHead(config.hidden_size, config.action_head_hidden_dim, 7)
        self.ava = AttentionWeightGenerator(config.ava_embed_dim,
            image_shape=(config.patch_grid, config.patch_grid, config.hidden_size),
            action_shape=(8, 7, config.hidden_size), text_dim=config.hidden_size, num_images=2,
            max_steps=config.attention_max_steps, head_type="softmaxscore", score_config=(1.9, .1, 0.),
            sink_ids=None, sink_weight=1.)

    @property
    def language_model(self) -> nn.Module:
        return self.backbone.language_model

    @property
    def vision_backbone(self) -> nn.Module:
        return self.backbone.visual  # includes Qwen's own learned merger

    @property
    def modulated_layers(self) -> list[int]:
        return [i for i, kind in enumerate(self.language_model.config.layer_types) if kind == "full_attention"]

    @classmethod
    def from_pretrained(cls, config: QwenVLAConfig, *, qwen_initial: str | Path = "hfasset://qwen_initial",
                        dtype: torch.dtype = torch.float32) -> "QwenVLA":
        from transformers import Qwen3_5Model
        checkpoint = verified_initial(config, qwen_initial)
        backbone, info = Qwen3_5Model.from_pretrained(str(checkpoint), local_files_only=True,
            trust_remote_code=False, dtype=dtype, attn_implementation="eager", output_loading_info=True)
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise ValueError("Native Qwen checkpoint loading left missing/mismatched vision or language weights")
        return cls(backbone, config).to(dtype=dtype)

    def encode_observation(self, inputs: dict[str, torch.Tensor]) -> tuple:
        return encode_observation(self.backbone, self.student_config, inputs)

    def forward(self, inputs: dict[str, torch.Tensor], proprio: torch.Tensor,
                state: torch.Tensor | None = None, *, batch_idx: int | None = None) -> QwenVLAOutput:
        embeddings, visual, instruction, image_mask, positions = self.encode_observation(inputs)
        batch, _, dim = embeddings.shape
        if proprio.shape != (batch, 8) or not torch.isfinite(proprio).all():
            raise ValueError("Expected finite normalized proprio [B,8]")
        if state is not None and (state.shape != (batch, 56, dim) or state.device != embeddings.device
                                  or not state.is_floating_point() or not torch.isfinite(state).all()):
            raise ValueError("Expected finite same-episode recurrent state [B,56,D] on model device")
        action_slots, (omega, _) = self.ava(visual, state, instruction,
            batch_idx=(0 if batch_idx is None else batch_idx) if self.training else None)
        proprio_token = self.proprio_projector(proprio.to(embeddings.dtype)).unsqueeze(1)
        embeddings = torch.cat((embeddings, proprio_token, action_slots), dim=1)
        image_mask = nn.functional.pad(image_mask, (0, 57), value=False)
        # Preserve native image M-RoPE, then continue positions for proprio/actions.
        next_position = positions.amax(dim=(0, 2)) + 1
        tail = next_position[None, :, None] + torch.arange(57, device=positions.device)[None, None, :]
        positions = torch.cat((positions, tail.expand(3, -1, -1)), dim=-1)
        valid = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)
        hidden, temporal = language_forward(self.language_model, embeddings, valid, omega,
            temporal_layer=self.student_config.temporal_feature_layer, visual_mask=image_mask, position_ids=positions)
        action_hidden = hidden[:, -56:]
        output = QwenVLAOutput(self.action_head.predict_action(action_hidden),
            action_hidden.reshape(batch, 8, -1), temporal[:, -56:], omega)
        output.validate()
        return output

    def save(self, directory: Path) -> None:
        """Save complete native VLM plus AVA/proprio/action; no episode state."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.backbone.save_pretrained(directory / "backbone", safe_serialization=True)
        torch.save({name: value for name, value in self.state_dict().items()
                    if not name.startswith("backbone.")}, directory / "components.pt")
        (directory / "student.json").write_text(json.dumps(asdict(self.student_config), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "QwenVLA":
        from transformers import Qwen3_5Model
        directory = Path(directory)
        config = QwenVLAConfig(**json.loads((directory / "student.json").read_text(encoding="utf-8")))
        config.validate()
        backbone = Qwen3_5Model.from_pretrained(directory / "backbone", local_files_only=True,
            dtype="auto", attn_implementation="eager", trust_remote_code=False)
        model = cls(backbone, config).to(dtype=next(backbone.parameters()).dtype)
        state = torch.load(directory / "components.pt", map_location="cpu", weights_only=True)
        expected = {name for name in model.state_dict() if not name.startswith("backbone.")}
        if set(state) != expected:
            raise ValueError("Student checkpoint component keys differ from the selected architecture")
        model.load_state_dict({**model.state_dict(), **state}, strict=True)
        return model
