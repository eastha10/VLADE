# Source: eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456
# Copyright 2026 LiAuto-DSR. Apache-2.0 and inherited MIT: AVA_LICENSE.txt.
# Teacher-local LIBERO/L1 subset, separated by role; upstream is unchanged.
"""Compose the active AVA recurrence, visual weights and original sink handling."""
from typing import ClassVar
import torch
from torch import nn
from .attention_utils import build_mlp, get_2d_position_embedding
from .attention_blocks import AttentionWeightBlock
from .attention_heads import SoftmaxScoreHead


class AttentionWeightGenerator(nn.Module):
    head_modules: ClassVar[dict[str, type[nn.Module]]] = {
        "softmaxscore": SoftmaxScoreHead,
    }

    def __init__(
        self,
        embed_dim,
        image_shape,
        action_shape,
        text_dim,
        num_images,
        max_steps=None,
        head_type="softmaxscore",
        score_config=(1.9, 0.1, 0.0),
        sink_ids=None,
        sink_weight=1,
    ):
        super().__init__()

        if head_type not in self.head_modules:
            raise ValueError("Teacher supports only the active softmaxscore AVA head")
        head_module = self.head_modules[head_type]
        head_kwargs = {"score_config": score_config} if head_type == "softmaxscore" else {}

        assert sink_ids is None or isinstance(sink_ids, (tuple, list))
        self.sink_ids = sink_ids
        self.sink_weight = sink_weight

        image_h, image_w, image_dim = image_shape
        self.image2embed = build_mlp(image_dim, embed_dim, hidden_size=embed_dim // 2)
        image_pos_embed = get_2d_position_embedding(image_h, image_w, embed_dim)
        self.register_buffer(
            "image_pos_embed",
            torch.from_numpy(image_pos_embed)
            .reshape(1, -1, embed_dim)
            .repeat(1, num_images, 1)
            .float(),
            persistent=False,
        )

        self.text2embed = build_mlp(text_dim, embed_dim, hidden_size=embed_dim // 2, out_act=True)

        action_frames, action_dim, action_feature_dim = action_shape
        self.action2embed_input = build_mlp(
            action_feature_dim,
            action_feature_dim,
            hidden_size=action_feature_dim // 2,
            out_act=False,
        )
        self.action2embed = build_mlp(
            action_feature_dim, embed_dim, hidden_size=embed_dim // 2, out_act=True
        )
        action_zeros = torch.zeros((1, action_frames * action_dim, action_feature_dim))  # (B, N, D)
        self.register_buffer("action_zeros", action_zeros, persistent=False)

        self.block = AttentionWeightBlock(embed_dim, image_shape, action_shape, num_images)
        self.head = head_module(embed_dim, max_steps=max_steps, **head_kwargs)

    def forward(
        self, image_tokens, action_tokens, text_tokens, selected_ratio=None, batch_idx=None
    ):
        B, _N, _ = image_tokens.shape
        image_tokens = self.image2embed(image_tokens)
        image_tokens = image_tokens + self.image_pos_embed

        if action_tokens is None:
            input_action_tokens = self.action_zeros.repeat(B, 1, 1)
        else:
            input_action_tokens = self.action2embed_input(action_tokens)

        action_tokens = self.action2embed(input_action_tokens)

        text_tokens = self.text2embed(text_tokens)

        image_tokens = self.block(image_tokens, action_tokens, text_tokens)
        attn_weights = self.head(
            image_tokens, selected_ratio=selected_ratio, batch_idx=batch_idx
        )  # (B, N, 1)

        if not isinstance(attn_weights, tuple):
            attn_weights = (attn_weights, attn_weights.clone())

        if self.sink_ids is not None:
            target_attn_weights = attn_weights[0]

            attn_sink = target_attn_weights[:, self.sink_ids]
            mask = (attn_sink < self.sink_weight).bfloat16()
            target_attn_weights[:, self.sink_ids] = attn_sink + mask * (
                self.sink_weight - attn_sink.detach()
            )

        return input_action_tokens, attn_weights
