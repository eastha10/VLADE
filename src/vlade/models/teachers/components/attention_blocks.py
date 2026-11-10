# Source: eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456
# Copyright 2026 LiAuto-DSR. Apache-2.0 and inherited MIT: AVA_LICENSE.txt.
# Teacher-local LIBERO/L1 subset, separated by role; upstream is unchanged.
"""AVA text FiLM, action cross-attention, image self-attention and FFN blocks."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from .attention_utils import DEFAULT_NORM_LAYER, apply_rope, build_mlp


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        max_position=1024,
        wo_qkv_proj=False,
        wo_out_proj=False,
        use_sdpa=True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5
        self.use_sdpa = use_sdpa

        self.wo_qkv_proj = wo_qkv_proj
        self.wo_out_proj = wo_out_proj
        if not wo_qkv_proj:
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        if not wo_out_proj:
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.apply(self._init_weights)

        self.max_position = max_position
        self._init_rope()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_rope(self):
        inv_freq = 1 / (10000 ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        max_seq_len = self.max_position + 128
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def forward(self, query, key, value, query_position_ids=None, key_position_ids=None):
        B, N_q, _ = query.shape
        N_k = key.shape[1]

        if not self.wo_qkv_proj:
            query_states = self.q_proj(query)
            key_states = self.k_proj(key)
            value_states = self.v_proj(value)
        else:
            query_states, key_states, value_states = query, key, value

        query_states = query_states.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Add RoPE
        if query_position_ids is not None:
            key_position_ids = (
                key_position_ids if key_position_ids is not None else query_position_ids
            )
            query_states = apply_rope(
                query_states, self.cos_cache, self.sin_cache, query_position_ids
            )
            key_states = apply_rope(key_states, self.cos_cache, self.sin_cache, key_position_ids)

        if self.use_sdpa:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, scale=self.scale
            )
            attn_weights = None
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, N_q, self.embed_dim)

        if not self.wo_out_proj:
            attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class InjectTextModule(nn.Module):
    def __init__(self, embed_dim, norm_layer=DEFAULT_NORM_LAYER):
        super().__init__()

        self.proj = build_mlp(embed_dim, embed_dim * 2, hidden_size=embed_dim * 4, dropout=0.1)

    def forward(self, tokens, text_tokens):
        text_token = text_tokens.mean(dim=-2)
        text_token = self.proj(text_token)

        scale, shift = torch.chunk(text_token.unsqueeze_(1), 2, dim=-1)

        tokens = tokens * (1 + scale) + shift

        return tokens


class InjectActionModule(nn.Module):
    def __init__(
        self, embed_dim, image_shape, action_shape, num_images, norm_layer=DEFAULT_NORM_LAYER
    ):
        super().__init__()

        self.pre_norm = norm_layer(embed_dim)
        self.pre_action_norm = norm_layer(embed_dim)
        self.cross_attn = MultiheadAttention(embed_dim, num_heads=embed_dim // 128)

        image_h, image_w, _ = image_shape
        num_patches = image_h * image_w
        action_frames, action_dim, _ = action_shape
        num_actions = action_frames * action_dim
        query_position_ids = torch.arange(num_patches * num_images)
        key_position_ids = torch.arange(
            num_patches * num_images + 38, num_patches * num_images + 38 + num_actions
        )
        self.register_buffer("query_position_ids", query_position_ids)
        self.register_buffer("key_position_ids", key_position_ids)

    def forward(self, tokens, action_tokens):
        residual = tokens
        tokens = self.pre_norm(tokens)

        action_tokens = self.pre_action_norm(action_tokens)

        attn_out, _ = self.cross_attn(
            tokens, action_tokens, action_tokens, self.query_position_ids, self.key_position_ids
        )

        tokens = residual + attn_out
        return tokens


class SelfAttentionModule(nn.Module):
    def __init__(self, embed_dim, image_shape, num_images, norm_layer=DEFAULT_NORM_LAYER):
        super().__init__()

        self.pre_norm = norm_layer(embed_dim)
        self.pre_ori_norm = norm_layer(embed_dim)
        self.self_attn = MultiheadAttention(embed_dim, num_heads=embed_dim // 128)

        image_h, image_w, _ = image_shape
        num_patches = image_h * image_w
        # Single-image input only uses the sin-cos position embedding.
        position_ids = None if num_images == 1 else torch.arange(num_patches * num_images)
        self.register_buffer("position_ids", position_ids)

    def forward(self, tokens, ori_tokens):
        residual = tokens
        tokens = self.pre_norm(tokens)

        ori_tokens = self.pre_ori_norm(ori_tokens)

        attn_out, _ = self.self_attn(tokens, ori_tokens, ori_tokens, self.position_ids)

        tokens = residual + attn_out
        return tokens


class FFN(nn.Module):
    def __init__(self, embed_dim, norm_layer=DEFAULT_NORM_LAYER):
        super().__init__()

        self.pre_norm = norm_layer(embed_dim)
        self.proj = build_mlp(embed_dim, embed_dim, hidden_size=embed_dim * 2, dropout=0.1)

    def forward(self, tokens):
        residual = tokens
        tokens = self.pre_norm(tokens)

        ffn_out = self.proj(tokens)

        tokens = residual + ffn_out
        return tokens


class AttentionWeightBlock(nn.Module):
    def __init__(
        self, embed_dim, image_shape, action_shape, num_images, norm_layer=DEFAULT_NORM_LAYER
    ):
        super().__init__()

        self.inject_text = InjectTextModule(embed_dim, norm_layer=norm_layer)
        self.inject_action = InjectActionModule(
            embed_dim, image_shape, action_shape, num_images, norm_layer=norm_layer
        )
        self.self_attn = SelfAttentionModule(
            embed_dim, image_shape, num_images, norm_layer=norm_layer
        )
        self.ffn = FFN(embed_dim, norm_layer=norm_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, image_tokens, action_tokens, text_tokens):
        ori_image_tokens = image_tokens

        image_tokens = self.inject_text(image_tokens, text_tokens)
        image_tokens = self.inject_action(image_tokens, action_tokens)
        image_tokens = self.self_attn(image_tokens, ori_image_tokens)

        image_tokens = self.ffn(image_tokens)
        return image_tokens
