# Source: eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456
# Copyright 2026 LiAuto-DSR. Apache-2.0 and inherited MIT: AVA_LICENSE.txt.
# Teacher-local LIBERO/L1 subset, separated by role; upstream is unchanged.
"""Small AVA MLP and spatial/rotary position utilities."""
from functools import partial
import numpy as np
import torch
from torch import nn

DEFAULT_NORM_LAYER = partial(nn.LayerNorm, eps=1e-6)


def build_mlp(
    in_size,
    out_size,
    hidden_size=None,
    depth=2,
    dropout=None,
    out_act=False,
    out_drop=False,
):
    if hidden_size is None:
        hidden_size = out_size

    modules = [nn.Linear(in_size, hidden_size)]
    for _ in range(2, depth):
        modules.append(nn.SiLU())
        modules.append(nn.Dropout(dropout) if dropout is not None else nn.Identity())
        modules.append(nn.Linear(hidden_size, hidden_size))

    modules += [
        nn.SiLU(),
        nn.Dropout(dropout) if dropout is not None else nn.Identity(),
        nn.Linear(hidden_size, out_size),
        nn.SiLU() if out_act else nn.Identity(),
        nn.Dropout(dropout) if out_drop and dropout is not None else nn.Identity(),
    ]
    return nn.Sequential(*modules)


def get_1d_position_embedding(size, embed_dim):
    def get_angles(pos, i, embed_dim):
        return pos / np.power(10000, (2 * (i // 2)) / embed_dim)

    angle_rads = get_angles(np.arange(size)[:, None], np.arange(embed_dim)[None, :], embed_dim)

    pos_embed = np.zeros_like(angle_rads)
    pos_embed[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_embed[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_embed


def get_2d_position_embedding(h, w, embed_dim):
    pos_embed_h = get_1d_position_embedding(h, embed_dim)
    pos_embed_w = get_1d_position_embedding(w, embed_dim)

    pos_embed = pos_embed_h[:, None, :] + pos_embed_w[None, :, :]
    return pos_embed


def apply_rope(tokens, cos_cache, sin_cache, position_ids):
    cos_cache = cos_cache[position_ids][None, None, ...]
    sin_cache = sin_cache[position_ids][None, None, ...]

    tokens_1 = tokens[..., ::2]  # (B, num_head, seq_len, dim)
    tokens_2 = tokens[..., 1::2]

    rotated_tokens_1 = tokens_1 * cos_cache[..., ::2] - tokens_2 * sin_cache[..., ::2]
    rotated_tokens_2 = tokens_1 * sin_cache[..., 1::2] + tokens_2 * cos_cache[..., 1::2]

    tokens = torch.stack([rotated_tokens_1, rotated_tokens_2], dim=-1).flatten(-2)
    return tokens
