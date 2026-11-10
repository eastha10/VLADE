# Source: eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456
# Copyright 2026 LiAuto-DSR. Apache-2.0 and inherited MIT: AVA_LICENSE.txt.
# Teacher-local LIBERO/L1 subset, separated by role; upstream is unchanged.
"""The active softmaxscore AVA head and its score parsing."""
import ast
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def parse_score_config(score_config):
    values = ast.literal_eval(score_config) if isinstance(score_config, str) else score_config
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError("score_config must be a tuple/list of (pos_score, neg_score, score_bias)")
    return tuple(float(value) for value in values)


class SoftmaxScoreHead(nn.Module):
    def __init__(
        self, embed_dim, init_tau=1.0, final_tau=0.1, max_steps=None, score_config=(1.9, 0.1, 0.0)
    ):
        super().__init__()

        pos_score, neg_score, score_bias = parse_score_config(score_config)
        scores = (pos_score, neg_score)
        self.scores = scores
        self.score_bias = score_bias

        self.pre_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 2, bias=False)
        self.init_tau = init_tau
        self.final_tau = final_tau
        self.alpha = (
            -np.log(final_tau / init_tau) / (max_steps * 0.8) if max_steps is not None else None
        )
        self._init_weights()

    def _init_weights(self):
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias[0], 0.05)
            nn.init.constant_(self.proj.bias[1], 0.0)

    def forward(self, tokens, selected_ratio=None, batch_idx=None):
        B, N, D = tokens.shape
        tokens = self.pre_norm(tokens.reshape(B * N, D)).reshape(B, N, D)
        logits = self.proj(tokens)  # (B, N, 2)

        if self.alpha is not None and batch_idx is not None:  # Training
            tau = float(max(self.final_tau, self.init_tau * np.exp(-self.alpha * batch_idx)))
            probs = F.softmax(logits / tau, dim=-1)
            attn_weights = (
                self.score_bias
                + probs[:, :, [0]] * self.scores[0]
                + probs[:, :, [1]] * self.scores[1]
            )
            ret_value = (attn_weights, probs[:, :, [0]])
        elif selected_ratio is None:  # Automatically select activated tokens during inference
            assert not self.training
            probs = F.softmax(logits / self.final_tau, dim=-1)
            attn_weights = (
                self.score_bias
                + probs[:, :, [0]] * self.scores[0]
                + probs[:, :, [1]] * self.scores[1]
            )
            ret_value = attn_weights
        else:  # Directly select activated tokens during inference
            assert not self.training
            probs = F.softmax(logits / self.final_tau, dim=-1)
            attn_weights = (
                self.score_bias
                + probs[:, :, [0]] * self.scores[0]
                + probs[:, :, [1]] * self.scores[1]
            )

            _, N, _ = attn_weights.shape
            num_pruned = int(N * (1 - selected_ratio))
            indexes = torch.topk(attn_weights, k=num_pruned, dim=-2, largest=False)[1]
            attn_weights.scatter_(-2, indexes, 0)
            ret_value = attn_weights

        return ret_value
