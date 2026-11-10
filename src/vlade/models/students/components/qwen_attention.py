"""AVA Eq. 11-12 on Qwen full attention; native DeltaNet remains unchanged."""
from __future__ import annotations

import torch
from torch import nn


def visual_attention_mask(valid: torch.Tensor, omega: torch.Tensor, *, visual_start: int = 1,
                          dtype: torch.dtype = torch.float32,
                          visual_mask: torch.Tensor | None = None) -> torch.Tensor:
    """log(omega) gives softmax(C)*U / sum(softmax(C)*U); U's diagonal is 1."""
    batch, length = valid.shape
    if omega.ndim != 3 or omega.shape[0] != batch or omega.shape[-1] != 1:
        raise ValueError("Expected omega [B,visual_tokens,1]")
    if not torch.isfinite(omega).all() or (omega <= 0).any():
        raise ValueError("AVA visual weights must be positive, finite and in sequence bounds")
    if visual_mask is None:
        end = visual_start + omega.shape[1]
        if visual_start < 0 or end > length:
            raise ValueError("Visual token span is outside the input sequence")
        visual_mask = torch.zeros_like(valid, dtype=torch.bool)
        visual_mask[:, visual_start:end] = True
    if (visual_mask.shape != valid.shape or visual_mask.dtype != torch.bool
            or not (visual_mask.sum(-1) == omega.shape[1]).all()):
        raise ValueError("Visual mask must identify exactly the native image placeholder tokens")
    key_bias = omega.new_zeros(batch, length, dtype=torch.float32)
    key_bias = key_bias.masked_scatter(visual_mask, omega.squeeze(-1).float().log())
    bias = key_bias[:, None, :].expand(batch, length, length)
    diagonal = torch.eye(length, dtype=torch.bool, device=valid.device)
    bias = torch.where(diagonal[None], 0., bias)
    allowed = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()[None]
    allowed = allowed & valid.bool()[:, None, :]
    # Every query has at least the leading valid prompt token, including padding queries.
    return bias.masked_fill(~allowed, torch.finfo(dtype).min).to(dtype)[:, None]


def language_forward(language_model: nn.Module, embeddings: torch.Tensor, valid: torch.Tensor,
                     omega: torch.Tensor, *, temporal_layer: int = -2,
                     visual_mask: torch.Tensor | None = None,
                     position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stateless native decoder layers with separate full/linear masks.

    Mirrors Qwen3_5TextModel's embeddings-only path (Transformers 5.9.0).
    Native Qwen visual tokens and M-RoPE positions come from the VLM caller.
    No global patches, forward hooks, logits, or KV/recurrent cache.
    Passing a 4D AVA mask through the text model's public 2D mask input would also
    route it to DeltaNet in 5.9.0, so masks are supplied at the decoder boundary.
    """
    config = language_model.config
    if config._attn_implementation not in ("eager", "sdpa"):
        raise ValueError("AVA additive attention mask requires eager or sdpa attention")
    batch, length, _ = embeddings.shape
    full_mask = visual_attention_mask(valid, omega, dtype=embeddings.dtype, visual_mask=visual_mask)
    positions = torch.arange(length, device=embeddings.device)[None].expand(batch, -1)
    if position_ids is None:
        position_ids = positions[None].expand(3, -1, -1)
    if position_ids.shape != (3, batch, length):
        raise ValueError("Expected native Qwen M-RoPE position IDs [3,B,L]")
    rotary = language_model.rotary_emb(embeddings, position_ids)
    hidden = embeddings
    # HF hidden_states convention: embeddings, each intermediate layer, final norm.
    selected_index = len(language_model.layers) + 1 + temporal_layer if temporal_layer < 0 else temporal_layer
    if not 0 <= selected_index <= len(language_model.layers):
        raise ValueError("Temporal hidden-state layer is outside the backbone")
    selected = hidden if selected_index == 0 else None
    for index, layer in enumerate(language_model.layers):
        layer_mask = valid if config.layer_types[index] == "linear_attention" else full_mask
        hidden = layer(hidden, position_embeddings=rotary, attention_mask=layer_mask,
                       position_ids=positions, past_key_values=None, use_cache=False)
        if index + 1 == selected_index:
            selected = hidden
    hidden = language_model.norm(hidden)
    return hidden, hidden if selected_index == len(language_model.layers) else selected
