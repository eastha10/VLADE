"""Explicit Llama-2 backbone connection for the AVA/OFT research policies.

The decoder implementation is supplied by the pinned Transformers environment.
AVA's extra visual attention remains in that fork; no Prismatic checkout is used.
"""

from __future__ import annotations

from torch import nn
from .. import configuration_ava_vla


def build_llama_backbone(config: configuration_ava_vla.PrismaticConfig) -> nn.Module:
    """Build LlamaForCausalLM with the original HF initialization and AVA flags."""
    from transformers import AutoModelForCausalLM, LlamaForCausalLM

    if config.text_config.model_type != "llama":
        raise ValueError("The current Teacher/OFT policies require the Llama-2 backbone")
    attention = getattr(config, "extra_attn_weights", None) or {}
    if attention:
        config.text_config.force_eager_attn = attention["force_eager_attn"]
        config.text_config.extra_layer_ids = attention["extra_layer_ids"]
    language_model = AutoModelForCausalLM.from_config(
        config.text_config, attn_implementation=config._attn_implementation
    )
    if not isinstance(language_model, LlamaForCausalLM):
        raise TypeError("The selected Transformers environment did not construct LlamaForCausalLM")
    return language_model
