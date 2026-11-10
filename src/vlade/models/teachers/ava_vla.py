# Source: eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456
# Copyright 2026 LiAuto-DSR. Apache-2.0 and inherited MIT: AVA_LICENSE.txt.
# Teacher-local LIBERO/L1 subset, separated by role; upstream is unchanged.
"""Teacher policy composition; model internals live in dedicated modules."""
from transformers import PretrainedConfig
from .configuration_ava_vla import OpenVLAConfig
from .backbones.vlm import PrismaticForConditionalGeneration
from .components.action_prediction import L1ActionPredictionMixin


class OpenVLAForActionPrediction(L1ActionPredictionMixin, PrismaticForConditionalGeneration):
    """The active AVA Teacher with the original checkpoint class and parameter keys."""

    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of
