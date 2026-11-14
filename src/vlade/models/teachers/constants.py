# Source: eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456
# Copyright 2026 LiAuto-DSR. Apache-2.0 and inherited MIT: AVA_LICENSE.txt.
# Teacher-local LIBERO/L1 subset; external submodule is unchanged.
"""The active LIBERO action/token/normalization contract, without argv detection."""
from enum import Enum

IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8


class NormalizationType(str, Enum):
    BOUNDS_Q99 = "bounds_q99"


ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
