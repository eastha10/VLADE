# Source: eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456
# Copyright 2026 LiAuto-DSR. Apache-2.0 and inherited MIT: AVA_LICENSE.txt.
# Teacher-local LIBERO/L1 subset; external submodule is unchanged.
import torch
from .constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = cumsum > ACTION_DIM

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask
