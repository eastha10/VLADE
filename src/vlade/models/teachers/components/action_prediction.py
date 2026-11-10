# Source: eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456
# Copyright 2026 LiAuto-DSR. Apache-2.0 and inherited MIT: AVA_LICENSE.txt.
# Teacher-local LIBERO/L1 subset, separated by role; upstream is unchanged.
"""LIBERO continuous L1 inference, Q99 unnormalization and temporal features."""
from typing import Any, Dict, Optional
import numpy as np
import torch
from ..constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)


class L1ActionPredictionMixin:
    """Action inference helpers; projector/head/AVA modules are caller supplied."""

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK))
            .to(input_ids.device)
            .to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = (
            torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype)
            * STOP_INDEX
        )
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1]))
            .to(labels.device)
            .to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = (
                np.array(action_norm_stats["q99"]),
                np.array(action_norm_stats["q01"]),
            )
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-08) + action_low,
            normalized_actions,
        )
        return actions

    def _regression_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        extra_attn_weights=None,
        temporal_feature_layer=-1,
        input_action_tokens=None,
    ):
        """Run the active L1 regression head and select native temporal features."""
        if input_action_tokens is None:
            all_actions_mask = all_actions_mask.unsqueeze(-1)
            input_embeddings = input_embeddings * ~all_actions_mask
        else:
            for i in range(input_embeddings.shape[0]):
                mask_ids = all_actions_mask[i].bool()
                input_embeddings[i][mask_ids] = input_action_tokens[i]
        if extra_attn_weights is None:
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
        else:
            multimodal_embeddings, multimodal_attention_mask, extra_attn_weights = (
                self._build_multimodal_attention(
                    input_embeddings, projected_patch_embeddings, attention_mask, extra_attn_weights
                )
            )
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            **self._language_attention_kwargs(extra_attn_weights),
        )
        last_hidden_states = language_model_output.hidden_states[-1]
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES
            + NUM_PROMPT_TOKENS : NUM_PATCHES
            + NUM_PROMPT_TOKENS
            + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]
        normalized_actions = action_head.predict_action(actions_hidden_states)
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        normalized_actions = normalized_actions.float().cpu().detach().numpy()
        last_hidden_states = language_model_output.hidden_states[temporal_feature_layer]
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES
            + NUM_PROMPT_TOKENS : NUM_PATCHES
            + NUM_PROMPT_TOKENS
            + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]
        return (normalized_actions, actions_hidden_states, last_hidden_states)

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        temporal_context: Optional[torch.FloatTensor] = None,
        temporal_feature_layer=-1,
        vision_attn_weight_generator=None,
        vision_attn_ratio=None,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict the active continuous L1 chunk and return native temporal features."""
        if action_head is None or noisy_action_projector is not None or use_film:
            raise ValueError(
                "Teacher supports LIBERO continuous L1 with native vision and AVA recurrence only"
            )
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (
                    input_ids,
                    torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device),
                ),
                dim=1,
            )
        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        projected_patch_embeddings = self._process_vision_features(
            pixel_values, language_embeddings, use_film
        )
        extra_attn_weights = None
        vision_attn_weights = None
        input_action_tokens = None
        if vision_attn_weight_generator is not None:
            input_action_tokens, (extra_attn_weights, vision_attn_weights) = (
                vision_attn_weight_generator(
                    projected_patch_embeddings,
                    temporal_context,
                    language_embeddings,
                    selected_ratio=vision_attn_ratio,
                )
            )
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(
                projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype
            )
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )
        if (
            extra_attn_weights is not None
            and projected_patch_embeddings.shape[1] != extra_attn_weights.shape[1]
        ):
            B, N = (
                extra_attn_weights.shape[0],
                projected_patch_embeddings.shape[1] - extra_attn_weights.shape[1],
            )
            extra_attn_weights = torch.cat(
                [extra_attn_weights, extra_attn_weights.new_ones((B, N, 1))], dim=1
            )
        NUM_PATCHES = (
            self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        )
        if use_proprio:
            NUM_PATCHES += 1
        normalized_actions, actions_hidden_states, last_hidden_states = self._regression_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            NUM_PATCHES,
            NUM_PROMPT_TOKENS,
            action_head,
            extra_attn_weights=extra_attn_weights,
            temporal_feature_layer=temporal_feature_layer,
            input_action_tokens=input_action_tokens,
        )
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)
        return (actions, actions_hidden_states, last_hidden_states, input_ids, vision_attn_weights)

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
