# Source: eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456
# Copyright 2026 LiAuto-DSR. Apache-2.0 and inherited MIT: AVA_LICENSE.txt.
# Teacher-local LIBERO/L1 subset, separated by role; upstream is unchanged.
"""Current-image/proprio/AVA token assembly and native Llama forward."""
from typing import List, Optional, Tuple, Union
import torch
from ..action_masks import get_current_action_mask, get_next_actions_mask
from ..constants import IGNORE_INDEX
from ..outputs import PrismaticCausalLMOutputWithPast


class MultimodalForwardMixin:
    """Stateless input processing; all parameters remain on the HF model."""

    def _language_attention_kwargs(self, weights):
        """AVA injects visual weights; OFT uses its native Llama call unchanged."""
        if self.extra_layer_ids:
            return {"extra_attn_weights": weights, "extra_layer_ids": self.extra_layer_ids}
        return {}

    def _process_action_masks(self, labels):
        """Helper to get action masks from labels"""
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        """Project current native DINOv2/SigLIP patch features into Llama space."""
        if use_film:
            raise ValueError("Teacher uses the native vision path without FiLM vision variants")
        return self.projector(self.vision_backbone(pixel_values))

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """Process proprioceptive features and append to vision features"""
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
            # proprio: (bsz, proprio_dim) or (propro_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # For simplicity, just append proprio token to the end of projected vision patch tokens
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(
        self,
        input_embeddings,
        projected_patch_embeddings,
        attention_mask,
        extra_attention_weights=None,
    ):
        """Build multimodal embeddings and attention mask"""
        # Update attention mask
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]],
            dim=1,
        )
        if extra_attention_weights is not None:
            B, N = extra_attention_weights.shape[0], input_embeddings.shape[1]
            extra_attention_weights = torch.cat(
                [
                    extra_attention_weights.new_ones((B, 1, 1)),
                    extra_attention_weights,
                    extra_attention_weights.new_ones(B, N - 1, 1),
                ],
                dim=1,
            )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]],
                dim=1,
            )

        if extra_attention_weights is None:
            return multimodal_embeddings, multimodal_attention_mask
        else:
            return multimodal_embeddings, multimodal_attention_mask, extra_attention_weights

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """Build multimodal labels with IGNORE_INDEX for patch embeddings"""
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        proprio_projector=None,
        noisy_actions=None,
        noisy_action_projector=None,
        diffusion_timestep_embeddings=None,
        use_film: bool = False,
        temporal_context: Optional[torch.FloatTensor] = None,
        vision_attn_weight_generator=None,
        batch_idx=None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        if (
            noisy_actions is not None
            or noisy_action_projector is not None
            or diffusion_timestep_embeddings is not None
            or use_film
        ):
            raise ValueError(
                "Teacher supports LIBERO continuous L1 with native vision and AVA recurrence only"
            )
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_projector_features = (
            output_projector_features if output_projector_features is not None else False
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache and (not self.training)
        projected_patch_embeddings = None
        if (
            input_ids is None
            or pixel_values is None
            or labels is None
            or inputs_embeds is not None
            or past_key_values is not None
        ):
            raise ValueError(
                "Teacher forward requires current camera tensors, input IDs and action-label masks"
            )
        if input_ids.ndim != 2 or input_ids.shape[0] != pixel_values.shape[0]:
            raise ValueError("Teacher forward requires equal image/text batch sizes")
        assert (
            past_key_values is None
        ), "Unexpected key `past_key_values` provided during multimodal forward!"
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
                    batch_idx=batch_idx,
                )
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
                    input_embeddings,
                    projected_patch_embeddings,
                    attention_mask,
                    extra_attn_weights,
                )
            )
        multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=multimodal_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **self._language_attention_kwargs(extra_attn_weights),
        )
        if not return_dict:
            if output_projector_features and projected_patch_embeddings is not None:
                return (*language_model_output, projected_patch_embeddings)
            return language_model_output
        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
            vision_attn_weights=vision_attn_weights,
        )
