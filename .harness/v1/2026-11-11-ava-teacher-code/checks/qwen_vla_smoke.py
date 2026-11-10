"""Tiny complete native Qwen vision/merger/language + original AVA/proprio/L1."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from vlade.models.contracts import masked_action_l1
from vlade.models.students.configuration_qwen_vla import QwenVLAConfig
from vlade.models.students.qwen_vla import QwenVLA


def tiny_student() -> QwenVLA:
    from transformers import Qwen3_5Config, Qwen3_5Model, Qwen3_5TextConfig, Qwen3_5VisionConfig
    text = Qwen3_5TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=4, linear_num_value_heads=4,
        layer_types=["linear_attention"]*3 + ["full_attention"],
        pad_token_id=0, eos_token_id=2, max_position_embeddings=2048,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    vision = Qwen3_5VisionConfig(depth=2, hidden_size=32, intermediate_size=64,
        num_heads=4, out_hidden_size=32, patch_size=2, spatial_merge_size=2,
        temporal_patch_size=2, num_position_embeddings=16, deepstack_visual_indexes=[])
    native = Qwen3_5Config(text_config=text, vision_config=vision,
        image_token_id=92, video_token_id=93, vision_start_token_id=90, vision_end_token_id=91)
    native._attn_implementation = "eager"
    config = QwenVLAConfig(hidden_size=32, image_size=8, patch_grid=2,
                           ava_embed_dim=128, action_head_hidden_dim=64)
    return QwenVLA(Qwen3_5Model(native), config)


def tiny_inputs():
    ids = torch.tensor([[1,10,90,92,92,92,92,91,11,90,92,92,92,92,91,12,2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        pixel_values=torch.randn(2*16, 3*2*2*2), image_grid_thw=torch.tensor([[1,4,4]]*2),
        mm_token_type_ids=(ids == 92).long())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = tiny_student()
    inputs, proprio = tiny_inputs(), torch.randn(1,8)
    target, valid = torch.randn(1,8,7), torch.ones(1,8,7,dtype=torch.bool)
    first = model(inputs, proprio)
    second = model(inputs, proprio, first.recurrent_state, batch_idx=1)
    loss = masked_action_l1(first,target,valid) + masked_action_l1(second,target,valid)
    loss.backward()
    groups = dict(vision=model.vision_backbone, merger=model.vision_backbone.merger,
        language=model.language_model, ava=model.ava, proprio=model.proprio_projector, action=model.action_head)
    for name,module in groups.items():
        if not any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
                   for p in module.parameters()):
            raise AssertionError(f"No finite nonzero gradient in {name}")
    model.eval()
    with torch.no_grad():
        reference = model(inputs,proprio).actions
        model.save(args.output_dir / "model")
        restored = QwenVLA.load(args.output_dir / "model").eval()
        torch.testing.assert_close(restored(inputs,proprio).actions,reference)
    report = dict(status="passed", action_shape=[1,8,7], state_shape=[1,56,32],
        loss=float(loss.detach()), torch=torch.__version__, full_attention_layers=model.modulated_layers,
        scope="tiny random native Qwen vision/merger/language + original AVA/proprio/L1",
        external_projector=False, dinov2_siglip=False, pretrained_0_8b=False, gpu=False)
    (args.output_dir / "result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
