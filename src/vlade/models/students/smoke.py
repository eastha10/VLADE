"""Native tiny random Qwen3.5 multimodal CPU check, without pretrained downloads."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from vlade.models.contracts import masked_action_l1
from vlade.models.students.qwen import HistoryFrame, QwenStudent, StudentConfig, load_processor, prepare_inputs


def tiny_student(*, native_processor: bool = False) -> QwenStudent:
    from transformers import Qwen3_5Config, Qwen3_5Model, Qwen3_5TextConfig, Qwen3_5VisionConfig

    text = Qwen3_5TextConfig(
        vocab_size=248320 if native_processor else 128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=4, linear_num_value_heads=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        max_position_embeddings=8192, pad_token_id=248044 if native_processor else 0,
        eos_token_id=248044 if native_processor else 2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2],
                         "mrope_interleaved": True},
    )
    vision = Qwen3_5VisionConfig(depth=2, hidden_size=32, intermediate_size=64,
                                 num_heads=4, out_hidden_size=32, patch_size=16 if native_processor else 2,
                                 spatial_merge_size=2, temporal_patch_size=2,
                                 num_position_embeddings=256 if native_processor else 16)
    config = Qwen3_5Config(text_config=text, vision_config=vision,
                          image_token_id=248056 if native_processor else 92,
                          video_token_id=248057 if native_processor else 93,
                          vision_start_token_id=248053 if native_processor else 90,
                          vision_end_token_id=248054 if native_processor else 91)
    return QwenStudent(Qwen3_5Model(config), replace(StudentConfig(), hidden_size=32, head_size=32))


def tiny_inputs() -> dict[str, torch.Tensor]:
    # Four images = two cameras x two queries, each 4x4 native vision patches.
    ids = [1, 10]
    for index in range(4):
        ids += [20 + index, 90, 92, 92, 92, 92, 91]
    ids += [11, 2]
    input_ids = torch.tensor([ids])
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.randn(4 * 16, 3 * 2 * 2 * 2),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * 4),
            "mm_token_type_ids": (input_ids == 92).long()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, help="Official pinned tokenizer/processor files, no model weights needed")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = tiny_student(native_processor=args.processor_dir is not None)
    if args.processor_dir:
        from PIL import Image

        processor = load_processor(StudentConfig(), directory=args.processor_dir)
        frames = [HistoryFrame(t, Image.new("RGB", (256, 256), color), Image.new("RGB", (256, 256), "green"))
                  for t, color in ((0, "red"), (8, "blue"))]
        inputs = prepare_inputs(processor, StudentConfig(), instruction="pick the bowl", frames=frames, current_timestep=8)
        # Checkpoint embeddings may pad the vocabulary beyond actual tokenizer IDs.
        if max(processor.tokenizer.get_vocab().values()) >= model.backbone.config.text_config.vocab_size:
            raise AssertionError("Pinned tokenizer ID exceeds backbone embedding vocabulary")
        if processor.tokenizer.convert_tokens_to_ids("<|image_pad|>") != model.backbone.config.image_token_id:
            raise AssertionError("Pinned image placeholder differs from backbone")
    else:
        inputs = tiny_inputs()
    proprio = torch.randn(1, 8)
    target, valid = torch.randn(1, 8, 7), torch.ones(1, 8, 7, dtype=torch.bool)
    # One stateless update verifies connectivity; this is not a training recipe.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    output = model(inputs, proprio)
    loss = masked_action_l1(output, target, valid)
    loss_value = loss.item()
    loss.backward()
    gradient_groups = {
        "vision": model.backbone.visual.parameters(),
        "language": model.backbone.language_model.parameters(),
        "proprio": model.head.proprio.parameters(),
        "action": model.head.action.parameters(),
    }
    gradient_report = {}
    for name, parameters in gradient_groups.items():
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
            raise AssertionError(f"No finite nonzero {name} gradient")
        gradient_report[name] = True
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(inputs, proprio).actions
        perturbations = {}
        for name in ("vision", "language", "proprio"):
            changed = {key: value.clone() for key, value in inputs.items()}
            new_proprio = proprio.clone()
            if name == "vision":
                changed["pixel_values"].zero_()
            elif name == "language":
                # Alter a real instruction token, not any image placeholder.
                token_position = 5 if args.processor_dir else 1
                changed["input_ids"][0, token_position] = 12
            else:
                new_proprio.zero_()
            delta = (model(changed, new_proprio).actions - expected).abs().max().item()
            if delta <= 1e-8:
                raise AssertionError(f"Output ignores {name}")
            perturbations[name] = delta
    model.save(args.output_dir / "checkpoint")
    del model, optimizer, output, loss
    import gc

    gc.collect()
    restored = QwenStudent.load(args.output_dir / "checkpoint").eval()
    with torch.no_grad():
        actual = restored(inputs, proprio).actions
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    import transformers

    report = {"status": "passed", "test_type": "tiny_random_native_qwen3_5_cpu_not_pretrained_0_8b",
              "torch": torch.__version__, "transformers": transformers.__version__,
              "action_shape": list(expected.shape), "loss": loss_value,
              "finite_nonzero_gradient_groups": gradient_report,
              "input_sensitivity_max_abs": perturbations, "save_load_exact": True,
              "official_tokenizer_processor_tested": args.processor_dir is not None,
              "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
              "pretrained_weights_tested": False, "gpu_tested": False, "edge_tested": False}
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0
