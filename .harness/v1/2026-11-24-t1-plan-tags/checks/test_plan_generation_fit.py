"""Exercise real tiny Llama generation and native Qwen action-KD optimization."""
from io import StringIO
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))

from vlade.distillation.teacher_plan import InvalidTeacherPlan
from vlade.distillation.teacher_plan import plan_generation_kwargs
from vlade.training.fixed_cache import SequentialTeacher
from vlade.training.teacher_inference import TeacherSignal
from vlade.training import qwen_vla_task


def tiny_teacher(preferred: int = 2):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"<unk>": 0, "<s>": 1, "</s>": 2, "Move": 3, "123": 4}, unk_token="<unk>")),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    llama = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        bos_token_id=1, eos_token_id=2, pad_token_id=2)).eval()
    # Controlled logits reproduce early EOS and special-token-only generation.
    llama.lm_head = torch.nn.Linear(16, 8, bias=True)
    with torch.no_grad():
        llama.lm_head.weight.zero_()
        llama.lm_head.bias.fill_(-10)
        llama.lm_head.bias[3] = 6
        llama.lm_head.bias[2] = 7
        llama.lm_head.bias[preferred] = 10
    llama.requires_grad_(False)

    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer

        def __call__(self, prompt, image, **kwargs):
            return {"input_ids": torch.tensor([[1, 3]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
                    "pixel_values": torch.zeros(1, 1, 16)}

    teacher = SequentialTeacher.__new__(SequentialTeacher)
    teacher.device, teacher.timings, teacher.proprio, teacher.processor = "cpu", None, None, Processor()
    teacher.context = torch.randn(1, 56, 16)
    teacher.model = SimpleNamespace(
        get_input_embeddings=llama.get_input_embeddings, get_output_embeddings=llama.get_output_embeddings,
        _process_vision_features=lambda pixels: pixels,
        _process_proprio_features=lambda vision, state, projector: vision,
        _build_multimodal_attention=lambda text, vision, mask: (
            torch.cat((text, vision.to(text.dtype)), dim=1), torch.ones(1, 4, dtype=torch.long)),
        language_model=llama,
    )
    sample = {"episode_id": "synthetic", "timestep": 0, "instruction": "Move toward cup",
              "proprio": np.zeros(8, dtype=np.float32), "gt_action": np.zeros((8, 7), dtype=np.float32),
              "action_valid_mask": np.ones((8, 7), dtype=bool)}
    data = SimpleNamespace(image=lambda *args: Image.new("RGB", (8, 8)))
    return teacher, sample, data


@pytest.mark.parametrize("preferred", [2, 0, 1])
def test_real_generate_recovers_from_early_eos_unk_and_bos(preferred):
    teacher, sample, data = tiny_teacher(preferred)
    original_state = teacher.context.clone()
    assert teacher.plan(sample, data, max_new_tokens=8) == "Move"
    torch.testing.assert_close(teacher.context, original_state)


def test_nonnatural_output_is_still_rejected_without_a_canned_plan():
    teacher, sample, data = tiny_teacher(4)
    with pytest.raises(InvalidTeacherPlan, match="token_ids"):
        teacher.plan(sample, data, max_new_tokens=8)


def test_native_student_fits_generated_plan_and_action_targets():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    teacher, sample, data = tiny_teacher()
    helpers = runpy.run_path(str(ROOT / ".harness/v1/2026-11-11-ava-teacher-code/checks/qwen_vla_smoke.py"))
    model = helpers["tiny_student"]()
    inputs = helpers["tiny_inputs"]()
    seen = []

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            texts = [part["text"] for part in messages[0]["content"] if part["type"] == "text"]
            plan = next(text for text in texts if text.startswith("Teacher approach plan"))
            seen.append(plan)
            encoded = dict(inputs)
            # Keep native camera tokens while appending the generated plan token.
            ids = torch.cat((inputs["input_ids"], torch.tensor([[3]])), dim=1)
            encoded.update(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=(ids == 92).long())
            return encoded

    def signal(row):
        return TeacherSignal(torch.full((8, 7), .25), teacher.plan(row, data, max_new_tokens=8))

    plan_log, terms = StringIO(), []
    loss, _, _ = qwen_vla_task.action_loss(model, data, Processor(), model.student_config,
        torch.device("cpu"), plan_teacher=SimpleNamespace(get_signal=signal), plan_log=plan_log,
        action_weight=.5, loss_terms=terms)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    values = []
    for step in range(12):
        optimizer.zero_grad(set_to_none=True)
        objective = loss((sample, None, None), step)
        assert torch.isfinite(objective)
        objective.backward()
        for module in (model.vision_backbone, model.language_model, model.ava,
                       model.proprio_projector, model.action_head):
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                       for p in module.parameters())
        optimizer.step()
        values.append(objective.item())
    assert values[-1] < values[0]
    assert seen == ["Teacher approach plan at t=0: Move"] * 12
    assert len(terms) == 12 and all(set(row) == {"task_l1", "action_kd_l1"} for row in terms)
    assert all(row["status"] == "ok" for row in map(json.loads, plan_log.getvalue().splitlines()))
    print(json.dumps({"scope": "tiny_random_native_Qwen; controlled_tiny_Llama_logits", "optimizer_steps": 12,
                      "initial_loss": values[0], "last_loss": values[-1]}))


def test_generation_masks_action_padding_and_special_ids_but_keeps_eos():
    tokenizer = SimpleNamespace(eos_token_id=2, all_special_ids=[0, 1, 2, 32000],
                                bos_token_id=1, unk_token_id=0, pad_token_id=32000)
    generation = plan_generation_kwargs(tokenizer, 32064, 48)
    assert generation["min_new_tokens"] == 1
    assert {0, 1, 31744, 32000, 32063} <= set(generation["suppress_tokens"])
    assert 2 not in generation["suppress_tokens"]
    assert 3 not in generation["suppress_tokens"]


@pytest.mark.parametrize("budget", [0, 129, True])
def test_invalid_generation_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="max_new_tokens"):
        plan_generation_kwargs(SimpleNamespace(eos_token_id=2), 8, budget)
