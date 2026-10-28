"""Fast head/input-boundary tests, no pretrained checkpoint or model download."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.models.contracts import masked_action_l1
from vlade.models.students.qwen import ContinuousActionHead, HistoryFrame, StudentConfig, prepare_inputs


def test_checked_in_student_configuration_matches_contract():
    path = Path(__file__).resolve().parents[2] / "configs/models/student/qwen3_5_0_8b.json"
    assert StudentConfig.from_json(path) == StudentConfig()


def test_head_padding_mask_and_training():
    torch.manual_seed(7)
    head = ContinuousActionHead(replace(StudentConfig(), hidden_size=16, head_size=16))
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0]])
    proprio = torch.randn(2, 8, requires_grad=True)
    output = head(hidden, mask, proprio)
    altered = hidden.detach().clone()
    altered[~mask.bool()] = float("nan")
    torch.testing.assert_close(head(altered, mask, proprio).actions, output.actions)
    valid = torch.ones(2, 8, 7, dtype=torch.bool)
    valid[:, -1] = False
    target = torch.zeros_like(output.actions)
    target[:, -1] = float("nan")
    masked_action_l1(output, target, valid).backward()
    assert torch.isfinite(proprio.grad).all() and proprio.grad.abs().sum() > 0
    assert hidden.grad[~mask.bool()].abs().sum() == 0
    with pytest.raises(ValueError):
        head(hidden, torch.zeros_like(mask), proprio)


class ProcessorSpy:
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return {"input_ids": torch.ones(1, 8, dtype=torch.long)}


def test_history_is_causal_marked_and_episode_start_repeated():
    spy = ProcessorSpy()
    config = StudentConfig()
    prepare_inputs(spy, config, instruction="pick", frames=[HistoryFrame(0, "p0", "w0")], current_timestep=0)
    content = spy.messages[0]["content"]
    assert [item["image"] for item in content if item["type"] == "image"] == ["p0", "w0", "p0", "w0"]
    assert not spy.kwargs["enable_thinking"]
    frames = [HistoryFrame(t, f"p{t}", f"w{t}") for t in (0, 8, 16)]
    prepare_inputs(spy, config, instruction="pick", frames=frames, current_timestep=16)
    assert [item["image"] for item in spy.messages[0]["content"] if item["type"] == "image"] == ["p8", "w8", "p16", "w16"]
    with pytest.raises(ValueError):
        prepare_inputs(spy, config, instruction="pick", frames=frames, current_timestep=8)
    with pytest.raises(ValueError):
        prepare_inputs(spy, config, instruction="pick", frames=frames[1:], current_timestep=8)
