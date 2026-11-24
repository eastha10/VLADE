"""Reproduce plan rejection and inspect the actual FFT objective on a tiny graph."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))

from vlade.distillation.teacher_plan import normalize_generated_plan, normalize_plan
from vlade.evaluation.plan_observations import encode_query, live_response
from vlade.evaluation.s0_libero_rollout import RolloutQuery
from vlade.training.fft import compute_loss


@pytest.mark.parametrize("decoded,old_valid,current_valid", [
    ("Move toward the cup.", True, True),
    ("<plan>Move toward the cup.</plan>", False, True),
    ("", False, False),
    ("123", False, False),
    ("<plan>Move toward the cup.", False, False),
    ("<plan><other>Move.</other></plan>", False, False),
])
def test_validation_failure_is_the_source_of_none(decoded, old_valid, current_valid):
    query = RolloutQuery(0, 0, "Pick up cup", Image.new("RGB", (256, 256)),
                         Image.new("RGB", (256, 256)), np.zeros(8))
    request = {"live": "query", **encode_query(query), "action": False, "plan": True}
    for normalizer, valid in ((normalize_plan, old_valid), (normalize_generated_plan, current_valid)):
        policy = SimpleNamespace(plan=lambda *args, **kwargs: normalizer(decoded))
        response = live_response(policy, request, 48)
        assert (response["plan"] is not None) is valid


def test_real_fft_objective_does_not_train_the_language_output_head():
    class TinyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 4)
            self.lm_head = torch.nn.Linear(4, 4)
            self.last_output = None

        def forward(self, input_ids, **kwargs):
            hidden = self.embedding(input_ids)
            logits = self.lm_head(hidden)
            # Provide a differentiable LM loss; production compute_loss ignores it.
            self.last_output = SimpleNamespace(hidden_states=(hidden,), loss=logits.square().mean())
            return self.last_output

    class ActionHead(torch.nn.Linear):
        def predict_action(self, hidden):
            return self(hidden).reshape(1, 8, 7)

    model, head = TinyTeacher(), ActionHead(4, 1)
    batch = {
        "labels": torch.tensor([[-100] + [31744] * 56 + [2]]),
        "input_ids": torch.ones(1, 58, dtype=torch.long),
        "attention_mask": torch.ones(1, 58, dtype=torch.long),
        "pixel_values": torch.zeros(1, 1), "proprio": torch.zeros(1, 8),
        "actions": torch.zeros(1, 8, 7),
    }
    loss = compute_loss(model, head, None, None, batch, torch.device("cpu"), 0, 0)
    assert model.last_output.loss.requires_grad
    loss.backward()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert model.embedding.weight.grad is not None
    assert model.lm_head.weight.grad is None and model.lm_head.bias.grad is None
