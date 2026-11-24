"""Check the T1 output boundary and the Student prompt contract."""

from pathlib import Path
from contextlib import nullcontext
from io import StringIO
import json
import queue
import sys
from types import SimpleNamespace

import pytest
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))

from vlade.distillation.teacher_plan import InvalidTeacherPlan, normalize_generated_plan, prompt_text


@pytest.mark.parametrize("decoded", ["Move toward the cup.", " \n<plan>Move toward\n the cup.</plan> "])
def test_generated_plan_reaches_student_as_plain_text(decoded):
    plan = normalize_generated_plan(decoded)
    assert prompt_text(plan, 8) == "Teacher approach plan at t=8: Move toward the cup."


@pytest.mark.parametrize("decoded", [
    "", "<plan> </plan>", "<plan>Move toward the cup.", "<plan><other>Move.</other></plan>",
    "<plan>Move.</plan> trailing text", "<plan>123</plan>", f"<plan>{'a' * 2049}</plan>",
])
def test_malformed_or_empty_generated_plan_is_rejected(decoded):
    with pytest.raises(InvalidTeacherPlan):
        normalize_generated_plan(decoded)


@pytest.mark.parametrize("decoded", [None, 123, [], b"Move."])
def test_non_text_output_uses_the_plan_error_contract(decoded):
    with pytest.raises(InvalidTeacherPlan, match="must be text"):
        normalize_generated_plan(decoded)


def test_invalid_output_keeps_a_bounded_original_preview():
    decoded = "<plan>" + "a" * 300 + "TAIL"
    with pytest.raises(InvalidTeacherPlan) as caught:
        normalize_generated_plan(decoded)
    message = str(caught.value)
    assert repr(decoded[:256]) in message
    assert "TAIL" not in message and "truncated" in message


def test_live_invalid_plan_reports_its_query_and_original_output(capsys):
    from vlade.evaluation.plan_observations import encode_query, live_response
    from vlade.evaluation.s0_libero_rollout import RolloutQuery

    def plan(*args, **kwargs):
        return normalize_generated_plan("<plan> </plan>")

    query = RolloutQuery(5, 8, "Pick up cup", Image.new("RGB", (256, 256)),
                         Image.new("RGB", (256, 256)), np.zeros(8))
    policy = SimpleNamespace(plan=plan, predict=lambda *args: (torch.ones(8, 7), None))
    result = live_response(policy, {"live": "query", **encode_query(query), "action": True, "plan": True}, 48)
    assert result["plan"] is None and result["action"] == [[1.] * 7] * 8
    streams = capsys.readouterr()
    assert not streams.out
    diagnostic = json.loads(streams.err)
    assert (diagnostic["event"], diagnostic["task_id"], diagnostic["timestep"]) == (
        "invalid_teacher_plan", 5, 8)
    assert "<plan> </plan>" in diagnostic["error"]


def test_invalid_output_crosses_worker_pipe_and_reaches_training_log(monkeypatch):
    from vlade.training import fixed_cache, qwen_vla_task, teacher_inference
    from vlade.models.students.configuration_qwen_vla import QwenVLAConfig

    def plan(*args, **kwargs):
        return normalize_generated_plan("<plan>Broken response")

    fake = SimpleNamespace(get=lambda *args: torch.zeros(8, 7), plan=plan)
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda value: Path(value))
    monkeypatch.setattr("vlade.data.episode_data.EpisodeData", lambda *args: object())
    monkeypatch.setattr(teacher_inference, "teacher_identity", lambda *args: (Path("T1"), {"temporal_feature_layer": -2}))
    monkeypatch.setattr(fixed_cache, "SequentialTeacher", lambda *args, **kwargs: object())
    monkeypatch.setattr(teacher_inference, "EpisodeTeacher", lambda *args, **kwargs: fake)
    monkeypatch.setattr(sys, "argv", ["teacher", "--checkpoint", "T1", "--data", "data",
                                     "--device", "cpu", "--plan-max-new-tokens", "48"])
    sample = {"episode_id": "a", "timestep": 8, "instruction": "Pick up cup"}
    request = {"episode_id": "a", "timestep": 8, "signal": "action_and_plan"}
    wire, errors = StringIO(), StringIO()
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(request) + "\n"))
    monkeypatch.setattr(sys, "stdout", wire)
    monkeypatch.setattr(sys, "stderr", errors)
    with pytest.raises(SystemExit) as caught:
        teacher_inference.main()
    assert caught.value.code == 1
    ready, failure = map(json.loads, wire.getvalue().splitlines())
    assert "ready" in ready and failure["error"] == "InvalidTeacherPlan"
    assert "<plan>Broken response" in failure["error_message"]

    client = teacher_inference.InferenceActions.__new__(teacher_inference.InferenceActions)
    client.plan_tokens, client.timings, client.timeout = 48, None, 1
    client.replies = queue.Queue()
    client.replies.put(json.dumps(failure))
    monkeypatch.setattr(client, "_send", lambda request: client._receive())
    monkeypatch.setattr(qwen_vla_task.torch, "autocast", lambda *args, **kwargs: nullcontext())
    plan_log = StringIO()
    loss, _, _ = qwen_vla_task.action_loss(None, None, None, QwenVLAConfig(), "cpu",
                                           plan_teacher=client, plan_log=plan_log)
    with pytest.raises(RuntimeError, match="Broken response"):
        loss((sample, None, None), 0)
    row = json.loads(plan_log.getvalue())
    assert row["status"] == "plan_error" and row["timestep"] == 8
    assert "<plan>Broken response" in row["error"]
