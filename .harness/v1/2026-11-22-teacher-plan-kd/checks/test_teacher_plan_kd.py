"""CPU checks for the query-aligned plan path and validation-based lambda selection."""
from contextlib import closing, nullcontext
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.train.select_plan_kd_weight import select
from vlade.models.contracts import PolicyOutput
from vlade.models.students.configuration_qwen_vla import QwenVLAConfig
from vlade.models.students.qwen import StudentConfig
from vlade.research.configuration import load
from vlade.training import qwen_vla_task, student_task
from vlade.training.fixed_cache import SequentialTeacher
from vlade.training.run_control import seal_checkpoint
from vlade.training.teacher_inference import EpisodeTeacher, InferenceActions, TeacherSignal


def query(timestep):
    valid = np.ones((8, 7), dtype=bool)
    valid[-1] = False
    return {"episode_id": "a", "timestep": timestep, "instruction": "place the cube",
            "proprio": np.zeros(8, dtype=np.float32), "gt_action": np.zeros((8, 7), dtype=np.float32),
            "action_valid_mask": valid}


class Data:
    def image(self, sample, camera):
        assert camera in {"primary", "wrist"}
        return Image.new("RGB", (4, 4))

    def episodes(self, split, *, order):
        assert order == ["a"]
        assert split in {"train", "validation"}
        yield {}, [query(timestep) for timestep in (0, 8, 16)]


class Processor:
    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append([row["text"] for row in messages[0]["content"] if row["type"] == "text"])
        return {"input_ids": torch.tensor([[len(self.prompts)]])}


class Teacher:
    def __init__(self):
        self.calls = []

    def get_signal(self, sample):
        self.calls.append(("action_and_plan", sample["timestep"]))
        return TeacherSignal(torch.full((8, 7), .25), "Approach the cube.")

    def get_plan(self, sample, *, split):
        self.calls.append((split, sample["timestep"]))
        return "Align the gripper."


def test_config_opt_in_rejects_cached_actions():
    config = ROOT / "configs/research/qwen_kd.json"
    assert load(config)["distillation"]["teacher_plan_max_new_tokens"] == 0
    assert load(config, ["distillation.teacher_plan_max_new_tokens=48"])["distillation"]["teacher_plan_max_new_tokens"] == 48
    with pytest.raises(ValueError, match="teacher_mode=inference"):
        load(config, ['distillation.teacher_mode="cache"', "distillation.teacher_plan_max_new_tokens=48"])


def test_plan_queries_do_not_move_the_action_replay_cursor():
    calls = []
    policy = SimpleNamespace(
        reset=lambda: calls.append("reset"),
        predict=lambda sample, data: (calls.append(sample["timestep"]), None),
        plan=lambda sample, data, **kw: f"Gripper approach at {sample['timestep']}",
    )
    teacher = EpisodeTeacher(policy, Data())
    teacher.get("a", 8)
    assert teacher.plan("a", 16, split="validation", max_new_tokens=48).endswith("16")
    assert teacher.plan("a", 0, split="train", max_new_tokens=48).endswith("0")
    teacher.get("a", 16)
    assert calls == ["reset", 0, 8, 16]
    with pytest.raises(ValueError):
        teacher.plan("a", 8, split="test", max_new_tokens=48)


def test_actual_json_pipe_returns_paired_signal_and_checks_timestep(tmp_path, monkeypatch):
    script = tmp_path / "teacher.py"
    script.write_text('''import json, sys
assert sys.argv[1:] == ["48"]
print(json.dumps({"ready": {"step": 5}}), flush=True)
for line in sys.stdin:
    row = json.loads(line)
    if row["signal"] == "action_and_plan":
        row["action"] = [[0.25] * 7 for _ in range(8)]
    row["plan"] = "Move the gripper toward the cube."
    if row["timestep"] == 16:
        row["timestep"] = 8
    print(json.dumps(row), flush=True)
''', encoding="utf-8")
    popen = subprocess.Popen
    monkeypatch.setattr("vlade.training.teacher_inference.teacher_identity", lambda *a: (tmp_path, {"step": 5}))
    monkeypatch.setattr("vlade.training.teacher_inference.subprocess.Popen",
        lambda command, **kw: popen([sys.executable, "-B", str(script), command[-1]], **kw))
    settings = {"teacher_checkpoint": "hfasset://T1", "teacher_python": sys.executable, "teacher_mode": "inference",
                "teacher_plan_max_new_tokens": 48, "teacher_timeout_seconds": 10}
    with closing(InferenceActions(settings, SimpleNamespace(root=tmp_path))) as teacher:
        signal = teacher.get_signal(query(8))
        assert signal.plan.endswith("cube.") and signal.action.shape == (8, 7)
        assert not signal.action.requires_grad
        assert teacher.get_plan(query(8), split="validation") == signal.plan
        with pytest.raises(ValueError, match="requested observation"):
            teacher.get_signal(query(16))


def test_teacher_plan_uses_current_views_proprio_and_text_generation():
    sample = query(8)
    teacher = SequentialTeacher.__new__(SequentialTeacher)
    teacher.device = "cpu"
    teacher.timings = None
    teacher.proprio = None
    seen = []

    def processor(prompt, image, *, return_tensors):
        assert "place the cube" in prompt and "timestep 8" in prompt
        assert image.size == (4, 4)
        seen.append(prompt)
        return {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.ones((1, 1, 2)),
                "attention_mask": torch.ones((1, 2), dtype=torch.long)}

    def vision(pixels):
        assert pixels.shape == (1, 2, 2)
        return pixels

    def proprio(projected, state, _):
        assert state.shape == (8,) and state.dtype == projected.dtype
        return projected

    def generate(*, inputs_embeds, attention_mask, **kwargs):
        assert not torch.is_grad_enabled()
        assert inputs_embeds.shape == (1, 4, 2) and attention_mask.shape == (1, 4)
        assert kwargs["max_new_tokens"] == 48 and not kwargs["do_sample"]
        return torch.tensor([[12, 13, 2]])

    processor.tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda ids, **kw: "Align; grasp.")
    teacher.processor = processor
    teacher.model = SimpleNamespace(
        get_input_embeddings=lambda: torch.nn.Embedding(16, 2, dtype=torch.bfloat16),
        get_output_embeddings=lambda: torch.nn.Linear(2, 16),
        _process_vision_features=vision,
        _process_proprio_features=proprio,
        _build_multimodal_attention=lambda text, pixels, mask: (torch.cat((text, pixels), dim=1),
                                                                 mask.repeat(1, 2)),
        language_model=SimpleNamespace(generate=generate),
    )
    assert teacher.plan(sample, Data(), max_new_tokens=48) == "Align; grasp."
    assert len(seen) == 2


def test_both_student_paths_condition_on_matching_plan_and_apply_final_action_loss(monkeypatch):
    data, teacher, current, previous = Data(), Teacher(), query(8), query(0)
    processor = Processor()
    monkeypatch.setattr(student_task, "samples", lambda *a, **kw: iter([(current, [previous, current])]))
    trained = next(student_task.batches(data, processor, StudentConfig(), split="train", plan_teacher=teacher))
    evaluated = next(student_task.batches(data, processor, StudentConfig(), split="validation", plan_teacher=teacher))
    assert len(trained) == 5 and len(evaluated) == 4
    assert "Teacher approach plan at t=8: Approach the cube." in processor.prompts[0]
    assert "Teacher approach plan at t=8: Align the gripper." in processor.prompts[1]

    class Model:
        def __init__(self):
            self.value = torch.ones(1, 8, 7, requires_grad=True)

        def __call__(self, inputs, proprio, state, *, batch_idx):
            text = processor.prompts[inputs["input_ids"].item() - 1]
            assert any(row.startswith("Teacher approach plan") for row in text)
            output = PolicyOutput(self.value, self.value)
            output.recurrent_state = torch.ones(1)
            return output

    monkeypatch.setattr(qwen_vla_task.torch, "autocast", lambda *a, **kw: nullcontext())
    model = Model()
    terms = []
    plan_log = StringIO()
    loss, _, _ = qwen_vla_task.action_loss(model, data, processor, QwenVLAConfig(),
        torch.device("cpu"), plan_teacher=teacher, plan_log=plan_log, action_weight=.5, loss_terms=terms)
    result = loss((current, None, previous), 0)
    assert result.item() == 1.375 and terms == [{"task_l1": 1., "action_kd_l1": .75}]
    result.backward()
    assert model.value.grad[:, :-1].abs().sum() > 0 and torch.all(model.value.grad[:, -1] == 0)
    assert "Teacher approach plan at t=0: Align the gripper." in processor.prompts[-1]
    val_loss, _, _ = qwen_vla_task.action_loss(model, data, processor, QwenVLAConfig(),
        torch.device("cpu"), plan_teacher=teacher, plan_log=plan_log, phase="validation")
    assert val_loss((current, None, previous), 0).item() == 1.
    assert teacher.calls == [("action_and_plan", 8), ("validation", 8), ("action_and_plan", 8),
                             ("train", 0), ("validation", 8), ("validation", 0)]
    common = {"episode_id": "a", "timestep": 8, "instruction": "place the cube",
              "optimizer_step": 1, "status": "ok", "task_l1": 1.}
    assert list(map(json.loads, plan_log.getvalue().splitlines())) == [
        {**common, "split": "train", "plan": "Approach the cube.", "action_kd_l1": .75},
        {**common, "split": "validation", "optimizer_step": 0, "plan": "Align the gripper.", "action_kd_l1": None},
    ]


def test_failed_plan_is_recorded_at_its_query():
    def fail(sample):
        raise TimeoutError("T1 timed out")

    plan_log = StringIO()
    loss, _, _ = qwen_vla_task.action_loss(None, Data(), Processor(), QwenVLAConfig(), "cpu",
        plan_teacher=SimpleNamespace(get_signal=fail), plan_log=plan_log)
    with pytest.raises(TimeoutError):
        loss((query(16), None, None), 7)
    row = json.loads(plan_log.getvalue())
    assert (row["status"], row["error_type"], row["timestep"], row["optimizer_step"]) == (
        "plan_error", "TimeoutError", 16, 8,
    )


def test_select_uses_final_validation_gt_l1_and_rejects_unmatched_runs(tmp_path):
    def make(name, weight, loss, *, step=2, seed=42):
        path = tmp_path / name
        path.mkdir()
        settings = {"max_steps": 2, "accumulation_steps": 2, "seed": seed, "distillation": {
            "teacher_mode": "inference", "teacher_plan_max_new_tokens": 48,
            "task_weight": 1., "action_kd_weight": weight,
        }}
        (path / "identity.json").write_text(json.dumps({"settings": settings, "teacher_inference": {"step": 5}}))
        (path / "metrics.json").write_text(json.dumps({"step": step, "validation_l1": loss, "loss": -weight}))
        seal_checkpoint(path, step, step * 2)
        (path / "COMPLETE").write_text("ok\n")
        return path

    a, b = make("a", .25, .42), make("b", 1., .45)
    assert select([b, a])["selected"]["lambda"] == .25
    with pytest.raises(ValueError, match="identities differ"):
        select([a, make("wrong-seed", .5, .2, seed=9)])
    with pytest.raises(ValueError, match="final-step"):
        select([a, make("early", .5, .2, step=1)])
