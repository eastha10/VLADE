"""Bounded CPU checks; no HF transfer, AVA installation or GPU experiment."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))

from vlade.data.colab_assets import sha256
from vlade.models.students.qwen import StudentConfig, QwenStudent
from vlade.research.configuration import load, validate
from vlade.training.run_control import seal_checkpoint
from vlade.training.student_initialization import initial_checkpoint
from vlade.training.teacher_inference import EpisodeTeacher, InferenceActions, teacher_identity


class Data:
    def __init__(self, root):
        self.root = root
        root.mkdir()
        (root / "manifest.json").write_text("fixture")
        self.manifest = {"normalization_sha256": "n" * 64, "split_sha256": "s" * 64}

    def episodes(self, split, *, order):
        assert split == "train"
        if order[0] not in {"a", "b"}:
            raise ValueError("Not a train episode")
        yield {}, [{"episode_id": order[0], "timestep": i * 8} for i in range(3)]


class Policy:
    def __init__(self):
        self.resets, self.calls = 0, []
        self.state = 0

    def reset(self):
        self.resets += 1
        self.state = 0

    def predict(self, sample, data):
        self.calls.append((sample["episode_id"], sample["timestep"]))
        self.state = self.state * 2 + sample["timestep"] + 1
        return torch.full((8, 7), float(self.state)), None


def seal(root, step=4):
    seal_checkpoint(root, step, step)
    (root / "COMPLETE").write_text("ok\n")


def s0(root, data, config):
    (root / "model").mkdir(parents=True)
    (root / "model/student.json").write_text(json.dumps(asdict(config)))
    (root / "model/action_head.pt").write_bytes(b"head")
    (root / "model/backbone").mkdir()
    (root / "model/backbone/config.json").write_text("{}")
    (root / "model/backbone/model.safetensors").write_bytes(b"synthetic-backbone")
    (root / "processor").mkdir()
    (root / "processor/tokenizer.json").write_text("{}")
    (root / "identity.json").write_text(json.dumps({"data_manifest_sha256": sha256(data.root / "manifest.json")}))
    seal(root)


def t1(root, data):
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"synthetic-backbone")
    for name in ("action_head", "proprio_projector", "vision_attn_weight_generator"):
        (root / (name + "--4_checkpoint.pt")).write_bytes(b"synthetic-component")
    (root / "vlade_fft.json").write_text(json.dumps({"model": "ava_vla", "step": 4,
        "prepared_data_identity": {"manifest_sha256": sha256(data.root / "manifest.json"),
            **{key: data.manifest[key] for key in ("normalization_sha256", "split_sha256")}}}))
    seal(root)


def test_episode_inference_replays_resume_prefix_and_resets(tmp_path):
    data, policy = Data(tmp_path / "data"), Policy()
    teacher = EpisodeTeacher(policy, data)
    expected = teacher.get("a", 16)
    assert policy.calls == [("a", 0), ("a", 8), ("a", 16)]
    torch.testing.assert_close(teacher.get("a", 16), expected)
    assert policy.resets == 2
    teacher.get("b", 0)
    assert policy.resets == 3 and policy.calls[-1] == ("b", 0)
    teacher.get("b", 8)
    assert policy.resets == 3
    teacher.get("b", 0)
    assert policy.resets == 4
    with pytest.raises(ValueError):
        teacher.get("a", 3)
    with pytest.raises(ValueError):
        teacher.get("a", 24)
    with pytest.raises(ValueError):
        teacher.get("test-episode", 0)


def test_s0_and_t1_require_hf_provenance_before_reading(tmp_path):
    data = Data(tmp_path / "data")
    for value in ("unregistered-local", tmp_path / "unregistered"):
        with pytest.raises(ValueError, match="HF"):
            teacher_identity(value, data)
        with pytest.raises(ValueError, match="HF"):
            initial_checkpoint({"initial_checkpoint": str(value)}, data, StudentConfig())


def test_verified_s0_requires_config_data_and_complete_seal(tmp_path, monkeypatch):
    data, config = Data(tmp_path / "data"), StudentConfig()
    root = tmp_path / "s0"
    s0(root, data, config)
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda value: root)
    settings = {"initial_checkpoint": "hfasset://s0", "initialization": "student_checkpoint"}
    path, identity = initial_checkpoint(settings, data, config)
    assert path == root and identity["initial_student_checkpoint"]["step"] == 4
    (root / "model/action_head.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        initial_checkpoint(settings, data, config)
    seal(root)
    (root / "model/student.json").write_text(json.dumps({**asdict(config), "schema_version": 2}))
    seal(root)
    with pytest.raises(ValueError, match="architecture"):
        initial_checkpoint(settings, data, config)
    (root / "model/student.json").write_text(json.dumps(asdict(config)))
    seal(root)
    (data.root / "manifest.json").write_text("other-data")
    with pytest.raises(ValueError, match="different prepared data"):
        initial_checkpoint(settings, data, config)


def test_t1_identity_rejects_cross_data_and_tampered_weight(tmp_path, monkeypatch):
    data, root = Data(tmp_path / "data"), tmp_path / "t1"
    t1(root, data)
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda value: root)
    _, signature = teacher_identity("hfasset://t1", data)
    assert signature["frozen"] and signature["step"] == 4
    data.manifest["split_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="matching data"):
        teacher_identity("hfasset://t1", data)
    (root / "vlade_fft.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        teacher_identity("hfasset://t1", data)


def test_actual_json_pipe_detaches_actions_and_closes_worker(tmp_path, monkeypatch):
    signature = {"step": 4}
    worker = tmp_path / "fake_worker.py"
    worker.write_text('''import json, sys
print(json.dumps({"ready": {"step": 4}}), flush=True)
for line in sys.stdin:
    value = json.loads(line)
    value["action"] = [[0.25] * 7 for _ in range(8)]
    print(json.dumps(value), flush=True)
''')
    popen = subprocess.Popen
    monkeypatch.setattr("vlade.training.teacher_inference.teacher_identity", lambda *a: (tmp_path, signature))
    monkeypatch.setattr("vlade.training.teacher_inference.subprocess.Popen",
        lambda *a, **kw: popen([sys.executable, "-B", str(worker)], **kw))
    teacher = InferenceActions({"teacher_checkpoint": "hfasset://t1", "teacher_python": sys.executable,
        "teacher_timeout_seconds": 10}, SimpleNamespace(root=tmp_path))
    try:
        action = teacher.get({"episode_id": "a", "timestep": 8})
        assert action.shape == (8, 7) and not action.requires_grad and action.mean() == .25
        from vlade.distillation.action import action_distillation_loss
        from vlade.models.contracts import PolicyOutput
        student = torch.ones(1, 8, 7, requires_grad=True)
        loss, terms = action_distillation_loss(PolicyOutput(student, student), torch.zeros_like(student),
            action[None], torch.ones_like(student, dtype=torch.bool), action_weight=.5)
        assert loss.item() == 1.375
        loss.backward()
        assert student.grad.abs().sum() > 0 and action.grad is None
    finally:
        teacher.close()
    assert teacher.process.poll() is not None


def test_config_selects_full_s0_online_t1_without_cache():
    config = load(ROOT / "configs/research/qwen_kd.json")
    assert config["initialization"] == "student_checkpoint"
    assert config["checkpoint"] == "hfasset://student_s0"
    assert config["distillation"]["teacher_checkpoint"] == "hfasset://teacher_t1"
    assert "teacher_cache" not in config["distillation"]
    config["distillation"]["action_kd_weight"] = float("nan")
    with pytest.raises(ValueError):
        validate(config)


def test_full_s0_restores_trained_head_and_native_backbone(tmp_path, monkeypatch):
    from transformers import Qwen3_5Config, Qwen3_5Model, Qwen3_5TextConfig, Qwen3_5VisionConfig
    text = Qwen3_5TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=64,
        layer_types=["full_attention"], rope_parameters={"rope_type": "default", "rope_theta": 10000,
        "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]})
    vision = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=32,
        num_heads=2, out_hidden_size=16, deepstack_visual_indexes=[])
    config = StudentConfig(hidden_size=16, head_size=8, head_heads=2)
    model = QwenStudent(Qwen3_5Model(Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict())), config)
    with torch.no_grad():
        model.head.action[-1].bias.fill_(.375)
    model.save(tmp_path / "model")
    restored = QwenStudent.load(tmp_path / "model")
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)


def test_sequential_teacher_is_frozen_and_uses_saved_temporal_layer(tmp_path, monkeypatch):
    import numpy as np
    from vlade.training.fixed_cache import SequentialTeacher

    (tmp_path / "vlade_fft.json").write_text(json.dumps({"configuration": {"temporal_feature_layer": -1}}))

    class Model(torch.nn.Linear):
        llm_dim = 4

        def predict_action(self, **kwargs):
            assert not self.training and not torch.is_grad_enabled()
            assert kwargs["temporal_feature_layer"] == -1
            assert "gt_action" not in kwargs and "labels" not in kwargs
            return np.zeros((8, 7)), torch.ones(1, 56, 4)

    model, modules = Model(1, 1), [torch.nn.Linear(1, 1) for _ in range(3)]
    def processor(*args, **kwargs):
        return {"input_ids": torch.ones(1, 2, dtype=torch.long), "pixel_values": torch.ones(1, 1, 2)}
    monkeypatch.setattr("vlade.training.fixed_cache.configure_tensorflow_cpu", lambda: None)
    monkeypatch.setattr("vlade.training.fft._make_model_and_components", lambda *a: (model, processor, *modules))
    teacher = SequentialTeacher(tmp_path, device="cpu", temporal_feature_layer=-1)
    assert all(not p.requires_grad for module in (model, *modules) for p in module.parameters())
    teacher.predict({"instruction": "move", "proprio": np.zeros(8)}, SimpleNamespace(image=lambda *a: None))
    assert teacher.context.shape == (1, 56, 4) and not teacher.context.requires_grad
    teacher.reset()
    assert teacher.context is None


def test_online_s0_preflight_needs_no_cache_or_teacher_gpu(tmp_path, monkeypatch, capsys):
    from vlade.training.student_task import run

    data = Data(tmp_path / "data")
    data.manifest["counts"] = {"train": 1}
    student, teacher = tmp_path / "s0", tmp_path / "t1"
    s0(student, data, StudentConfig())
    t1(teacher, data)
    paths = {"hfasset://s0": student, "hfasset://t1": teacher, "hfasset://data": data.root}
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda value: paths.get(str(value), Path(value)))
    monkeypatch.setattr("vlade.training.student_task.EpisodeData", lambda value: data)
    def no_worker(*a, **kw):
        raise AssertionError("preflight must not load the Teacher")
    monkeypatch.setattr("vlade.training.teacher_inference.InferenceActions", no_worker)
    run({"method": "action_kd", "teacher_mode": "inference", "teacher_checkpoint": "hfasset://t1",
         "initial_checkpoint": "hfasset://s0", "initialization": "student_checkpoint", "student": {},
         "data": "hfasset://data", "batch_size": 1, "max_steps": 1, "save_steps": 1,
         "accumulation_steps": 1, "task_weight": 1., "action_kd_weight": 1.}, preflight=True)
    report = json.loads(capsys.readouterr().out)
    assert report["teacher_cache"] is None and report["teacher_inference"]["step"] == 4


def test_unified_notebook_keeps_execution_opt_in_and_task_only_separate():
    notebook = json.loads((ROOT / "experiments/colab_vlade.ipynb").read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    assert "RUN_SETUP = False" in cells["config"] and "RUN_EXPERIMENT = False" in cells["config"]
    assert "qwen_kd" in cells["config"] and "qwen_task" in cells["config"]
    assert load(ROOT / "configs/research/qwen_task.json").get("initialization", "pretrained_backbone") == "pretrained_backbone"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
