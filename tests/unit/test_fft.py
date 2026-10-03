"""CPU-only checks for FFT configuration, preflight, and full gradients."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlade.training.fft import (  # noqa: E402
    FFTConfig, compute_loss, find_component, preflight, register_libero_plus_dataset, save_checkpoint,
)


def test_config_loads_distinct_runs():
    path = ROOT / "configs/experiments/libero_plus_fft.json"
    ava = FFTConfig.from_json(path, "ava_vla")
    oft = FFTConfig.from_json(path, "openvla_oft")
    assert ava.checkpoint != oft.checkpoint
    assert ava.optimizer == "paged_adamw_8bit"
    assert ava.dataset_name == "libero_plus_mixdata"


def test_preflight_requires_component_checkpoints(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").touch()
    for filename in ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"):
        (checkpoint / filename).touch()
    data_root = tmp_path / "data"
    (data_root / "libero_plus_mixdata").mkdir(parents=True)
    (data_root / "libero_plus_mixdata" / "dataset_info.json").touch()
    cfg = FFTConfig("ava_vla", checkpoint, data_root, tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="action_head"):
        preflight(cfg)


def test_find_component_rejects_ambiguous_checkpoint(tmp_path):
    for step in (1, 2):
        (tmp_path / f"action_head--{step}_checkpoint.pt").touch()
    with pytest.raises(FileNotFoundError, match="found 2"):
        find_component(tmp_path, "action_head")


def test_config_rejects_non_fft_optimizer(tmp_path):
    raw = json.loads((ROOT / "configs/experiments/libero_plus_fft.json").read_text(encoding="utf-8"))
    raw["fft_defaults"]["optimizer"] = "unknown"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer"):
        FFTConfig.from_json(path, "ava_vla")


def test_libero_plus_rlds_uses_disk_builder(monkeypatch, tmp_path):
    tfds = ModuleType("tensorflow_datasets")
    tfds.builder = lambda name, **kwargs: (name, kwargs)
    tfds.builder_from_directory = lambda path: path
    configs = ModuleType("prismatic.vla.datasets.rlds.oxe.configs")
    configs.OXE_DATASET_CONFIGS = {"libero_4_task_suites_no_noops": {"image_obs_keys": {"primary": "image"}}}
    transforms = ModuleType("prismatic.vla.datasets.rlds.oxe.transforms")
    transforms.OXE_STANDARDIZATION_TRANSFORMS = {"libero_4_task_suites_no_noops": object()}
    for name in ("prismatic", "prismatic.vla", "prismatic.vla.datasets", "prismatic.vla.datasets.rlds",
                 "prismatic.vla.datasets.rlds.oxe"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", tfds)
    monkeypatch.setitem(sys.modules, "prismatic.vla.datasets.rlds.oxe.configs", configs)
    monkeypatch.setitem(sys.modules, "prismatic.vla.datasets.rlds.oxe.transforms", transforms)
    version = tmp_path / "libero_plus_mixdata" / "1.0.0"
    version.mkdir(parents=True)
    (version / "dataset_info.json").touch()
    register_libero_plus_dataset("libero_plus_mixdata")
    assert tfds.builder("libero_plus_mixdata", data_dir=str(tmp_path)) == str(version)
    assert "libero_plus_mixdata" in configs.OXE_DATASET_CONFIGS


def test_continuous_loss_updates_backbone_and_ava_components(monkeypatch):
    utils = ModuleType("prismatic.training.train_utils")
    utils.get_current_action_mask = lambda labels: torch.ones_like(labels, dtype=torch.bool)
    utils.get_next_actions_mask = lambda labels: torch.zeros_like(labels, dtype=torch.bool)
    constants = ModuleType("prismatic.vla.constants")
    constants.ACTION_DIM = 7
    constants.NUM_ACTIONS_CHUNK = 8
    for name in ("prismatic", "prismatic.training", "prismatic.vla"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "prismatic.training.train_utils", utils)
    monkeypatch.setitem(sys.modules, "prismatic.vla.constants", constants)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, **kwargs):
            value = self.scale + kwargs["proprio_projector"].scale
            value = value + kwargs["vision_attn_weight_generator"].scale
            return SimpleNamespace(hidden_states=[value * torch.ones(1, 59, 1)])

    class DummyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def predict_action(self, hidden):
            return hidden.reshape(1, 8, 7) * self.scale

    class DummyComponent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

    model, head, proprio, generator = DummyModel(), DummyHead(), DummyComponent(), DummyComponent()
    batch = {
        "input_ids": torch.ones(1, 57, dtype=torch.long),
        "attention_mask": torch.ones(1, 57, dtype=torch.bool),
        "pixel_values": torch.ones(1, 2),
        "labels": torch.ones(1, 57, dtype=torch.long),
        "proprio": torch.ones(8),
        "actions": torch.zeros(1, 8, 7),
    }
    loss = compute_loss(model, head, proprio, generator, batch, torch.device("cpu"), 2, 0)
    loss.backward()
    assert loss.item() > 0
    assert all(component.scale.grad is not None for component in (model, head, proprio, generator))


def test_full_checkpoint_saves_model_and_all_components(monkeypatch, tmp_path):
    utilities = ModuleType("prismatic.vla.datasets.rlds.utils.data_utils")
    utilities.save_dataset_statistics = lambda stats, path: (path / "dataset_statistics.json").write_text(
        json.dumps(stats), encoding="utf-8"
    )
    for name in ("prismatic", "prismatic.vla", "prismatic.vla.datasets", "prismatic.vla.datasets.rlds",
                 "prismatic.vla.datasets.rlds.utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "prismatic.vla.datasets.rlds.utils.data_utils", utilities)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(total_memory=1024))

    class DummySaver(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def save_pretrained(self, path, **kwargs):
            torch.save(self.state_dict(), path / "model.pt")

    class DummyProcessor:
        def save_pretrained(self, path):
            (path / "processor.json").write_text("{}", encoding="utf-8")

    source = tmp_path / "source"
    source.mkdir()
    for name in ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"):
        (source / name).write_text("# model code\n", encoding="utf-8")
    cfg = FFTConfig("ava_vla", source, tmp_path, tmp_path / "out")
    module = DummySaver()
    result = save_checkpoint(cfg, 1, module, DummyProcessor(), module, module, module, None,
                             SimpleNamespace(dataset_statistics={"ok": True}))
    assert (result / "COMPLETE").is_file()
    assert (result / "model.pt").is_file()
    assert (result / "vision_attn_weight_generator--1_checkpoint.pt").is_file()
    assert (result / "modeling_prismatic.py").is_file()
    assert json.loads((result / "vlade_fft.json").read_text(encoding="utf-8"))["step"] == 1
