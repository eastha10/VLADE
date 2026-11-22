"""CPU-only checks; production input/model/GPU boundaries are mocked explicitly."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))
from vlade.evaluation import quantization_comparison as comparison
from vlade.evaluation import s0_quantization as evaluation


@pytest.fixture
def cfg():
    value = comparison.load_config(ROOT / "configs/quantization/s0_comparison.json")
    value.update(calibration_queries=3, eval_queries=3, warmup=1)
    return value


@pytest.mark.parametrize("edit", [
    lambda c: c.update(seeds=[True]),
    lambda c: c.update(seeds=[42, 42]),
    lambda c: c.update(calibration_queries=2),
    lambda c: c.update(phase="final"),
    lambda c: c.update(data="local-data"),
    lambda c: c["conditions"][4].update(scope="language_vision"),
    lambda c: c["conditions"][4].update(target_bpw=4),
    lambda c: c["conditions"][1].update(target_bpw=4),
    lambda c: c["conditions"][1].update(id="BF16"),
    lambda c: c["conditions"].pop(0),
])
def test_invalid_conditions_rejected(cfg, edit):
    edit(cfg)
    with pytest.raises(ValueError):
        comparison.validate_config(cfg)


def test_plan_is_download_and_model_free(cfg, monkeypatch):
    import vlade.storage.assets as assets
    def forbidden(*a, **kw):
        raise AssertionError("No input resolution/model work in planning")
    monkeypatch.setattr(assets, "resolve_input", forbidden)
    monkeypatch.setattr(assets, "fetch_asset", forbidden)
    proposal = comparison.plan(cfg)
    assert len(proposal["conditions"]) == 8
    assert proposal["gpu_execution"] is False and proposal["downloads"] is False
    assert proposal["automatic_winner"] is None and proposal["selection_eligible"] is False
    assert len({r["comparison_group"] for r in proposal["conditions"][1:5]}) == 1
    assert len({r["comparison_group"] for r in proposal["conditions"][5:]}) == 1


@pytest.fixture
def synthetic_runtime(tmp_path, monkeypatch):
    import vlade.storage.assets as assets
    import vlade.data.episode_data as episode_data
    import vlade.training.run_control as control
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "manifest.json").write_text('{"normalization_sha256":"n","split_sha256":"s"}')
    data_hash = hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
    (checkpoint / "identity.json").write_text(json.dumps({"data_sha256": data_hash}))
    (checkpoint / "checkpoint_manifest.json").write_text('{}')
    monkeypatch.setattr(assets, "resolve_input", lambda v, **kw: checkpoint if str(v) == "hfasset://S0" else dataset)
    monkeypatch.setattr(control, "verify_checkpoint", lambda p: {"step": 400})
    class Data:
        def __init__(self, path):
            self.root = dataset
            self.manifest = {"normalization_sha256": "n", "split_sha256": "s"}
            self.entries = [{"split": "train", "is_calibration": True, "episode_id": "train-cal"},
                            {"split": "train", "is_calibration": False, "episode_id": "not-cal"},
                            {"split": "validation", "is_calibration": False, "episode_id": "validation"}]
        def episodes(self, split, order):
            assert set(order) <= ({"train-cal"} if split == "train" else {"validation"})
            for key in order:
                yield {"episode_id": key}, [{"episode_id": key, "timestep": i * 8,
                    "instruction": str(i + 1), "proprio": np.zeros(4, np.float32),
                    "gt_action": np.full((2, 4), .1 * (i + 1), np.float32),
                    "action_valid_mask": np.ones((2, 4), bool)} for i in range(6)]
        def image(self, sample, camera):
            return sample["instruction"]
    monkeypatch.setattr(episode_data, "EpisodeData", Data)
    loaded = []
    class ToyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.language_model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))
            self.backbone.visual = nn.Module()
            self.backbone.visual.blocks = nn.Sequential(nn.Linear(4, 4))
            self.action_head = nn.Linear(4, 4)
            self.student_config = SimpleNamespace(image_size=2)
            with torch.no_grad():
                for p in self.parameters():
                    p.copy_(torch.arange(p.numel()).reshape_as(p) * .013 + .007)
        @classmethod
        def load(cls, path):
            model = cls()
            loaded.append(copy.deepcopy(model.state_dict()))
            return model
        def forward(self, inputs, proprio, state):
            x = self.backbone.visual.blocks(inputs["x"])
            x = self.backbone.language_model(x)
            actions = self.action_head(x)
            return SimpleNamespace(actions=actions, recurrent_state=actions)
    processor = ModuleType("transformers")
    processor.AutoProcessor = SimpleNamespace(from_pretrained=lambda *a, **kw: object())
    model_module = ModuleType("vlade.models.students.qwen_vla")
    model_module.QwenVLA = ToyPolicy
    preparation = ModuleType("vlade.models.students.backbones.qwen")
    preparation.prepare_observation = lambda *a, instruction, **kw: {"x": torch.full((1, 2, 4), float(instruction) / 10.)}
    for name, module in (("transformers", processor), (model_module.__name__, model_module), (preparation.__name__, preparation)):
        monkeypatch.setitem(sys.modules, name, module)
    original_version = evaluation.importlib.metadata.version
    monkeypatch.setattr(evaluation.importlib.metadata, "version", lambda name: "synthetic" if name == "transformers" else original_version(name))
    monkeypatch.setattr(evaluation, "evaluation_device", lambda: torch.device("cpu"))
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "synthetic CPU control-flow check")
    source = lambda: {"git_sha": "f" * 40, "source_dirty": False}
    monkeypatch.setattr(comparison, "source_identity", source)
    monkeypatch.setattr(evaluation, "source_identity", source)
    calls = []
    def worker(command, log):
        calls.append(command)
        monkeypatch.setattr(sys, "argv", command[2:])
        evaluation.main()
        return 0
    return SimpleNamespace(worker=worker, calls=calls, loaded=loaded)


def test_all_conditions_real_cpu_quantization_and_resume(cfg, tmp_path, synthetic_runtime):
    runtime = synthetic_runtime
    output = tmp_path / "comparison"
    result = comparison.run_comparison(cfg, output, worker=runtime.worker)
    assert result["status"] == "completed"
    assert len(result["jobs"]) == 8 and len(runtime.calls) == 8
    assert len(runtime.loaded) == 8
    for state in runtime.loaded[1:]:
        assert all(torch.equal(v, runtime.loaded[0][key]) for key, v in state.items())
    for key, job in result["jobs"].items():
        report = json.loads((output / job["report"]).read_text())
        assert report["status"] == "completed"
        assert len(report["runs"]) == 1
        assert len(next(iter(report["runs"].values()))["samples"]["validation"]) == 3
    summary = json.loads((output / "summary.json").read_text())
    assert summary["automatic_winner"] is None
    comparison.run_comparison(cfg, output, resume=True, worker=lambda *a: pytest.fail("Completed measurements must be reused"))
    assert len(runtime.calls) == 8


def test_failure_keeps_baseline_and_resume_retries_only_failed(cfg, tmp_path, synthetic_runtime):
    cfg["conditions"] = cfg["conditions"][:3]
    output = tmp_path / "failed"
    def fail_act(command, log):
        if command[command.index("--condition") + 1] == "actquant":
            return 1
        return synthetic_runtime.worker(command, log)
    first = comparison.run_comparison(cfg, output, worker=fail_act)
    assert first["status"] == "incomplete"
    baseline = output / first["jobs"]["seed-42/BF16"]["report"]
    baseline_bytes = baseline.read_bytes()
    second = comparison.run_comparison(cfg, output, resume=True, worker=synthetic_runtime.worker)
    assert second["status"] == "completed"
    assert baseline.read_bytes() == baseline_bytes
    assert second["jobs"]["seed-42/L8-ActQuant"]["attempt"] == 2
    assert len(synthetic_runtime.calls) == 3


def test_baseline_failure_blocks_other_jobs(cfg, tmp_path, synthetic_runtime):
    calls = []
    def failed(command, log):
        calls.append(command)
        return 1
    result = comparison.run_comparison(cfg, tmp_path / "blocked", worker=failed)
    assert len(calls) == 1
    assert result["jobs"]["seed-42/BF16"]["status"] == "failed"
    assert all(j["status"] == "blocked" for k, j in result["jobs"].items() if not k.endswith("/BF16"))


def test_resume_rejects_changed_config_and_report(cfg, tmp_path, synthetic_runtime):
    cfg["conditions"] = cfg["conditions"][:1]
    output = tmp_path / "tamper"
    result = comparison.run_comparison(cfg, output, worker=synthetic_runtime.worker)
    with pytest.raises(ValueError, match="identical config"):
        comparison.run_comparison({**cfg, "eval_queries": 4}, output, resume=True, worker=synthetic_runtime.worker)
    report = output / result["jobs"]["seed-42/BF16"]["report"]
    report.write_text(report.read_text() + "\n")
    with pytest.raises(ValueError, match="report changed"):
        comparison.run_comparison(cfg, output, resume=True, worker=synthetic_runtime.worker)


def test_legacy_failure_saves_completed_bf16(cfg, tmp_path, synthetic_runtime, monkeypatch):
    import vlade.evaluation.qwen_quantization as quantization
    output = tmp_path / "legacy.json"
    monkeypatch.setattr(sys, "argv", ["worker", "--checkpoint", cfg["checkpoint"], "--data", cfg["data"],
        "--registry", str(comparison.registry_path(cfg)), "--output", str(output), "--actquant", "on",
        "--calibration-queries", "3", "--eval-queries", "3", "--warmup", "0"])
    def fail(*a, **kw):
        raise RuntimeError("Synthetic calibration failure")
    monkeypatch.setattr(quantization, "calibrate", fail)
    monkeypatch.setenv("VLADE_ASSET_REGISTRY", "previous-registry")
    with pytest.raises(RuntimeError):
        evaluation.main()
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and "bf16" in report["runs"]
    assert len(report["reference_actions"]) == 3
    assert os.environ["VLADE_ASSET_REGISTRY"] == "previous-registry"


def test_cli_rejects_mixed_independent_and_unsupported_quant(cfg, tmp_path, monkeypatch):
    base = ["worker", "--checkpoint", cfg["checkpoint"], "--data", cfg["data"], "--output", str(tmp_path / "r.json")]
    for suffix in (["--condition", "actquant", "--qvla", "on"],
                   ["--condition", "quant", "--target-bpw", "4"],
                   ["--condition", "quant", "--target-bpw", "8", "--scope", "language_vision"]):
        monkeypatch.setattr(sys, "argv", base + suffix)
        with pytest.raises(SystemExit):
            evaluation.parse_args()


def test_dirty_source_rejected_before_worker(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "source_identity", lambda: {"git_sha": "a" * 40, "source_dirty": True})
    with pytest.raises(ValueError, match="Commit the reviewed"):
        comparison.run_comparison(cfg, tmp_path / "not-created", worker=lambda *a: pytest.fail("Must not execute"))
    assert not (tmp_path / "not-created").exists()


def test_atomic_json_rejects_nan_without_overwriting(tmp_path):
    path = tmp_path / "report.json"
    comparison.atomic_json(path, {"valid": 1})
    with pytest.raises(ValueError):
        comparison.atomic_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"valid": 1}


def test_scope_excludes_projector_head_and_ava():
    parent = nn.Module()
    parent.backbone = nn.Module()
    parent.backbone.language_model = nn.Sequential(nn.Linear(4, 4))
    parent.backbone.visual = nn.Module()
    parent.backbone.visual.blocks = nn.Sequential(nn.Linear(4, 4))
    parent.action_head = nn.Linear(4, 4)
    parent.ava = nn.Linear(4, 4)
    assert set(evaluation.select_layers(parent, "language")) == {"backbone.language_model.0"}
    assert set(evaluation.select_layers(parent, "language_vision")) == {"backbone.language_model.0", "backbone.visual.blocks.0"}


def test_output_cannot_write_into_asset_cache(cfg, tmp_path, monkeypatch):
    cache = tmp_path / "existing-cache"
    monkeypatch.setenv("VLADE_ASSET_CACHE", str(cache))
    with pytest.raises(ValueError, match="outside dataset/checkpoint/HF caches"):
        comparison.run_comparison(cfg, cache / "new-output", worker=lambda *a: pytest.fail("No worker"))
    assert not cache.exists()


def test_pairing_rejects_different_inputs_queries_and_fake_budget(cfg, tmp_path, synthetic_runtime):
    cfg["conditions"] = cfg["conditions"][:2]
    output = tmp_path / "pairing"
    manifest = comparison.run_comparison(cfg, output, worker=synthetic_runtime.worker)
    baseline = json.loads((output / manifest["jobs"]["seed-42/BF16"]["report"]).read_text())
    report = json.loads((output / manifest["jobs"]["seed-42/L8-Uniform"]["report"]).read_text())
    for field in ("source", "hf_inputs", "seed", "eval_queries", "checkpoint_seal_sha256", "calibration_queries", "warmup", "dtype", "gpu"):
        changed = copy.deepcopy(report)
        changed[field] = "different"
        with pytest.raises(ValueError, match="identity mismatch"):
            comparison.paired_result(changed, baseline, cfg["conditions"][1])
    for edit in (
        lambda r: r["runs"]["uniform_w8"]["samples"]["validation"].reverse(),
        lambda r: r["runs"]["uniform_w8"].update(weight_bpw_excluding_metadata=4),
        lambda r: r["runs"]["uniform_w8"].update(weight_bpw_excluding_metadata=float("nan")),
        lambda r: r["runs"]["uniform_w8"].update(action_mae_vs_ground_truth=float("nan")),
    ):
        changed = copy.deepcopy(report)
        edit(changed)
        with pytest.raises(ValueError):
            comparison.paired_result(changed, baseline, cfg["conditions"][1])


def test_requested_rollout_failure_is_not_completed(cfg, tmp_path, synthetic_runtime):
    cfg["conditions"] = cfg["conditions"][:1]
    output = tmp_path / "rollout-status"
    manifest = comparison.run_comparison(cfg, output, worker=synthetic_runtime.worker)
    report = json.loads((output / manifest["jobs"]["seed-42/BF16"]["report"]).read_text())
    report["rollout_count_requested"] = 4
    report["runs"]["bf16"]["rollout"] = {"status": "unavailable", "reason": "synthetic missing simulator"}
    assert comparison.paired_result(report, report, cfg["conditions"][0])["status"] == "rollout_unavailable"
