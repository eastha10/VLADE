"""CPU-only framework checks; never load production weights or data."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))
from vlade.quantization.vision.quantizer import VisionQuantizer, fake_quantize
from vlade.quantization.vision.sensitivity import losses, measure
from vlade.quantization.vision.allocation import allocate
from vlade.quantization.vision import runner


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(4, 4))
        self.mlp = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))
        self.norm = nn.LayerNorm(4)

    def forward(self, x):
        return self.norm(x + self.attn(x) + self.mlp(x))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(Block(), Block())
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 4, 2)
        self.head = nn.Linear(4, 4)

    def forward(self, x):
        return self.blocks(x)


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.featurizer = Backbone()
        self.fused_featurizer = Backbone()

    def forward(self, x):
        return torch.cat((self.featurizer(x), self.fused_featurizer(x)), -1)


@pytest.fixture
def model():
    torch.manual_seed(3)
    return Vision().eval()


def test_fp_restore_and_scope(model):
    parent = nn.ModuleDict({"vision": model, "llm": nn.Linear(4, 4),
                            "projector": nn.Linear(8, 4), "ava": nn.Linear(4, 1)})
    original = copy.deepcopy(parent.state_dict())
    x = torch.randn(2, 5, 4)
    reference = model(x).detach().clone()
    q = VisionQuantizer(model)
    q.apply({})
    assert torch.equal(model(x), reference)
    for bits in (4, 8):
        q.apply(dict.fromkeys(q.groups, bits))
        changed = {k for k, v in parent.state_dict().items() if not torch.equal(v, original[k])}
        assert changed
        assert changed <= {"vision." + k + ".weight" for k in q.groups}
        q.restore()
        assert all(torch.equal(v, original[k]) for k, v in parent.state_dict().items())
        assert torch.equal(model(x), reference)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantization_grid(bits):
    w = torch.tensor([[0., .12, -.81, 1.], [0., 0., 0., 0.]])
    result = fake_quantize(w, bits)
    assert torch.isfinite(result).all()
    assert torch.equal(result[1], w[1])
    limit = 2 ** (bits - 1) - 1
    assert torch.allclose(result[0] * limit, (result[0] * limit).round())
    assert (result - w).abs().max() <= .5 / limit + 1e-6


def test_patch_and_roles(model):
    q = VisionQuantizer(model, quantize_dinov2=False, quantize_patch_embed=True)
    assert all(k.startswith("fused_featurizer.") for k in q.groups)
    assert "fused_featurizer.patch_embed.proj" in q.groups
    assert not any("head" in k or "norm" in k for k in q.groups)


def test_exception_restores(model):
    q = VisionQuantizer(model)
    with pytest.raises(RuntimeError):
        with q.temporary({next(iter(q.groups)): 4}):
            raise RuntimeError("probe failure")
    assert all(torch.equal(m.weight, q.original[k]) for k, m in q.groups.items())


def test_sensitivity_isolated_and_fixed(model):
    q = VisionQuantizer(model)
    x = [torch.randn(1, 3, 4), torch.randn(1, 3, 4)]
    context = torch.randn(1, 3, 8)
    fixed = context.clone()
    def evaluate(query):
        changed = [k for k, m in q.groups.items() if not torch.equal(m.weight, q.original[k])]
        assert len(changed) <= 1
        z = model(query)
        return z, torch.sigmoid(z + context)
    result = measure(q, x, evaluate, alpha=.5, beta=2., batch_size=2)
    assert torch.equal(context, fixed)
    assert all(torch.equal(m.weight, q.original[k]) for k, m in q.groups.items())
    for row in result.values():
        for scores in row.values():
            assert scores["combined"] == pytest.approx(.5 * scores["visual"] + 2 * scores["ava"])
    second = measure(q, x, evaluate, alpha=.5, beta=2., batch_size=1)
    assert result == second


def test_measure_failure_restores(model):
    q = VisionQuantizer(model)
    count = 0
    def fail(query):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("failed")
        return torch.zeros(1), torch.ones(1)
    with pytest.raises(RuntimeError):
        measure(q, [None], fail, alpha=1, beta=1)
    assert all(torch.equal(m.weight, q.original[k]) for k, m in q.groups.items())


def test_loss_stability_and_formula():
    z = torch.zeros(3)
    assert losses((z, z), (z, z)) == (0, 0)
    v, a = losses((torch.ones(2), torch.ones(2)), (torch.zeros(2), torch.ones(2) * 2))
    assert v == pytest.approx(1)
    assert a == pytest.approx(float(torch.log(torch.tensor(2.)).square()))
    with pytest.raises(ValueError):
        losses((z, z), (torch.full((3,), float("nan")), z))
    with pytest.raises(ValueError):
        losses((z, -torch.ones(3)), (z, z))


@pytest.mark.parametrize("target", [4, 4.01, 5, 6, 7.999, 8])
def test_budget_and_reproducibility(target):
    sizes = {"a": 3, "b": 6, "c": 9}
    scores = {n: {"4": {"combined": s}, "8": {"combined": .1}} for n, s in zip(sizes, [3, 2, 1])}
    for seed in (None, 42):
        bits = allocate(sizes, scores, target, random_seed=seed)
        assert set(bits.values()) <= {4, 8}
        assert sum(sizes[n] * b for n, b in bits.items()) <= target * sum(sizes.values())
        assert bits == allocate(sizes, scores, target, random_seed=seed)


def test_negative_gain_and_greedy_order():
    scores = {n: {"4": {"combined": s}, "8": {"combined": 1}} for n, s in [("a", 3), ("b", 2), ("c", 0)]}
    assert allocate({"a": 1, "b": 2, "c": 1}, scores, 5) == {"a": 8, "b": 4, "c": 4}
    assert allocate({"a": 1, "b": 2, "c": 1}, scores, 8)["c"] == 4


def test_config_and_no_download_preflight(monkeypatch):
    cfg = runner.load_config(ROOT / "configs/quantization/vision_ptq.json")
    assert [r["method"] for r in cfg["ablations"].values()] == ["fp", "w8", "w4", "random", "visual", "ava", "combined"]
    monkeypatch.setattr(runner, "inspect_hf", lambda records: (_ for _ in ()).throw(AssertionError("network")))
    result = runner.preflight(cfg, remote=False)
    assert result["weight_or_dataset_downloads"] is False and result["gpu_execution"] is False
    assert result["inputs"]["checkpoint"]["revision"] == "1fdc672416f2129624b22022e6977345a3b429e3"
    assert runner.backbone_roles({"timm_model_ids": ["vit_dinov2", "vit_siglip"]}) == {
        "featurizer": "dinov2", "fused_featurizer": "siglip"}


@pytest.mark.parametrize("key,value", [("method", "bad"), ("batch_size", 0), ("beta", -1),
    ("target_avg_bits", 9), ("seed", True), ("calibration_data", "local"), ("quantize_siglip", 1)])
def test_invalid_config(tmp_path, key, value):
    cfg = json.loads((ROOT / "configs/quantization/vision_ptq.json").read_text())
    cfg[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        runner.load_config(path)


def test_runner_json_and_restore(model, tmp_path, monkeypatch):
    import vlade.storage.assets as assets
    import vlade.data.episode_data as episodes
    import vlade.training.teacher_inference as teacher
    cfg = runner.load_config(ROOT / "configs/quantization/vision_ptq.json")
    cfg.update(method="combined", output_root=str(tmp_path), device="cpu")
    (tmp_path / "config.json").write_text(json.dumps({"timm_model_ids": ["vit_dinov2", "vit_siglip"]}))
    data = SimpleNamespace(manifest={"dataset_revision": "test", "split_sha256": "s", "normalization_sha256": "n"})
    monkeypatch.setattr(assets, "resolve_input", lambda *a, **k: tmp_path)
    monkeypatch.setattr(episodes, "EpisodeData", lambda path: data)
    monkeypatch.setattr(teacher, "teacher_identity", lambda *a: (tmp_path, {"temporal_feature_layer": -2}))
    monkeypatch.setattr(runner, "inspect_hf", lambda *a: {})
    original = copy.deepcopy(model.state_dict())
    x = [torch.randn(1, 3, 4)]
    def evaluate(query):
        z = model(query)
        return z, torch.sigmoid(z)
    monkeypatch.setattr(runner, "calibration_queries", lambda *a: (
        SimpleNamespace(model=nn.ModuleDict({"vision_backbone": model})),
        [{"episode_id": "e", "timestep": 0}], lambda query: evaluate(x[0])))
    for method in runner.METHODS:
        paths = runner.execute({**cfg, "method": method}, pilot=True)
        assert len(paths) == 1
        assert {p.name for p in paths[0].iterdir()} == {
            "config.json", "manifest.json", "sensitivity.json", "bit_allocation.json", "quantization_summary.json"}
        summary = json.loads((paths[0] / "quantization_summary.json").read_text())
        assert summary["integer_acceleration"] is False
        if method in {"mixed", "random", "visual", "ava", "combined"}:
            assert summary["effective_avg_bits"] <= cfg["target_avg_bits"]
        assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


def test_capture_calibration_and_native_ava_contexts(model, monkeypatch):
    import vlade.training.fixed_cache as fixed
    from vlade.models.teachers.components.attn_weight_generator import AttentionWeightGenerator
    cfg = runner.load_config(ROOT / "configs/quantization/vision_ptq.json")
    cfg.update(calibration_episodes=2, max_queries_per_episode=2, device="cpu")
    generator = AttentionWeightGenerator(embed_dim=128, image_shape=(2, 2, 8),
        action_shape=(2, 2, 8), text_dim=8, num_images=1, max_steps=None).eval()
    parent = nn.ModuleDict({"vision_backbone": model, "projector": nn.Linear(8, 8)})
    generator_before = copy.deepcopy(generator.state_dict())
    class Policy:
        def __init__(self, *args, **kwargs):
            self.model, self.generator = parent, generator
            self.context = None
        def reset(self):
            self.context = None
        def predict(self, sample, data):
            x = torch.full((1, 4, 4), (sample["timestep"] + 1) / 10.)
            z = parent.projector(parent.vision_backbone(x))
            generator(z, self.context, torch.ones(1, 2, 8), batch_idx=None)
            self.context = torch.full((1, 4, 8), float(sample["timestep"] + 1))
    entries = [{"episode_id": "a", "split": "train", "is_calibration": True},
               {"episode_id": "b", "split": "train", "is_calibration": False},
               {"episode_id": "c", "split": "validation", "is_calibration": False},
               {"episode_id": "d", "split": "train", "is_calibration": True}]
    class Data:
        def __init__(self):
            self.entries = entries
        def episodes(self, split, order):
            assert split == "train" and set(order) == {"a", "d"}
            for key in order:
                yield {"episode_id": key}, [{"timestep": i} for i in (0, 8, 16)]
    monkeypatch.setattr(fixed, "SequentialTeacher", Policy)
    policy, queries, evaluate = runner.calibration_queries(cfg, Path("unused"), Data(), {"temporal_feature_layer": -2})
    assert len(queries) == 4
    assert queries[0]["action"] is None and queries[2]["action"] is None
    assert torch.equal(queries[1]["action"], torch.ones(1, 4, 8))
    assert torch.equal(queries[3]["action"], torch.ones(1, 4, 8))
    saved_context = queries[1]["action"].clone()
    before = evaluate(queries[1])
    after = evaluate(queries[1])
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert torch.equal(queries[1]["action"], saved_context)
    assert all(torch.equal(v, generator_before[k]) for k, v in generator.state_dict().items())
    assert not parent.vision_backbone._forward_pre_hooks and not generator._forward_pre_hooks


def test_cli_all_offline_no_model_load(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ptq", "--config", str(ROOT / "configs/quantization/vision_ptq.json"),
                                     "--all", "--dry-run", "--offline"])
    monkeypatch.setattr(runner, "execute", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model load")))
    runner.main()
    assert capsys.readouterr().out.count('"status": "configuration_valid"') == 7
