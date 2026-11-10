"""Standalone research controls, real tiny Llama and rollout accounting."""

import copy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(os.environ["VLADE_PROJECT_ROOT"])


def test_every_research_plan_is_side_effect_free(tmp_path, capsys):
    from vlade.research.cli import main

    work = tmp_path / "not-created"
    for config in (ROOT / "configs/research").glob("*.json"):
        assert main(["--config", str(config), "--workspace", str(work)]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["third_party_required"] is False
        assert plan["execution"] == "not_started"
    assert not work.exists()


def test_fft_controls_reach_the_original_trainer(tmp_path, monkeypatch):
    from vlade.research.configuration import load
    from vlade.research.execution import dispatch
    import vlade.training.fft as fft

    config = load(ROOT / "configs/research/ava_fft.json", [
        "train.max_steps=7", "train.warmup_steps=1", "train.save_steps=3",
        "train.accumulation_steps=2", "train.learning_rate=0.0001",
    ])
    captured = []
    monkeypatch.setattr(fft, "train", captured.append)
    dispatch(config, ROOT, tmp_path, tmp_path / "fft")
    actual = captured[0]
    assert (actual.max_steps, actual.warmup_steps, actual.save_steps) == (7, 1, 3)
    assert actual.grad_accumulation_steps == 2 and actual.learning_rate == 0.0001


def test_rollout_count_and_initial_states_are_paired(tmp_path, monkeypatch):
    from vlade.evaluation import rollouts, policies
    from vlade.runtime import benchmark as runtime
    from benchmarks.libero import environment

    monkeypatch.setattr(runtime, "prepare", lambda *a: None)
    package = ModuleType("libero.libero")
    class Suite:
        n_tasks = 2
        def get_task(self, task):
            return SimpleNamespace(language=str(task))
        def get_task_init_states(self, task):
            return [0, 1, 2]
    package.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_10": Suite})
    monkeypatch.setitem(sys.modules, "libero", ModuleType("libero"))
    monkeypatch.setitem(sys.modules, "libero.libero", package)
    loaded = []
    monkeypatch.setattr(policies, "load_policy", lambda *a: loaded.append(a) or object())
    monkeypatch.setattr(environment, "create_environment", lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(rollouts, "episode", lambda *a: {"success": True, "environment_steps": 1})
    config = {"seed": 42, "model": "ava_vla", "checkpoints": {"T0": "first", "T1": "second"},
              "evaluation": {"task_suites": ["libero_10"], "tasks_per_suite": 2, "rollouts_per_task": 3}}
    result = rollouts.run(config, tmp_path / "evaluation", tmp_path)
    assert result["complete"] and result["rollouts"] == 12
    assert len(result["records"]) == result["successes"] == 12
    first = [(r["task_id"], r["initial_state_index"]) for r in result["records"] if r["model"] == "T0"]
    second = [(r["task_id"], r["initial_state_index"]) for r in result["records"] if r["model"] == "T1"]
    assert first == second and len(loaded) == 2
    rollouts.run(config, tmp_path / "evaluation", tmp_path)
    assert len(loaded) == 2
    changed = copy.deepcopy(config)
    changed["evaluation"]["rollouts_per_task"] = 2
    with pytest.raises(ValueError, match="different config"):
        rollouts.run(changed, tmp_path / "evaluation", tmp_path)


def test_episode_obeys_horizon_and_open_loop_controls(monkeypatch):
    from vlade.evaluation.rollouts import episode
    from benchmarks.libero import observation, action

    monkeypatch.setattr(observation, "prepare_observation", lambda value: value)
    monkeypatch.setattr(action, "to_environment", lambda value: value)
    calls, resets, frames = [], [], []
    env = SimpleNamespace(reset=lambda: None, set_init_state=lambda _: {},
                          step=lambda value: (calls.append(value) or {}, 0, False, {}))
    policy = SimpleNamespace(reset=resets.append, predict=lambda obs, text, step: frames.append(step) or np.zeros((8, 7)))
    result = episode(env, policy, "task", {"settle_steps": 2, "max_steps": 5, "open_loop_steps": 2}, 0, "libero_10")
    assert result == {"success": False, "environment_steps": 5}
    assert len(calls) == 7 and frames == [0, 2, 4] and resets == ["libero_10"]


def test_oft_native_llama_backward_without_ava_decoder_arguments():
    from transformers import LlamaConfig, LlamaForCausalLM
    from vlade.models.teachers.backbones.llama import build_llama_backbone
    from vlade.models.teachers.ava_vla import OpenVLAForActionPrediction
    from test_ava_teacher import tiny_components

    torch.set_num_threads(2)
    config = SimpleNamespace(text_config=LlamaConfig(vocab_size=32064, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=256), extra_attn_weights={}, _attn_implementation="eager")
    decoder = build_llama_backbone(config)
    assert isinstance(decoder, LlamaForCausalLM)
    bundle = tiny_components(OpenVLAForActionPrediction)
    model = bundle["model"]
    model.language_model = decoder
    model.extra_layer_ids = []
    assert model._language_attention_kwargs(None) == {}
    result = model(input_ids=torch.tensor([[1, 10, 29871] + [1]*56 + [2]]),
        attention_mask=torch.ones(1, 60, dtype=torch.long), pixel_values=torch.randn(1, 8, 16),
        labels=torch.tensor([[-100]*3 + [31744]*56 + [2]]), proprio=torch.randn(1, 8),
        proprio_projector=bundle["proprio"], output_hidden_states=True, use_cache=False)
    action = bundle["head"].predict_action(result.hidden_states[-1][:, 12:68])
    action.square().mean().backward()
    assert action.shape == (1, 8, 7) and torch.isfinite(action).all()
    assert decoder.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_benchmark_namespace_preserves_resources_and_rejects_replacement(tmp_path, monkeypatch):
    from vlade.runtime import benchmark_source
    from vlade.utils import source as source_module

    pin = {"repository": "https://example.invalid/benchmark.git", "commit": "a"*40}
    checkout = tmp_path / "checkout"
    package = checkout / "libero/libero"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    python = tmp_path / "environment/bin/python"
    site = python.parent.parent / "site-packages"
    site.mkdir(parents=True)
    monkeypatch.setattr(source_module, "stage_source", lambda *a: checkout)
    def output(command, **kwargs):
        if command[0] == str(python):
            return str(site)
        if "rev-parse" in command:
            return pin["commit"]
        if "remote" in command:
            return pin["repository"]
        return ""
    monkeypatch.setattr(benchmark_source.subprocess, "check_output", output)
    benchmark_source.install({"benchmark_source": pin}, tmp_path, python, {})
    benchmark_source.verify({"benchmark_source": pin}, python)
    (site / "vlade-benchmark.pth").write_text(str(tmp_path / "other"))
    with pytest.raises(ValueError, match="source changed"):
        benchmark_source.verify({"benchmark_source": pin}, python)


def test_suite_main_requires_its_completed_pilot_and_verified_map(tmp_path, monkeypatch):
    from vlade.research.suite import prepare_phase
    from vlade.storage import assets
    from vlade.training.run_control import seal_checkpoint
    from vlade.utils.source_identity import source_identity
    from vlade.data.episode_data import write_json

    mapping = tmp_path / "verified-map"
    mapping.mkdir()
    (mapping / "suite_map.json").write_text('{"by_instruction":{"task":"goal"}}')
    verified = []
    monkeypatch.setattr(assets, "resolve_input", lambda key: verified.append(key) or mapping)
    output = tmp_path / "suite"
    checkpoint = output / "pilot4/checkpoints/step-0000004"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights").write_bytes(b"synthetic")
    seal_checkpoint(checkpoint, 4, 32)
    (checkpoint / "COMPLETE").write_text("ok")
    report = {"status": "completed", "step": 4, "checkpoint": str(checkpoint), "fingerprint": {
        "suite_map.json": assets.sha256(mapping / "suite_map.json"),
        "source_commit": source_identity(ROOT)["git_commit"]}}
    write_json(output / "pilot4/pilot_report.json", report)
    config = {"settings": {"phase": "main", "suite_map": "hfasset://suite_map_seed42"}}
    assert prepare_phase(config, output) == ("main", mapping / "suite_map.json")
    assert verified == ["hfasset://suite_map_seed42"]
    report["status"] = "failed"
    write_json(output / "pilot4/pilot_report.json", report)
    with pytest.raises(ValueError, match="completed four-step pilot"):
        prepare_phase(config, output)
