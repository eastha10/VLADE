"""CPU checks for measured affordability and optimizer-boundary budget stops."""
import os
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(os.environ["VLADE_PROJECT_ROOT"]) / "src"))
from vlade.training.compute_budget import pilot_profile, plan_fft, should_stop, BudgetExhausted


def profile():
    return dict(step_seconds=10, checkpoint_seconds=100, validation_seconds=20, startup_seconds=60)


def test_budget_caps_time_and_reserves_two_real_saves():
    plan = plan_fft(1_000_000, profile(), 8 * 3600)
    expected = math.floor((8*3600 - 3600 - 300 - 1.3*(60+2*100+2*20)) / 13)
    assert plan["optimizer_steps"] == expected
    assert plan["budget_limited"]
    assert plan["estimated_main_seconds"] + 3600 + 300 <= 8*3600
    assert len(set(range(plan["save_steps"], expected+1, plan["save_steps"])) | {expected}) == 2
    assert plan["warmup_steps"] == math.ceil(expected * .1)


def test_budget_never_adds_epochs_just_to_fill_time():
    plan = plan_fft(80, profile(), 8*3600)
    assert plan["optimizer_steps"] == 10
    assert plan["effective_epochs"] == 1
    assert not plan["budget_limited"]


@pytest.mark.parametrize("remaining", [1, 3600, float("nan"), float("inf"), -1])
def test_refuse_unaffordable_or_invalid_budget(remaining):
    with pytest.raises(ValueError):
        plan_fft(8000, profile(), remaining)


def test_pilot_excludes_warmup_and_requires_completed_timing(tmp_path):
    run = dict(run_id="pilot", status="completed", step=4, startup_seconds=30,
               hardware=dict(gpu="NVIDIA A100", vram_bytes=40*1024**3))
    (tmp_path / "run-pilot.json").write_text(json.dumps(run))
    rows = [dict(step=i, optimizer_step_seconds=s) for i,s in enumerate([99, 2, 3, 4], 1)]
    rows[-1].update(checkpoint_seconds=20, validation_seconds=5)
    (tmp_path / "metrics-pilot.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    measured = pilot_profile(tmp_path)
    assert measured["step_seconds"] == 4
    run["status"] = "budget_exhausted"
    (tmp_path / "run-pilot.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="completed"):
        pilot_profile(tmp_path)


def test_time_stop_saves_only_after_complete_accumulation():
    import torch
    from vlade.training.loop import run_loop, make_scheduler, LoopState
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    state = LoopState()
    saved = []

    def on_step(current, metrics):
        if should_stop(310, 300, 10):
            saved.append((current.step, current.microsteps, model.weight.detach().clone()))
            raise BudgetExhausted()

    with pytest.raises(BudgetExhausted):
        run_loop(modules=[model], optimizer=optimizer, scheduler=make_scheduler(optimizer, 8),
                 batches=[torch.ones(1,1)]*16, loss_fn=lambda batch,step: model(batch).square().mean(),
                 max_steps=8, accumulation_steps=8, max_grad_norm=1., state=state, on_step=on_step)
    assert saved[0][:2] == (1, 8)
    assert state.step == 1 and state.microsteps == 8
    assert not should_stop(1000, 300, 10)


def test_budget_notebook_keeps_managed_patch_and_compiles(tmp_path,monkeypatch):
    from vlade.utils import colab_notebooks
    monkeypatch.setattr(colab_notebooks,'ROOT',tmp_path)
    for path in colab_notebooks.generate()[:-1]:
        nb=json.loads(path.read_text(encoding='utf-8'))
        for c in nb['cells']:
            if c['cell_type']=='code': compile(''.join(c['source']),c['id'],'exec')
    root=Path(os.environ['VLADE_PROJECT_ROOT'])
    assert json.loads((root/'configs/runtime/ava.json').read_text())['managed_ava_patch'] is True
    assert 'runtime.unassign' in (root/'src/vlade/research/client.py').read_text()


def test_fft_budget_exit_is_saved_and_not_reported_as_completed(tmp_path, monkeypatch):
    import time
    import torch
    from types import SimpleNamespace
    from vlade.training import fft, compute_budget, run_control
    monkeypatch.setenv("VLADE_TRAIN_DEADLINE_UNIX", str(time.time()+1000))
    monkeypatch.setattr(compute_budget, "should_stop", lambda *args: True)
    for name in ("set_device", "synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 123)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "fake A100 CPU test")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(total_memory=1000))
    monkeypatch.setattr(fft, "preflight", lambda cfg: None)
    monkeypatch.setattr(fft, "_activate_upstream", lambda name: None)
    monkeypatch.setattr(fft, "set_seed", lambda seed: None)
    monkeypatch.setattr(run_control, "configure_tensorflow_cpu", lambda seed: None)
    monkeypatch.setattr(run_control, "checkpoint_space", lambda *args: 123)
    monkeypatch.setattr(run_control, "gradient_diagnostics", lambda groups: {"tested_separately": True})
    model = torch.nn.Linear(1,1)
    model.vision_backbone = SimpleNamespace(get_num_patches=lambda: 1)
    model.language_model = model.projector = SimpleNamespace()
    head, proprio = torch.nn.Linear(1,1), torch.nn.Linear(1,1)
    monkeypatch.setattr(fft, "_make_model_and_components", lambda *args: (model,None,head,proprio,None))
    monkeypatch.setattr(fft, "_make_loader", lambda *args,**kwargs: ([None]*32, None))
    monkeypatch.setattr(fft, "_make_optimizer", lambda cfg,params: torch.optim.SGD(params, lr=.01))
    monkeypatch.setattr(fft, "compute_loss", lambda *args: sum(p.square().sum() for m in [model,head,proprio] for p in m.parameters()))
    saved = []
    def save(cfg, step, *args, **kwargs):
        saved.append((step, kwargs['loop_state'].microsteps))
        return tmp_path / 'saved'
    monkeypatch.setattr(fft, "save_checkpoint", save)
    cfg = fft.FFTConfig("ava_vla", tmp_path, tmp_path, tmp_path / "run", max_steps=8,
                        grad_accumulation_steps=8, save_steps=4)
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda value: Path(value))
    fft.train(cfg)
    report = json.loads(next(cfg.output_dir.glob('run-*.json')).read_text())
    assert saved == [(1,8)]
    assert report['status'] == 'budget_exhausted' and report['step'] == 1
    metric = json.loads(next(cfg.output_dir.glob('metrics-*.jsonl')).read_text())
    assert metric['optimizer_step_seconds'] > 0 and metric['checkpoint_seconds'] > 0
