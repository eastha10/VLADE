"""Real cache-to-sampler and CPU gradient/resume checks without model downloads."""
import os
import io
import json
from itertools import islice
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(os.environ["VLADE_PROJECT_ROOT"]) / "src"))
from vlade.data.colab_assets import sha256
from vlade.data.episode_data import EpisodeData, prepare
from vlade.distillation.action import action_distillation_loss
from vlade.models.contracts import PolicyOutput
from vlade.training.cached_actions import CachedActions
from vlade.training.fixed_cache import generate
from vlade.training.loop import LoopState, make_scheduler, restore_training_state, run_loop, training_state
from vlade.training.student_task import batches, identity, run, samples
from vlade.models.students.qwen import StudentConfig


@pytest.fixture
def prepared(tmp_path):
    def episode(index):
        image = io.BytesIO()
        Image.new("RGB", (8, 8), (index, 10, 20)).save(image, format="PNG")
        return [{"action": np.array([index + t / 10] * 6 + [1], dtype=np.float32),
                 "observation": {"image": image.getvalue(), "wrist_image": image.getvalue(),
                                 "state": np.arange(8, dtype=np.float32) + index + t},
                 "language_instruction": f"move {index}".encode()} for t in range(10)]
    prepare(iter(episode(i) for i in range(30)), tmp_path / "data")
    return EpisodeData(tmp_path / "data")


class Teacher:
    def reset(self):
        pass

    def predict(self, sample, data):
        return torch.from_numpy(sample["gt_action"] * .5), torch.zeros(56, 4)


def make_cache(data, root, *, pilot=False):
    selection = dict(checkpoint="selected-T1", checkpoint_sha256="a" * 64, teacher_step=12,
                     frozen=True, quantized=False,
                     normalization_sha256=data.manifest["normalization_sha256"],
                     split_sha256=data.manifest["split_sha256"])
    generate(data, Teacher(), selection, root, max_episodes=2 if pilot else 0)
    return root


def test_cache_sampler_preserves_order_targets_history_and_resume(prepared, tmp_path):
    cache = CachedActions(make_cache(prepared, tmp_path / "cache"), prepared)
    stream = list(islice(samples(prepared, "train", repeat=True, seed=9), 20))
    resumed = list(islice(samples(prepared, "train", repeat=True, seed=9, skip=7), 13))
    assert [(q["episode_id"], q["timestep"]) for q, _ in stream[7:]] == [
        (q["episode_id"], q["timestep"]) for q, _ in resumed]
    for sample, history in stream:
        torch.testing.assert_close(cache.get(sample), torch.from_numpy(sample["gt_action"] * .5))
        assert all(q["episode_id"] == sample["episode_id"] for q in history)
        assert history[-1]["timestep"] == sample["timestep"]

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.ones(1, 8, dtype=torch.long)}

    batch = next(batches(prepared, Processor(), StudentConfig(), split="train", teacher=cache, seed=9))
    assert len(batch) == 5
    torch.testing.assert_close(batch[4], batch[2] * .5)
    assert batch[4].shape == (1, 8, 7)
    with pytest.raises(ValueError, match="train split"):
        next(batches(prepared, Processor(), StudentConfig(), split="validation", teacher=cache))


def test_partial_cache_rejected_for_full_training(prepared, tmp_path):
    root = make_cache(prepared, tmp_path / "pilot", pilot=True)
    with pytest.raises(ValueError, match="cover|complete"):
        CachedActions(root, prepared)
    cache = CachedActions(root, prepared, episode_limit=1)
    assert cache.signature["scope"] == "pilot"


def test_preflight_requires_no_gpu_and_resume_rejects_changed_cache(prepared, tmp_path, capsys, monkeypatch):
    # Synthetic fixtures isolate training contracts from HF acquisition.
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda value: Path(value))
    root = make_cache(prepared, tmp_path / "cache")
    checkpoint = tmp_path / "initial"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"synthetic-preflight-only")
    config = StudentConfig()
    (checkpoint / "download_manifest.json").write_text(json.dumps({
        "repository": config.model_id, "revision": config.revision,
        "files": [{"path": "model.safetensors", "sha256": sha256(checkpoint / "model.safetensors")}]}))
    settings = dict(method="action_kd", batch_size=1, max_steps=1, save_steps=1, accumulation_steps=1,
                    task_weight=1., action_kd_weight=1., data=str(prepared.root),
                    initial_checkpoint=str(checkpoint), teacher_cache=str(root), student={})
    run(settings, preflight=True)
    assert '"status": "preflight_ok"' in capsys.readouterr().out
    saved = tmp_path / "completed"
    saved.mkdir()
    (saved / "COMPLETE").write_text("ok")
    (saved / "identity.json").write_text(json.dumps(identity(settings, prepared)))
    manifest_path = root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audit_note"] = "changed cache identity"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="resume configuration/data/code/environment mismatch"):
        run(settings, resume=saved, preflight=True)


@pytest.mark.parametrize("damage", ["records", "part", "data", "statistics", "observation", "aggregate"])
def test_cache_rejects_tampering_and_cross_data(prepared, tmp_path, damage):
    root = make_cache(prepared, tmp_path / "cache")
    manifest_path = root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    folder = root / manifest["episodes"][0]["episode_id"]
    config_path = root / "cache_config.json"
    config = json.loads(config_path.read_text())
    if damage == "records":
        with (folder / "records.jsonl").open("ab") as handle:
            handle.write(b"{}\n")
    elif damage == "part":
        (folder / "manifest.json").write_text("{}")
    elif damage == "data":
        config["data_manifest_sha256"] = "0" * 64
        config_path.write_text(json.dumps(config))
    elif damage == "statistics":
        config["selection"]["normalization_sha256"] = "0" * 64
        config_path.write_text(json.dumps(config))
    elif damage == "aggregate":
        manifest["queries"] += 1
        manifest_path.write_text(json.dumps(manifest))
    else:
        # Even a correctly re-sealed record must match its actual observation.
        path = folder / "records.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["observation_refs"]["primary"] = "episodes/wrong/image.png"
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path.write_bytes(payload)
        part_path = folder / "manifest.json"
        part = json.loads(part_path.read_text())
        part.update(sha256=sha256(path), bytes=len(payload))
        part_path.write_text(json.dumps(part))
        manifest["episodes"][0]["manifest_sha256"] = sha256(part_path)
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, KeyError)):
        CachedActions(root, prepared)


def test_loss_masks_padding_detaches_teacher_and_updates_student():
    actions = torch.nn.Parameter(torch.ones(1, 8, 7))
    teacher = torch.full_like(actions, 2., requires_grad=True)
    target = torch.zeros_like(actions)
    valid = torch.ones_like(actions, dtype=torch.bool)
    valid[:, -1] = False
    target[:, -1] = float("nan")
    with torch.no_grad():
        teacher[:, -1] = float("nan")
    loss, terms = action_distillation_loss(PolicyOutput(actions, actions), target, teacher, valid,
                                          task_weight=1., action_weight=.5)
    assert loss.item() == 1.5 and terms == {"task_l1": 1., "action_kd_l1": 1.}
    loss.backward()
    assert teacher.grad is None and actions.grad[:, -1].abs().sum() == 0
    assert actions.grad[:, :-1].abs().sum() > 0


def test_kd_optimizer_state_resume_matches_uninterrupted_run(tmp_path):
    def components():
        model = torch.nn.Linear(1, 56)
        with torch.no_grad():
            model.weight.fill_(.2)
            model.bias.fill_(.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        return model, optimizer, make_scheduler(optimizer, 4)

    def train(model, optimizer, scheduler, steps, state=None):
        def loss_fn(batch, step):
            action = model(torch.ones(1, 1)).view(1, 8, 7)
            return action_distillation_loss(PolicyOutput(action, action), torch.zeros_like(action),
                torch.full_like(action, .4), torch.ones_like(action, dtype=torch.bool))[0]
        return run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler, batches=iter(range(20)),
                        loss_fn=loss_fn, max_steps=steps, accumulation_steps=2, state=state)
    full, opt, sched = components()
    train(full, opt, sched, 4)
    part, opt, sched = components()
    state = train(part, opt, sched, 2)
    saved = tmp_path / "resume.pt"
    torch.save({"model": part.state_dict(), "trainer": training_state(state, opt, sched)}, saved)
    restored, opt, sched = components()
    payload = torch.load(saved, weights_only=True)
    restored.load_state_dict(payload["model"])
    state = restore_training_state(payload["trainer"], opt, sched)
    assert (state.step, state.microsteps) == (2, 4)
    train(restored, opt, sched, 4, state)
    for expected, actual in zip(full.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_notebook_is_opt_in_and_keeps_task_only_separate():
    root=Path(os.environ['VLADE_PROJECT_ROOT'])
    for name,profile in [('colab_05_qwen_action_distillation.ipynb','qwen_kd'),('colab_03_qwen_task_only.ipynb','qwen_task')]:
        nb=json.loads((root/'experiments'/name).read_text(encoding='utf-8'))
        cells={row['id']:''.join(row['source']) for row in nb['cells']}
        assert 'RUN_SETUP = False' in cells['config'] and 'RUN_EXPERIMENT = False' in cells['config']
        assert profile+'.json' in cells['config']
        for row in nb['cells']:
            if row['cell_type']=='code': compile(''.join(row['source']),row['id'],'exec')
    kd=json.loads((root/'configs/research/qwen_kd.json').read_text())
    task=json.loads((root/'configs/research/qwen_task.json').read_text())
    assert kd['distillation']['method']=='action_kd' and 'distillation' not in task
    assert kd['pilot']['overfit_steps']>=10
