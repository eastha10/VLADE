"""Private atomic Hub commits, outage recovery and exact CPU resume contracts."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.training.hub import HubCheckpoints, download_checkpoint
from vlade.training.loop import LoopState, make_scheduler, restore_training_state, training_state
from vlade.training.run_control import seal_checkpoint, verify_checkpoint


class MemoryHub:
    def __init__(self, *, private=True):
        self.private = private
        self.files = {}
        self.fail = False
        self.commits = []

    def create_repo(self, **kwargs):
        assert kwargs['private'] is True

    def model_info(self, *args, **kwargs):
        return SimpleNamespace(private=self.private, sha='immutable-sha')

    def create_commit(self, **kwargs):
        # A client/retry implementation must not perturb the training RNG.
        torch.rand(3)
        if self.fail:
            raise ConnectionError('secret-must-never-be-logged')
        result = {}
        for op in kwargs['operations']:
            content = op.path_or_fileobj
            result[op.path_in_repo] = Path(content).read_bytes() if isinstance(content, str) else content.getvalue()
        self.files.update(result)
        self.commits.append(result)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv('VLADE_HUB_REPO', 'test/vlade-private')
    monkeypatch.setenv('VLADE_HUB_RUN_ID', 'seed42')
    monkeypatch.setattr('vlade.training.hub.time.sleep', lambda _: None)


def checkpoint(output, hub, step, value=None, *, trainer=None, model=None):
    path = output / f'step-{step:07d}'
    path.mkdir(parents=True)
    torch.save(trainer or {'optimizer': {}, 'scheduler': {}, 'rng': {}}, path / 'trainer_state.pt')
    torch.save(model or {}, path / 'model.pt')
    (path / 'tokenizer.json').write_text('{}')
    hub.prepare(path, step, {} if value is None else {'val_l1': value})
    seal_checkpoint(path, step, step)
    (path / 'COMPLETE').write_text('ok\n')
    return path


def test_atomic_last_best_final_and_resume_best(configured, tmp_path):
    api = MemoryHub()
    output = tmp_path / 'train'
    hub = HubCheckpoints(output, api=api)
    first = checkpoint(output, hub, 1, .2)
    hub.publish(first)
    resumed = HubCheckpoints(output, resume=first, api=api)
    assert not resumed.is_best({'val_l1': .3})
    assert not resumed.is_best({'val_l1': float('nan')})
    second = checkpoint(output, resumed, 2, .3)
    resumed.publish(second, final=True)
    prefix = 'runs/seed42/train/'
    assert json.loads(api.files[prefix+'best.json'])['step'] == 1
    for name in ('last', 'final'):
        assert json.loads(api.files[prefix+name+'.json'])['step'] == 2
    for commit in api.commits:
        for name, raw in commit.items():
            if name.endswith(('/last.json', '/best.json', '/final.json')):
                assert json.loads(raw)['checkpoint']+'/COMPLETE' in commit
    assert not resumed.is_best({})
    assert resumed.is_best({'val_l1': .1})


def test_disabled_does_not_import_or_connect(monkeypatch, tmp_path):
    monkeypatch.delenv('VLADE_HUB_REPO', raising=False)
    hub = HubCheckpoints(tmp_path)
    hub.prepare(tmp_path, 1, {})
    hub.publish(tmp_path)
    assert hub.flush() and not list(tmp_path.iterdir())


def test_public_and_missing_run_refused(configured, tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match='privacy'):
        HubCheckpoints(tmp_path/'train', api=MemoryHub(private=False))
    monkeypatch.delenv('VLADE_HUB_RUN_ID')
    with pytest.raises(ValueError, match='run names'):
        HubCheckpoints(tmp_path/'train', api=MemoryHub())


def test_outage_fifo_recovery_and_sanitized_log(configured, tmp_path, capsys):
    api = MemoryHub()
    api.fail = True
    output = tmp_path/'train'
    hub = HubCheckpoints(output, api=api)
    first = checkpoint(output, hub, 1, .3)
    hub.publish(first)
    second = checkpoint(output, hub, 2, .1)
    hub.publish(second, final=True)
    pending = json.loads((output/'hub_pending.json').read_text())
    assert len(pending['queue']) == 2 and verify_checkpoint(second)
    assert 'secret-must-never' not in capsys.readouterr().out
    assert 'secret-must-never' not in (output/'hub_pending.json').read_text()
    api.fail = False
    retry = HubCheckpoints(output, resume=second, api=api)
    assert retry.flush()
    assert [json.loads(c['runs/seed42/train/last.json'])['step'] for c in api.commits] == [1, 2]
    assert json.loads((output/'hub_pending.json').read_text())['queue'] == []


def test_tamper_incomplete_and_weight_only_refused(configured, tmp_path):
    hub = HubCheckpoints(tmp_path/'train', api=MemoryHub())
    saved = checkpoint(hub.output, hub, 1, .4)
    (saved/'trainer_state.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='integrity'):
        hub.publish(saved)
    (saved/'COMPLETE').unlink()
    with pytest.raises(ValueError, match='Completed'):
        hub.publish(saved)
    (saved/'trainer_state.pt').unlink()
    seal_checkpoint(saved, 1, 1)
    (saved/'COMPLETE').touch()
    with pytest.raises(ValueError, match='resumable'):
        hub.publish(saved)


def test_private_flip_does_not_upload(configured, tmp_path):
    api = MemoryHub()
    hub = HubCheckpoints(tmp_path/'train', api=api)
    saved = checkpoint(hub.output, hub, 1, .1)
    api.private = False
    hub.publish(saved)
    assert not api.commits and hub.queue


def test_upload_download_restores_optimizer_scheduler_rng(configured, tmp_path, monkeypatch):
    import huggingface_hub
    api = MemoryHub()
    output = tmp_path/'train'
    hub = HubCheckpoints(output, api=api)
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = make_scheduler(optimizer, 4, 1, 'cosine')

    def update():
        optimizer.zero_grad()
        model(torch.randn(2, 2)).square().mean().backward()
        optimizer.step()
        scheduler.step()

    update()
    saved = checkpoint(output, hub, 1, .2, trainer=training_state(LoopState(1, 1), optimizer, scheduler),
                       model=copy.deepcopy(model.state_dict()))
    hub.publish(saved)
    update()
    expected = copy.deepcopy(model.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    cache = tmp_path/'hf-cache'
    for name, content in api.files.items():
        path = cache/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: api)
    def download(repo, filename, *, revision):
        assert revision == 'immutable-sha'
        return str(cache/filename)
    def snapshot(repo, *, revision, allow_patterns):
        assert revision == 'immutable-sha' and len(allow_patterns) == 1
        return str(cache)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    restored = download_checkpoint('test/vlade-private', 'seed42/train', tmp_path/'resume')
    model.load_state_dict(torch.load(restored/'model.pt', weights_only=True))
    payload = torch.load(restored/'trainer_state.pt', weights_only=True)
    state = restore_training_state(payload, optimizer, scheduler)
    assert (state.step, state.microsteps) == (1, 1)
    update()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert scheduler.state_dict() == expected_scheduler
    with pytest.raises(FileExistsError):
        download_checkpoint('test/vlade-private', 'seed42/train', tmp_path/'resume')
