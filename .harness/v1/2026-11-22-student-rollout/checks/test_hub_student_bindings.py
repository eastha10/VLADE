"""A trained Hub checkpoint keeps immutable remote and actual-byte checks."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from vlade.storage import assets, local_inputs


@pytest.fixture(params=['hub_checkpoint', 'directory', 'upstream_files'])
def hub(tmp_path, monkeypatch, request):
    root = tmp_path / 'student'
    root.mkdir()
    payload = b'trained student'
    (root / 'weights.bin').write_bytes(payload)
    record = {'repo_id': 'VLADE-konkuk/VLADE-checkpoints', 'repo_type': 'model',
              'revision': 'a' * 40, 'format': request.param,
              'prefix': 'runs/example/gt_only/checkpoints/step-0000100',
              'local_name': 'step-0000100',
              'files': {'weights.bin': {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}}
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'schema_version': 1, 'assets': {'trained': record}}))
    remote = ('weights.bin' if request.param == 'upstream_files' else
              record['prefix'] + ('/files/' if request.param == 'directory' else '/') + 'weights.bin')
    entry = SimpleNamespace(rfilename=remote, size=len(payload),
                            lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()))
    info = SimpleNamespace(sha=record['revision'], siblings=[entry])
    monkeypatch.setattr(assets, 'configure_http', lambda: None)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(repo_info=lambda *a, **kw: info))
    bindings = tmp_path / 'bindings.json'
    monkeypatch.setenv('VLADE_LOCAL_INPUTS', str(bindings))
    return root, registry, bindings, info, entry


def test_bound_hub_checkpoint_reuses_verified_bytes_and_rejects_tampering(hub, monkeypatch):
    root, registry, bindings, _, _ = hub
    local_inputs.register_local_input('trained', root, bindings, registry=registry)
    monkeypatch.setattr(assets, 'fetch_asset', lambda *a, **kw: pytest.fail('No redownload fallback'))
    assert assets.resolve_input('hfasset://trained', registry=registry) == root
    (root / 'weights.bin').write_bytes(b'corrupted input')
    with pytest.raises(ValueError):
        assets.resolve_input('hfasset://trained', registry=registry)


@pytest.mark.parametrize('damage', ['revision', 'remote_hash', 'remote_layout'])
def test_remote_hub_checkpoint_pin_mismatch_never_registers(hub, damage):
    root, registry, bindings, info, entry = hub
    if damage == 'revision':
        info.sha = 'b' * 40
    elif damage == 'remote_hash':
        entry.lfs.sha256 = 'b' * 64
    else:
        entry.rfilename = 'different-layout/' + entry.rfilename
    with pytest.raises(ValueError):
        local_inputs.register_local_input('trained', root, bindings, registry=registry)
    assert not bindings.exists()
