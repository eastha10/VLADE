"""Check published trained-Student planning without weights or GPU work."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from vlade.evaluation.quantization_comparison import input_records, load_config, plan


def configuration(tmp_path, **changes):
    registry = json.loads((ROOT / 'configs/quantization/s0_assets.json').read_text())
    trained = copy.deepcopy(registry['assets']['S0'])
    trained.update(format='hub_checkpoint',
                   prefix='runs/example-gt_only/gt_only/checkpoints/step-0000100',
                   local_name='step-0000100', revision='1' * 40)
    trained.update(changes)
    registry['assets']['student_gt_full100'] = trained
    registry_path = tmp_path / 'registry.json'
    registry_path.write_text(json.dumps(registry))
    cfg = load_config(ROOT / 'configs/quantization/s0_comparison.json')
    cfg.update(checkpoint='hfasset://student_gt_full100', registry=str(registry_path), rollouts=4)
    return cfg


def test_published_student_plan_uses_selected_checkpoint(tmp_path):
    cfg = configuration(tmp_path)
    result = plan(cfg)
    assert input_records(cfg)['checkpoint']['prefix'].startswith('runs/example-gt_only/')
    assert result['inputs']['checkpoint']['revision'] == '1' * 40
    assert len(result['conditions']) == 8
    assert result['gpu_execution'] is False
    assert result['downloads'] is False
    assert result['automatic_winner'] is None


@pytest.mark.parametrize('changes', [
    {'revision': 'main'},
    {'repo_type': 'dataset'},
    {'repo_id': 'other/private-model'},
    {'files': {'model/weights': {'bytes': 1, 'sha256': '0' * 64}}},
])
def test_trained_student_keeps_existing_provenance_gate(tmp_path, changes):
    with pytest.raises(ValueError):
        plan(configuration(tmp_path, **changes))


@pytest.mark.parametrize('protected_kind',['model','data','source','registered','hf_cache'])
def test_shared_benchmark_workspace_cannot_write_inside_inputs(tmp_path,monkeypatch,protected_kind):
    from vlade.evaluation.s0_quantization import benchmark_workspace
    from vlade.storage import local_inputs
    protected=tmp_path/protected_kind
    monkeypatch.setattr(local_inputs,'registered_roots',lambda:[protected] if protected_kind=='registered' else [])
    monkeypatch.setenv('VLADE_BENCHMARK_WORK',str(protected/'runtime'))
    if protected_kind=='hf_cache':monkeypatch.setenv('HF_HUB_CACHE',str(protected))
    with pytest.raises(ValueError):
        benchmark_workspace(tmp_path/'output/report.json',inputs=(protected,) if protected_kind in ('model','data','source') else ())


def test_shared_benchmark_workspace_can_use_new_output(tmp_path,monkeypatch):
    from vlade.evaluation.s0_quantization import benchmark_workspace
    from vlade.storage import local_inputs
    monkeypatch.setattr(local_inputs,'registered_roots',lambda:[])
    monkeypatch.setenv('VLADE_BENCHMARK_WORK',str(tmp_path/'shared-runtime'))
    assert benchmark_workspace(tmp_path/'output/report.json',inputs=(tmp_path/'model',))==tmp_path/'shared-runtime'
