"""Plan-only relocation retains runtime roots, import contracts and current S2."""
import ast
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / 'AGENTS.md').is_file()
sys.path.insert(0,str(ROOT / 'src'))
BASE = ROOT / 'src/vlade/legacy/plan_cot'


def moved_files():
    tree = ast.parse((Path(__file__).parent / 'migrate_plan_cot.py').read_text())
    assignment = next(node for node in tree.body if isinstance(node,ast.Assign)
                      and any(isinstance(target,ast.Name) and target.id == 'SOURCES' for target in node.targets))
    return assignment.value.func.value.value.splitlines()


def test_all_29_plan_files_share_one_legacy_folder():
    names = moved_files()
    assert len(names) == 29
    for name in names:
        target = BASE / name.removeprefix('src/vlade/')
        assert target.is_file(), target
        if name != 'docs/environment/action_level_distillation.md':
            assert not (ROOT / name).exists(), name
    assert (ROOT / 'experiments/colab_s2_representation_kd_existing_session.ipynb').is_file()


@pytest.mark.parametrize('module',[
    'distillation.teacher_plan','training.teacher_plan_supervision','data.teacher_plan_annotations',
    'evaluation.plan_observations','evaluation.plan_rollout','research.plan_evaluation',
    'research.teacher_plan_training','utils.teacher_plan_training_notebook','utils.t1_plan_probe_cell',
])
def test_legacy_modules_import_from_new_location(module):
    loaded = importlib.import_module('vlade.legacy.plan_cot.'+module)
    assert Path(loaded.__file__).is_relative_to(BASE)
    if hasattr(loaded,'ROOT'): assert loaded.ROOT == ROOT


def test_current_s1_and_representation_s2_conditions_stay_matched():
    from vlade.research.distillation import load_settings, _checkpoint_location
    paths = ROOT / 'configs/experiments'
    s1 = json.loads((paths / 'qwen_vla_action_kd.json').read_text())
    s2 = json.loads((paths / 'qwen_vla_representation_kd.json').read_text())
    rep = s2['distillation'].pop('representation')
    assert s1 == s2 and rep['weight'] == 1
    _, variants = load_settings(paths / 'qwen_vla_representation_kd.json',{})
    assert _checkpoint_location('gt_kd',variants['gt_kd'],'same') == ('same','S2')
    for name in ('qwen_vla_plan_kd','qwen_vla_plan_kd_l4'):
        _, legacy = load_settings(BASE / 'configs/experiments' / (name+'.json'),{})
        assert legacy['gt_kd']['distillation']['teacher_plan_max_new_tokens'] == 48


def test_legacy_teacher_cli_plan_is_read_only(tmp_path):
    script = BASE / 'scripts/train/run_teacher_plan.py'
    work = tmp_path / 'not-created'
    result = subprocess.run([sys.executable,'-B',str(script),'--workspace',str(work),
        '--run-group','legacy-plan','--action','plan'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['settings']['plan_loss_weight'] == 1
    assert not work.exists()


def test_legacy_generators_write_new_paths_and_keep_current_defaults(tmp_path):
    from vlade.utils.action_kd_notebook import write_notebook, write_l4_notebook
    from vlade.legacy.plan_cot.utils.teacher_plan_training_notebook import write_notebook as teacher
    for name,fn,kwargs in [('legacy',write_notebook,{'teacher_plan':True}),
                           ('action',write_notebook,{}),('l4',write_l4_notebook,{}),('teacher',teacher,{})]:
        path = tmp_path / (name+'.ipynb'); fn(path,**kwargs)
        nb = json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']),name,'exec')
                assert cell['outputs'] == []
        code = '\n'.join(''.join(cell['source']) for cell in nb['cells'])
        if name in ('legacy','teacher'): assert 'src/vlade/legacy/plan_cot/configs/' in code
        if name == 'action': assert "CONFIG = 'configs/experiments/qwen_vla_action_kd.json'" in code
        if name == 'l4': assert "'s2_legacy_plan': 'src/vlade/legacy/plan_cot/" in code


def test_no_removed_python_imports_or_config_paths_in_active_sources():
    removed = ['vlade.'+name.removeprefix('src/vlade/').removesuffix('.py').replace('/','.')
               for name in moved_files() if name.startswith('src/')]
    for prefix in ('src','scripts'):
        for path in (ROOT / prefix).rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            for old in removed:
                assert not re.search(r'\b'+re.escape(old)+r'\b',text), (path,old)
            assert 'from .teacher_plan_supervision import' not in text


def test_all_legacy_sources_and_notebook_cells_compile():
    import nbformat
    for path in BASE.rglob('*.py'):
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
    for path in BASE.rglob('*.json'):
        json.loads(path.read_text(encoding='utf-8'))
    for path in BASE.rglob('*.ipynb'):
        notebook = nbformat.read(path,as_version=4); nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code': compile(cell.source,str(path),'exec')


def test_current_and_legacy_markdown_file_links_exist():
    documents = [ROOT / 'README.md',ROOT / 'docs/environment/action_level_distillation.md',
                 ROOT / 'docs/environment/colab_l4_training.md',*BASE.rglob('*.md')]
    for path in documents:
        for target in re.findall(r'\]\(([^\s)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')): continue
            target = target.split('#')[0]
            assert (path.parent / target).resolve().exists(), (path,target)
