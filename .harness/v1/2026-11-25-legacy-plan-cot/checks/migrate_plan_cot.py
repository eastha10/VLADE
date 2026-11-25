"""Move explicitly identified Plan-only files; preserve pending work and all history."""
from pathlib import Path
import json
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / 'AGENTS.md').is_file()
BASE = Path('src/vlade/legacy/plan_cot')
SOURCES = '''src/vlade/distillation/teacher_plan.py
src/vlade/data/teacher_plan_annotations.py
src/vlade/training/teacher_plan_supervision.py
src/vlade/evaluation/plan_observations.py
src/vlade/evaluation/plan_rollout.py
src/vlade/research/plan_evaluation.py
src/vlade/research/teacher_plan_training.py
src/vlade/utils/teacher_plan_training_notebook.py
src/vlade/utils/t1_plan_probe_cell.py
scripts/data/prepare_teacher_plans.py
scripts/train/run_teacher_plan.py
scripts/train/select_plan_kd_weight.py
scripts/prepare/generate_teacher_plan_kd_notebook.py
scripts/prepare/generate_teacher_plan_training_notebook.py
scripts/prepare/generate_plan_evaluation_notebooks.py
configs/experiments/teacher_plan_training.json
configs/experiments/t1_plan_alignment.json
configs/experiments/s1_s2_plan_effect.json
configs/experiments/qwen_vla_plan_kd.json
configs/experiments/qwen_vla_plan_kd_l4.json
experiments/colab_teacher_plan_kd.ipynb
experiments/colab_cot_action_kd_ready.ipynb
experiments/colab_teacher_plan_training.ipynb
experiments/colab_t1_plan_alignment.ipynb
experiments/colab_s1_s2_plan_effect.ipynb
experiments/colab_t1_plan_probe_existing_session.ipynb
docs/environment/teacher_plan_training.md
docs/environment/teacher_plan_generation.md
docs/environment/action_level_distillation.md'''.splitlines()
mapping = {name:(BASE / name.removeprefix('src/vlade/')).as_posix() for name in SOURCES}
modules = {name.removeprefix('src/').removesuffix('.py').replace('/','.'):
           dest.removeprefix('src/').removesuffix('.py').replace('/','.')
           for name,dest in mapping.items() if name.startswith('src/')}
tracked = subprocess.check_output(['git','ls-files'],text=True,cwd=ROOT).splitlines()
eligible = [name for name in tracked if name.startswith(('src/','scripts/','configs/','docs/','experiments/'))
            and not name.startswith(('experiments/runs/','experiments/manifests/'))
            and Path(name).suffix in {'.py','.json','.md','.ipynb'}]
eligible += ['README.md','STRUCTURE.md']
eligible = sorted(set(eligible) | set(SOURCES))
path_pattern = re.compile(r'(?<![\w/])('+'|'.join(re.escape(name) for name in sorted(mapping,key=len,reverse=True))+r')(?![\w])')
module_pattern = re.compile(r'\b('+'|'.join(re.escape(name) for name in sorted(modules,key=len,reverse=True))+r')\b')


def links(text,old,new):
    def replace(match):
        target = match.group(2)
        if re.match(r'^[a-zA-Z][\w+.-]*:',target) or target.startswith(('#','/')):
            return match.group(0)
        path,sep,fragment = target.partition('#')
        absolute = (ROOT / old).parent.joinpath(path).resolve()
        if not absolute.is_relative_to(ROOT): return match.group(0)
        relative = absolute.relative_to(ROOT).as_posix()
        if relative not in mapping and old == new: return match.group(0)
        if relative not in mapping and not absolute.exists(): return match.group(0)
        updated = ROOT / mapping.get(relative,relative)
        ref = os.path.relpath(updated,(ROOT / new).parent).replace('\\','/')
        return match.group(1)+ref+(sep+fragment if sep else '')+')'
    return re.sub(r'(\]\()([^\s)]+)\)',replace,text)


def transform(text,old,new):
    text = links(text,old,new)
    text = module_pattern.sub(lambda match:modules[match.group(0)],text)
    text = path_pattern.sub(lambda match:mapping[match.group(0)],text)
    text = text.replace('from vlade.distillation import teacher_plan',
                        'from vlade.legacy.plan_cot.distillation import teacher_plan')
    text = text.replace('from .teacher_plan_supervision import',
                        'from vlade.legacy.plan_cot.training.teacher_plan_supervision import')
    if old.startswith('src/vlade/') and old in mapping:
        text = text.replace('Path(__file__).resolve().parents[3]', 'Path(__file__).resolve().parents[5]')
    if old.startswith('scripts/') and old in mapping:
        text = text.replace('Path(__file__).resolve().parents[2]',
            'next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())')
    return text


for old,new in mapping.items():
    source,destination = ROOT / old, ROOT / new
    assert source.resolve().is_relative_to(ROOT) and source.is_file(), old
    assert destination.resolve().is_relative_to((ROOT / BASE).resolve()) and not destination.exists(), new
    assert subprocess.run(['git','check-ignore','-q',old],cwd=ROOT).returncode == 1, old
    assert subprocess.run(['git','check-ignore','-q',new],cwd=ROOT).returncode == 1, new

changed = []
for old in eligible:
    source = ROOT / old
    if not source.is_file(): continue
    new = mapping.get(old,old)
    raw = source.read_bytes()
    if old.endswith('.ipynb'):
        value = json.loads(raw)
        modified = False
        for cell in value['cells']:
            before = ''.join(cell['source'])
            after = transform(before,old,new)
            if before != after:
                cell['source'] = after.splitlines(keepends=True)
                modified = True
        updated = (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode() if modified else raw
    else:
        updated = transform(raw.decode('utf-8'),old,new).encode('utf-8')
    destination = ROOT / new
    if old in mapping:
        destination.parent.mkdir(parents=True,exist_ok=True)
        source.rename(destination)
    if updated != raw:
        destination.write_bytes(updated)
        changed.append(new)

print(json.dumps({'moved':mapping,'updated_references':changed},ensure_ascii=False,indent=2))
