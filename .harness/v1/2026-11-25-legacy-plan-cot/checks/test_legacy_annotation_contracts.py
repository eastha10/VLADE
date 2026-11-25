"""Run preserved annotation/workflow regressions with relocated imports and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / 'AGENTS.md').is_file()
original = ROOT / '.harness/v1/2026-11-25-teacher-plan-training/checks/test_teacher_plan_training.py'
source = original.read_text(encoding='utf-8')
# Only module/file locations change. Original tests and their semantics remain in
# the old record untouched; do not register compatibility aliases in production.
source = source.replace('from vlade.data import teacher_plan_annotations',
                        'from vlade.legacy.plan_cot.data import teacher_plan_annotations')
source = source.replace('from vlade.research import teacher_plan_training',
                        'from vlade.legacy.plan_cot.research import teacher_plan_training')
for old,new in {
    'vlade.training.teacher_plan_supervision':'vlade.legacy.plan_cot.training.teacher_plan_supervision',
    'vlade.utils.teacher_plan_training_notebook':'vlade.legacy.plan_cot.utils.teacher_plan_training_notebook',
    'configs/experiments/teacher_plan_training.json':'src/vlade/legacy/plan_cot/configs/experiments/teacher_plan_training.json',
    'src/vlade/research/teacher_plan_training.py':'src/vlade/legacy/plan_cot/research/teacher_plan_training.py',
    'scripts/train/run_teacher_plan.py':'src/vlade/legacy/plan_cot/scripts/train/run_teacher_plan.py',
    'experiments/colab_teacher_plan_training.ipynb':'src/vlade/legacy/plan_cot/experiments/colab_teacher_plan_training.ipynb',
}.items():
    source = source.replace(old,new)
exec(compile(source,str(original),'exec'),globals())
