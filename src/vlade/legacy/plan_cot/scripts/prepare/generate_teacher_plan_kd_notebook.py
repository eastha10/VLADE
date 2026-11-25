"""Regenerate the Colab notebook for Teacher plan plus action distillation."""
from pathlib import Path
import sys

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from vlade.utils.qwen_scope_notebook import write_notebook

if __name__ == "__main__":
    write_notebook(ROOT / "src/vlade/legacy/plan_cot/experiments/colab_teacher_plan_kd.ipynb", teacher_plan=True)
