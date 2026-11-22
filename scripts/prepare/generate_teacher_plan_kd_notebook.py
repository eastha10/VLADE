"""Regenerate the Colab notebook for Teacher plan plus action distillation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlade.utils.qwen_scope_notebook import write_notebook

if __name__ == "__main__":
    write_notebook(ROOT / "experiments/colab_teacher_plan_kd.ipynb", teacher_plan=True)
