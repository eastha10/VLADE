"""Generate the configuration-only Teacher Plan training notebook."""
from pathlib import Path
import sys

sys.path.insert(0, str(next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()) / "src"))
from vlade.legacy.plan_cot.utils.teacher_plan_training_notebook import write_notebook

if __name__ == "__main__":
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    print(write_notebook(root / "src/vlade/legacy/plan_cot/experiments/colab_teacher_plan_training.ipynb"))
