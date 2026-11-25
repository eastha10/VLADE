"""Generate the configuration-only Teacher Plan training notebook."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.utils.teacher_plan_training_notebook import write_notebook

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(write_notebook(root / "experiments/colab_teacher_plan_training.ipynb"))
