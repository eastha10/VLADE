"""Generate the existing-A100-session S2 representation KD cells."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vlade.utils.representation_kd_notebook import write_notebook

if __name__ == "__main__":
    print(write_notebook(ROOT / "experiments/colab_s2_representation_kd_existing_session.ipynb"))
