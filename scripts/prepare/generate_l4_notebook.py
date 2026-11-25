"""Regenerate the configuration-only Colab L4 S0/S1/S2 client."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.utils.action_kd_notebook import ROOT, write_l4_notebook


def main() -> None:
    print(write_l4_notebook(ROOT / "experiments/colab_l4_training.ipynb").relative_to(ROOT))


if __name__ == "__main__":
    main()
