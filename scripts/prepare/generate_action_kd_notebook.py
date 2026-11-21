"""Regenerate the requested dedicated HF T1 -> S0 Action KD notebook."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.utils.action_kd_notebook import main

if __name__ == "__main__":
    main()
