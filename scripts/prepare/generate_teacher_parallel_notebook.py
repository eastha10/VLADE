"""Regenerate the Teacher-only two-GPU Kaggle/Colab notebook."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.utils.teacher_parallel_notebook import main

if __name__ == "__main__":
    main()
