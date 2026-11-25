"""Thin entry point for annotation preparation and Plan-supervised Teacher FFT."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.research.teacher_plan_training import main

if __name__ == "__main__":
    main()
