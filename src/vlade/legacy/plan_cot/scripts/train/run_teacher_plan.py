"""Thin entry point for annotation preparation and Plan-supervised Teacher FFT."""
from pathlib import Path
import sys

sys.path.insert(0, str(next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()) / "src"))
from vlade.legacy.plan_cot.research.teacher_plan_training import main

if __name__ == "__main__":
    main()
