"""Prepare or publish explicitly reviewed Teacher Plan annotation outputs."""
from pathlib import Path
import sys

sys.path.insert(0, str(next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()) / "src"))
from vlade.legacy.plan_cot.data.teacher_plan_annotations import main

if __name__ == "__main__":
    main()
