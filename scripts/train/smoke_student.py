"""Thin entry point for native tiny Qwen Student validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.models.students.smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
