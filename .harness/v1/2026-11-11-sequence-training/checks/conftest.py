import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
os.environ["VLADE_PROJECT_ROOT"] = str(ROOT)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".harness/v1/2026-11-11-ava-teacher-code/checks")]
