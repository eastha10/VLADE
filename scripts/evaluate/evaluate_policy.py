"""Thin entry for project-owned LIBERO(+) evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from vlade.evaluation.cli import main
if __name__ == "__main__":
    raise SystemExit(main())
