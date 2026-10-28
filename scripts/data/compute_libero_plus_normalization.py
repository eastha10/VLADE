"""Compute canonical D/seed-42 LIBERO+ train-only normalization."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vlade.data.canonical_normalization import main


if __name__ == "__main__":
    main()
