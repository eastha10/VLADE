"""Backward-compatible entry point for the S0 fake-quantization worker."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.evaluation.s0_quantization import main, observations, parse_args

if __name__ == "__main__":
    main()
