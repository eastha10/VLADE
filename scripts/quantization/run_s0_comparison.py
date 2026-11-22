"""Plan or execute the independent S0 quantization comparison."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.evaluation.quantization_comparison import main

if __name__ == "__main__":
    main()
