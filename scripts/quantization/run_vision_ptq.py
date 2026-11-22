"""Run the vision-only PTQ configuration from a Git checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.quantization.vision.runner import main

if __name__ == "__main__":
    main()
