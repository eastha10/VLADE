"""Config-driven native QwenVLA Action KD and CoT+Action KD entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.research.distillation import main

if __name__ == "__main__":
    main()
