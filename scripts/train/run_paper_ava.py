"""Run the separate paper-informed K=4 LoRA reference experiment."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from vlade.training.paper_ava import main

if __name__ == "__main__":
    main()
