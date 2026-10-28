"""Build episode/group splits and exact train-only normalization for Colab."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.data.episode_data import main

if __name__ == "__main__":
    main()
