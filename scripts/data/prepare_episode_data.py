"""Stream Drive TFDS into local episode data with sealed periodic backups."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.data.episode_data import main

if __name__ == "__main__":
    main()
