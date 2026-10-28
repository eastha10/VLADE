"""Generate/reload/replay-check a selected T1 cache in an isolated AVA process."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.training.fixed_cache import main
if __name__ == "__main__":
    main()
