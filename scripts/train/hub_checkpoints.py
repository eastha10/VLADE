"""Check private Hub authentication, retry uploads or download a resume snapshot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.training.hub import main

if __name__ == "__main__":
    main()
