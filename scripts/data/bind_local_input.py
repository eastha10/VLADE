"""Register an existing input after complete immutable-HF verification."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.storage.local_inputs import main

if __name__ == "__main__":
    main()
