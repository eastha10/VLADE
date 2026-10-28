"""Download/verify pinned LIBERO+ data or AVA-VLA weights to a user root."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.data.colab_assets import main

if __name__ == "__main__":
    main()
