"""Generate the four Colab notebooks and their ordinary-source manifest."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.utils.colab_notebooks import main
if __name__ == "__main__":
    main()
