"""Set the verified repository root for agent-only checks."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
if not (ROOT / "AGENTS.md").is_file():
    raise RuntimeError("Cannot identify the VLADE project root")
os.environ["VLADE_PROJECT_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
