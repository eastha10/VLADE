"""Migrate approved existing data/weights from a mounted project to private HF."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from vlade.storage.migrate import main
if __name__=="__main__":main()
