"""Export the complete original frame stream with the frozen D/seed42 split."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from vlade.data.fullframes import main
if __name__=='__main__':main()
