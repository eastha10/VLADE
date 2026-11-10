"""Record a user's validation-based T1 selection; never choose the last step implicitly."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.data.colab_assets import sha256
from vlade.data.episode_data import EpisodeData, write_json
from vlade.training.fixed_cache import bundle_hash, validate_selection

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--validation-report", type=Path, required=True)
parser.add_argument("--reason", required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("Keep existing selections immutable; use another output name")
from vlade.storage.assets import resolve_input
checkpoint_reference = str(args.checkpoint)
if checkpoint_reference.startswith("hfasset:"):
    checkpoint_reference = "hfasset://" + checkpoint_reference[len("hfasset:"):].strip("/\\")
args.checkpoint = resolve_input(args.checkpoint)
data = EpisodeData(resolve_input(args.data))
metadata = json.loads((args.checkpoint / "vlade_fft.json").read_text(encoding="utf-8"))
selection = {"checkpoint": checkpoint_reference if checkpoint_reference.startswith("hfasset:") else str(args.checkpoint.resolve()), "checkpoint_sha256": bundle_hash(args.checkpoint),
             "teacher_step": metadata["step"], "quantized": False, "frozen": True,
             "normalization_sha256": data.manifest["normalization_sha256"],
             "split_sha256": data.manifest["split_sha256"], "selection_reason": args.reason,
             "validation_report": str(args.validation_report.resolve()),
             "validation_report_sha256": sha256(args.validation_report),
             "training_temporal_policy": metadata["ava_temporal_training"]}
validate_selection(selection, data)
args.output.parent.mkdir(parents=True, exist_ok=True)
write_json(args.output, selection)
print(args.output)
