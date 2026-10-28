"""Reload a real Student checkpoint and check a prepared-data forward pass."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from vlade.data.episode_data import EpisodeData, write_json
from vlade.models.students.qwen import QwenStudent, load_processor
from vlade.training.student_task import batches, identity
from vlade.training.run_control import pilot_context, verify_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
if not (args.checkpoint / "COMPLETE").is_file():
    raise ValueError("Completed Student checkpoint required")
data = EpisodeData(args.data)
saved_identity = json.loads((args.checkpoint / "identity.json").read_text(encoding="utf-8"))
sealed = verify_checkpoint(args.checkpoint, accumulation=saved_identity["settings"]["accumulation_steps"])
if identity(saved_identity["settings"], data) != saved_identity:
    raise ValueError("Checkpoint data/code/environment mismatch; rerun the pilot")
model = QwenStudent.load(args.checkpoint / "model").to("cuda:0").eval()
processor = load_processor(model.student_config, directory=args.checkpoint / "processor")
inputs, proprio, _, _ = next(batches(data, processor, model.student_config, split="train"))
inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    expected = model(inputs, proprio.to("cuda:0")).actions
    actual = model(inputs, proprio.to("cuda:0")).actions
if expected.shape != (1, 8, 7) or not torch.isfinite(expected).all() or not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
    raise ValueError("Student reload/forward repeat check failed")
write_json(args.report, {"status": "passed", "scope": "real_query_reload_not_rollout_success",
                         "shape": list(actual.shape), "trainer_step": sealed["step"],
                         "context": pilot_context(args.config, args.checkpoint, args.data, "student"),
                         "repeat_max_error": float((actual - expected).abs().max())})
print(args.report)
