"""Reload a real Student checkpoint and check a prepared-data forward pass."""
import argparse
from contextlib import closing, nullcontext
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from vlade.data.episode_data import EpisodeData, write_json
from vlade.distillation.teacher_plan import plan_token_budget
from vlade.models.students.qwen import QwenStudent
from vlade.storage.assets import resolve_input
from vlade.training.student_initialization import saved_processor
from vlade.training.student_task import batches, identity
from vlade.training.run_control import pilot_context, verify_checkpoint
from vlade.training.teacher_inference import InferenceActions

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
if not (args.checkpoint / "COMPLETE").is_file():
    raise ValueError("Completed Student checkpoint required")
data = EpisodeData(resolve_input(args.data))
saved_identity = json.loads((args.checkpoint / "identity.json").read_text(encoding="utf-8"))
sealed = verify_checkpoint(args.checkpoint, accumulation=saved_identity["settings"]["accumulation_steps"])
if identity(saved_identity["settings"], data) != saved_identity:
    raise ValueError("Checkpoint data/code/environment mismatch; rerun the pilot")
model = QwenStudent.load(args.checkpoint / "model").to("cuda:0").eval()
processor = saved_processor(model.student_config, args.checkpoint / "processor")
settings = saved_identity["settings"]
with (closing(InferenceActions(settings, data)) if plan_token_budget(settings) else nullcontext()) as teacher:
    if teacher is not None and teacher.signature != saved_identity["teacher_inference"]:
        raise ValueError("T1 identity changed before the Student reload check")
    inputs, proprio, _, _ = next(batches(
        data, processor, model.student_config, split="validation" if teacher else "train",
        limit=1 if teacher else 0, plan_teacher=teacher,
    ))
inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    expected = model(inputs, proprio.to("cuda:0")).actions
    actual = model(inputs, proprio.to("cuda:0")).actions
if expected.shape != (1, 8, 7) or not torch.isfinite(expected).all() or not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
    raise ValueError("Student reload/forward repeat check failed")
write_json(args.report, {"status": "passed", "scope": "real_query_reload_not_rollout_success",
                         "shape": list(actual.shape), "trainer_step": sealed["step"],
                         "context": pilot_context(args.config, args.checkpoint, data.root, "student"),
                         "repeat_max_error": float((actual - expected).abs().max())})
print(args.report)
