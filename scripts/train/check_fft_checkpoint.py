"""Reload a completed FFT checkpoint and run one real sequential episode query."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from vlade.data.episode_data import EpisodeData, write_json
from vlade.training.fixed_cache import SequentialTeacher
from vlade.training.fft import _code_fingerprint
from vlade.training.run_control import pilot_context

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
if not (args.checkpoint / "COMPLETE").is_file() or not (args.checkpoint / "trainer_state.pt").is_file():
    raise ValueError("A completed checkpoint with training state is required")
data = EpisodeData(args.data)
metadata = json.loads((args.checkpoint / "vlade_fft.json").read_text(encoding="utf-8"))
if metadata.get("training_code_sha256") != _code_fingerprint():
    raise ValueError("Checkpoint was produced by different training code; rerun the pilot")
teacher = SequentialTeacher(args.checkpoint)
_, queries = next(data.episodes("train"))
teacher.reset()
action, hidden = teacher.predict(queries[0], data)
teacher.reset()
again, _ = teacher.predict(queries[0], data)
if action.shape != (8, 7) or not torch.isfinite(action).all() or not torch.allclose(action, again, atol=1e-5, rtol=1e-4):
    raise ValueError("Checkpoint reload/forward/reset repeat check failed")
write_json(args.report, {"status": "passed", "scope": "reload_real_query_not_rollout_success",
                         "action_shape": list(action.shape), "hidden_shape": list(hidden.shape),
                         "reset_repeat_max_error": float((action - again).abs().max()),
                         "training_code_sha256": metadata["training_code_sha256"],
                         "prepared_data_identity": metadata["prepared_data_identity"],
                         "context": pilot_context(args.config, args.checkpoint, args.data, "fft"),
                         "gpu": torch.cuda.get_device_name(0)})
print(args.report)
