"""Reload a completed FFT checkpoint and run one real sequential episode query."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from vlade.data.episode_data import EpisodeData, write_json
from vlade.data.colab_assets import sha256
from vlade.training.fixed_cache import SequentialTeacher
from vlade.training.fft import _code_fingerprint
from vlade.training.run_control import (directory_size_bytes, peak_process_rss_bytes,
    pilot_context, verify_checkpoint)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--metrics", type=Path,
                    help="Training metrics JSONL from the process that produced CHECKPOINT")
parser.add_argument("--run-manifest", type=Path,
                    help="Run manifest from the process that produced CHECKPOINT")
args = parser.parse_args()
if (args.metrics is None) != (args.run_manifest is None):
    parser.error("--metrics and --run-manifest must be supplied together")
if not (args.checkpoint / "COMPLETE").is_file() or not (args.checkpoint / "trainer_state.pt").is_file():
    raise ValueError("A completed checkpoint with training state is required")
seal = verify_checkpoint(args.checkpoint)
data = EpisodeData(args.data)
metadata = json.loads((args.checkpoint / "vlade_fft.json").read_text(encoding="utf-8"))
if metadata.get("training_code_sha256") != _code_fingerprint():
    raise ValueError("Checkpoint was produced by different training code; rerun the pilot")
torch.cuda.reset_peak_memory_stats()
teacher = SequentialTeacher(args.checkpoint)
_, queries = next(data.episodes("train"))
teacher.reset()
action, hidden = teacher.predict(queries[0], data)
teacher.reset()
again, _ = teacher.predict(queries[0], data)
if action.shape != (8, 7) or not torch.isfinite(action).all() or not torch.allclose(action, again, atol=1e-5, rtol=1e-4):
    raise ValueError("Checkpoint reload/forward/reset repeat check failed")
report = {"status": "passed", "scope": "reload_real_query_not_rollout_success",
          "action_shape": list(action.shape), "hidden_shape": list(hidden.shape),
          "action_finite": True,
          "reset_repeat_max_error": float((action - again).abs().max()),
          "checkpoint_bytes": directory_size_bytes(args.checkpoint),
          "reload_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
          "reload_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
          "reload_peak_process_rss_bytes": peak_process_rss_bytes(),
          "training_code_sha256": metadata["training_code_sha256"],
          "prepared_data_identity": metadata["prepared_data_identity"],
          "context": pilot_context(args.config, args.checkpoint, args.data, "fft"),
          "gpu": torch.cuda.get_device_name(0)}
if args.metrics is not None and args.run_manifest is not None:
    rows = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    matching = [row for row in rows if row.get("step") == seal["step"]]
    run = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    required_metrics = ("loss", "grad_norm", "step_time_seconds", "peak_allocated_bytes",
                        "peak_reserved_bytes", "peak_process_rss_bytes")
    run_id = run.get("run_id")
    if (len(matching) != 1 or not isinstance(run_id, str) or
            args.metrics.name != f"metrics-{run_id}.jsonl" or
            args.run_manifest.name != f"run-{run_id}.json" or
            run.get("status") != "completed" or run.get("step") != seal["step"] or
            run.get("training_code_sha256") != metadata["training_code_sha256"] or
            run.get("prepared_data_identity") != metadata["prepared_data_identity"] or
            run.get("finite_checks") != {"predicted_action": True, "scalar_loss": True, "gradients": True} or
            not run.get("gradient_diagnostics") or
            any(key not in matching[0] or matching[0][key] is None or
                (isinstance(matching[0][key], float) and not math.isfinite(matching[0][key]))
                for key in required_metrics) or
            any(float(matching[0][key]) <= 0 for key in
                ("step_time_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
                 "peak_process_rss_bytes")) or float(matching[0]["grad_norm"]) < 0):
        raise ValueError("Training metrics/finite checks do not prove the checkpoint step")
    report.update(training_metrics=matching[0], gradient_diagnostics=run["gradient_diagnostics"],
                  training_finite_checks=run["finite_checks"], metrics_sha256=sha256(args.metrics),
                  run_manifest_sha256=sha256(args.run_manifest))
write_json(args.report, report)
print(args.report)
