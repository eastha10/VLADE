"""Data-based budgets, pilot acceptance and first-step gradient diagnostics."""
from __future__ import annotations
import hashlib
import json
import math
import shutil
import re
import sys
from pathlib import Path
from typing import Any

from vlade.data.episode_data import EpisodeData, query_count, write_json
from vlade.data.colab_assets import sha256

CHECKPOINT_MANIFEST = "checkpoint_manifest.json"


def peak_process_rss_bytes() -> int | None:
    """Return peak resident memory for the current process when the OS exposes it."""
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * 1024 if sys.platform.startswith(("linux", "freebsd")) else value
    except (ImportError, OSError, ValueError):
        try:
            import psutil
            return int(psutil.Process().memory_info().rss)
        except (ImportError, OSError):
            return None


def directory_size_bytes(directory: Path) -> int:
    """Measure regular files below a completed artifact directory."""
    return sum(path.stat().st_size for path in directory.rglob("*")
               if path.is_file() and not path.is_symlink())


def seal_checkpoint(directory: Path, step: int, microsteps: int) -> dict:
    if type(step) is not int or step < 1 or type(microsteps) is not int or microsteps < step:
        raise ValueError("Invalid checkpoint optimizer boundary")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.relative_to(directory).as_posix() not in {CHECKPOINT_MANIFEST, "COMPLETE"}:
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Checkpoint must contain its own regular files")
            files[path.relative_to(directory).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if not files:
        raise ValueError("Cannot seal an empty checkpoint")
    result = {"schema_version": 1, "step": step, "microsteps": microsteps, "files": files}
    write_json(directory / CHECKPOINT_MANIFEST, result)
    return result


def verify_checkpoint(directory: Path, *, accumulation: int | None = None) -> dict:
    if not (directory / "COMPLETE").is_file():
        raise ValueError("Completed checkpoint required")
    manifest = json.loads((directory / CHECKPOINT_MANIFEST).read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or type(manifest.get("step")) is not int or manifest["step"] < 1 or
            type(manifest.get("microsteps")) is not int or manifest["microsteps"] < manifest["step"]):
        raise ValueError("Invalid checkpoint step seal")
    if accumulation is not None and manifest["microsteps"] != manifest["step"] * accumulation:
        raise ValueError("Checkpoint microstep/accumulation mismatch")
    label = re.fullmatch(r"step-(\d+)", directory.name)
    if label and int(label.group(1)) != manifest["step"]:
        raise ValueError("Checkpoint directory/optimizer step mismatch")
    files = manifest.get("files", {})
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.relative_to(directory).as_posix() not in {CHECKPOINT_MANIFEST, "COMPLETE"}}
    if not files or actual != set(files):
        raise ValueError("Checkpoint file set differs from the seal")
    for name, expected in files.items():
        path = directory / name
        if (path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or
                path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]):
            raise ValueError(f"Checkpoint file integrity mismatch: {name}")
    return manifest


def pilot_context(config: Path, checkpoint: Path, data: Path, kind: str, *, policy: dict | None = None) -> dict:
    if kind not in {"fft", "student"}:
        raise ValueError("Unknown pilot kind")
    verify_checkpoint(checkpoint)
    core = Path(__file__).parent
    paths = [core / name for name in ("run_control.py", "loop.py")]
    paths.append(core.parent / "data/episode_data.py")
    if kind == "fft":
        paths += [core / name for name in ("fft.py", "prepared_fft.py", "debug_records.py")]
    else:
        paths += [core / "student_task.py", core.parent / "models/students/qwen.py", core.parent / "models/contracts.py"]
    return {"kind": kind, "configuration_sha256": sha256(config), "data_manifest_sha256": sha256(data / "manifest.json"),
            "checkpoint_manifest_sha256": sha256(checkpoint / CHECKPOINT_MANIFEST),
            "code_sha256": code_hashes(paths), "policy": policy or {}}


def configure_tensorflow_cpu(seed: int = 42) -> None:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(seed)


def epoch_budget(data: EpisodeData, epochs: int, batch_size: int = 1, accumulation: int = 8) -> dict:
    if any(type(value) is not int or value < 1 for value in (epochs, batch_size, accumulation)):
        raise ValueError("Epochs, batch and accumulation must be positive integers")
    count = query_count(data)
    if count < 1:
        raise ValueError("No training queries")
    steps = math.ceil(count * epochs / (batch_size * accumulation))
    return {"target_epochs": epochs, "train_queries_per_epoch": count, "optimizer_steps": steps,
            "query_presentations": steps * batch_size * accumulation,
            "rounding_extra_queries": steps * batch_size * accumulation - count * epochs,
            "sampler": "seeded_episode_shuffle_causal_queries_v1"}


def overfit_report(rows: list[dict], *, max_ratio: float = .8, window: int = 5) -> dict:
    """Pilot trend threshold, not a scientific success-rate or convergence claim."""
    import statistics
    losses = [float(row["loss"]) for row in sorted(rows, key=lambda row: row["step"])]
    if not 0 < max_ratio < 1 or window < 1 or len(losses) < 2 * window or not all(math.isfinite(x) for x in losses):
        return {"status": "failed", "reason": "insufficient_or_nonfinite_loss_history"}
    initial, final = statistics.median(losses[:window]), statistics.median(losses[-window:])
    ratio = final / initial if initial > 0 else float("inf")
    return {"status": "passed" if ratio <= max_ratio else "failed", "initial_median": initial,
            "final_median": final, "ratio": ratio if math.isfinite(ratio) else None,
            "max_ratio": max_ratio, "meaning": "pilot_loss_decrease_only_not_rollout_success"}


def require_reports(paths: list[Path], contexts: list[dict] | None = None) -> None:
    if contexts is not None and len(contexts) != len(paths):
        raise ValueError("Pilot context count mismatch")
    for index, path in enumerate(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "passed":
            raise ValueError(f"Main training prerequisite failed: {path}")
        if contexts is not None and report.get("context") != contexts[index]:
            raise ValueError(f"Pilot data/code/config/checkpoint context mismatch: {path}")


def validate_teacher_report(report: dict, checkpoint: Path, split_sha256: str) -> None:
    if (report.get("report_type") != "teacher_validation_comparison" or
            report.get("selection_split") != "validation" or report.get("used_final_test") is not False):
        raise ValueError("T1 selection requires a validation comparison report; final test cannot select T1")
    protocol = report.get("evaluation_protocol_sha256", "")
    if len(protocol) != 64 or any(c not in "0123456789abcdef" for c in protocol):
        raise ValueError("Report needs the real evaluation protocol SHA256")
    if report.get("data_split_sha256") != split_sha256:
        raise ValueError("Report/data split mismatch")
    candidates = report.get("candidates", [])
    paths = []
    for row in candidates:
        name = row.get("checkpoint")
        if not isinstance(name, str) or not name.strip() or row.get("role") not in {"T0", "T1"}:
            raise ValueError("Report needs nonempty T0/T1 checkpoint identities")
        paths.append(Path(name).resolve())
    if len(paths) != len(set(paths)):
        raise ValueError("T0/T1 candidates must have distinct checkpoint identities")
    if not any(row.get("role") == "T0" for row in candidates):
        raise ValueError("Report needs the T0 reference as well as selected T1")
    selected = [row for row in candidates if row.get("role") == "T1" and
                Path(row.get("checkpoint", "")).resolve() == checkpoint.resolve()]
    if len(selected) != 1:
        raise ValueError("Report must identify exactly one selected T1 checkpoint")
    for row in candidates:
        for prefix in ("validation", "original_libero"):
            score, episodes = row.get(prefix + "_success_rate"), row.get(prefix + "_episodes")
            if (type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1 or
                    type(episodes) is not int or episodes < 1):
                raise ValueError("Fill the report with measured rollout scores/counts; null/placeholders are not evidence")


def code_hashes(paths: list[Path]) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def gradient_diagnostics(groups: dict[str, Any]) -> dict:
    import torch
    result = {}
    for name, module in groups.items():
        parameters = [p for p in module.parameters() if p.requires_grad]
        grads = [p.grad for p in parameters if p.grad is not None]
        nonzero = any(bool(torch.count_nonzero(grad)) for grad in grads)
        if not grads or not nonzero or not all(bool(torch.isfinite(grad).all()) for grad in grads):
            raise RuntimeError(f"Missing/nonfinite/zero gradient path: {name}")
        result[name] = {"trainable_tensors": len(parameters), "gradient_tensors": len(grads),
                        "finite_nonzero": True}
    return result


def checkpoint_space(output: Path, modules: list[Any], optimizer: Any) -> int:
    """Conservative next-save space check; never deletes user checkpoints."""
    import torch
    def size(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(size(item) for item in value)
        return 0
    required = sum(size(module.state_dict()) for module in modules) + size(optimizer.state_dict())
    if shutil.disk_usage(output).free < required * 1.2 + 1024**3:
        raise OSError(f"Insufficient checkpoint space: estimated {required / 1024**3:.1f} GiB plus margin")
    return required


def planned_checkpoint_space(root: Path, reference: Path, max_steps: int, save_steps: int,
                             after_step: int = 0) -> dict:
    if min(max_steps, save_steps) < 1 or after_step < 0:
        raise ValueError("Invalid checkpoint plan")
    points = set(range(save_steps, max_steps + 1, save_steps)) | {max_steps}
    remaining = sum(step > after_step for step in points)
    one = sum(path.stat().st_size for path in reference.rglob("*") if path.is_file())
    if one < 1:
        raise ValueError("A real completed pilot checkpoint is required for the storage estimate")
    estimate = math.ceil(one * remaining * 1.2)
    if shutil.disk_usage(root).free < estimate + 1024**3:
        raise OSError("Insufficient space for planned checkpoints; adjust the save interval or storage before main training")
    return {"remaining_checkpoints": remaining, "pilot_checkpoint_bytes": one, "estimated_bytes": estimate}
