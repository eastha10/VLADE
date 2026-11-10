"""Colab-only orchestration around the existing FFT and suite-balanced loader.

No model or optimizer loop is replaced. Hooks live in the child process only;
shared sources, datasets and pretrained checkpoints remain read-only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import tarfile
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


class _SavedBudgetStop(Exception):
    """Internal control flow caught inside the loop hook, never by fft.train."""


def runtime_api_report(root: Path) -> dict:
    """Inspect the selected source tree instead of assuming optional budget APIs."""
    training = root / "src/vlade/training"
    fft_tree = ast.parse((training / "fft.py").read_text(encoding="utf-8"))
    loop_tree = ast.parse((training / "loop.py").read_text(encoding="utf-8"))
    train = next(node for node in fft_tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    imports = [node for node in ast.walk(train) if isinstance(node, ast.ImportFrom)]
    # A module-global run_loop alias cannot be intercepted by changing loop.run_loop later.
    if not any(node.module == "vlade.training.loop" and any(a.name == "run_loop" for a in node.names)
               for node in imports):
        raise ValueError("Unsupported FFT API: train must bind loop.run_loop at invocation")
    run_loop = next(node for node in loop_tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_loop")
    if "on_step" not in {arg.arg for arg in run_loop.args.kwonlyargs}:
        raise ValueError("Unsupported loop API: on_step callback missing")
    budget = training / "compute_budget.py"
    api = {"exists": budget.is_file(), "adapter_uses_native_budget": False}
    if budget.is_file():
        tree = ast.parse(budget.read_text(encoding="utf-8"))
        names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        api.update(names=sorted(names))
        stop = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "should_stop"), None)
        api["known_api"] = bool("BudgetExhausted" in names and stop and
            [arg.arg for arg in stop.args.args] == ["remaining_seconds", "save_reserve_seconds", "next_step_seconds"])
    elif any(node.module == "vlade.training.compute_budget" for node in imports):
        raise ValueError("Selected fft.train imports a missing compute_budget.py; source snapshot is incomplete")
    return {"compute_budget": api, "termination": "adapter_catches_saved_stop_inside_loop_hook",
            "timing": "adapter_monotonic_with_device_synchronization"}


def validate_suite_resume(cfg: Any, suite_map: Path, fingerprint: dict,
                          verify_checkpoint: Any) -> dict:
    """Reject incompatible cursor/state before the original trainer restores RNG."""
    import torch
    verify_checkpoint(cfg.resume, accumulation=cfg.grad_accumulation_steps)
    sampling = read(cfg.resume / "suite_sampling.json")
    payload = torch.load(cfg.resume / "trainer_state.pt", map_location="cpu", weights_only=True)
    state = payload["loop"]
    step = read(cfg.resume / "vlade_fft.json")["step"]
    if (sampling.get("sampling") != "suite_balanced" or sampling.get("validation_sampling") != "episode"
            or sampling.get("seed") != cfg.seed or sampling.get("suite_map_sha256") != sha256(suite_map)
            or sampling.get("fingerprint") != fingerprint
            or sampling.get("completed_optimizer_step") != step
            or sampling.get("query_cursor") != step * cfg.grad_accumulation_steps * cfg.batch_size
            or state.get("step") != step or state.get("microsteps") != step * cfg.grad_accumulation_steps
            or any(key not in payload for key in ("optimizer", "scheduler", "rng"))
            or any(key not in payload["rng"] for key in ("python", "numpy", "torch", "cuda"))):
        raise ValueError("Resume suite cursor, fingerprint, optimizer boundary or RNG state mismatch")
    return sampling


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    """Write and read back a small control file; never overwrite checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, default=str, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if read(path) != json.loads(payload):
        raise IOError(f"Control file readback failed: {path}")


def extract_archive(archive: Path, expected_sha256: str | None, destination: Path) -> Path:
    """Validate an explicitly selected TAR, then extract to a fresh local folder."""
    if destination.exists() or (expected_sha256 is not None and len(expected_sha256) != 64):
        raise ValueError("Use a new local destination and a valid expected SHA256, if available")
    if expected_sha256 is not None and sha256(archive) != expected_sha256:
        raise ValueError("TAR SHA256 mismatch")
    with tarfile.open(archive, "r:*") as stream:
        members = stream.getmembers()
        total = 0
        seen = set()
        for member in members:
            target = (destination / member.name).resolve()
            if (not target.is_relative_to(destination.resolve()) or
                    not (member.isfile() or member.isdir()) or target in seen):
                raise ValueError(f"Unsafe/duplicate TAR member: {member.name}")
            seen.add(target)
            total += member.size
        if shutil.disk_usage(destination.parent).free < total * 1.1:
            raise ValueError("Insufficient Colab local extraction space")
        destination.mkdir()
        stream.extractall(destination, members=members, filter="data")
    candidates = [p.parent for p in destination.rglob("manifest.json")
                  if (p.parent / "splits.json").is_file() and (p.parent / "normalization.json").is_file()]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one prepared dataset inside the TAR")
    return candidates[0]


def verify_data(root: Path, suite_map: Path) -> dict:
    """Check the canonical D identity and every declared payload, including sealed test files."""
    from vlade.data.episode_data import EpisodeData, query_count
    expected = {"manifest.json": "9a09c51dac170cb7452e0f3cd33f563e73429d6fe8f7a880daa8568ec2cf981b",
                "splits.json": "379376e5cfb19d0b9d0a5bdd548e01e4c658f09de3fdd1bff178b3aad7ed9900",
                "normalization.json": "e1dbf29ea83cc2e0fbcb90e68a7ad15660eb210baa5a333d20bd16c80d275af3"}
    if any(sha256(root / name) != digest for name, digest in expected.items()):
        raise ValueError("Extracted data differs from the recorded canonical D prepared manifest")
    data = EpisodeData(root)
    if (data.manifest.get("counts") != {"train": 11477, "validation": 1428, "test": 1442}
            or data.manifest.get("seed") != 42 or query_count(data) != 228908
            or data.manifest.get("split_policy") != "canonical_D_content_group_80_10_10_seed_42"
            or data.manifest.get("canonical_manifest_sha256") !=
            "487d2be1039ba6a4facc3ba44785ba5159245dfe53462a1d5a41ba8230446e80"
            or not data.stats.get("train_only") or data.stats.get("num_trajectories") != 11477):
        raise ValueError("Dataset does not match the confirmed D split")
    for name, dim in (("action", 7), ("proprio", 8)):
        if any(len(data.stats[name][key]) != dim for key in ("q01", "q99", "mask")):
            raise ValueError("BOUNDS_Q99 statistics dimension mismatch")
    for entry in data.entries:
        folder = root / "episodes" / entry["episode_id"]
        for name, expected in entry["files"].items():
            path = folder / name
            if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink() or sha256(path) != expected:
                raise ValueError(f"Dataset payload hash mismatch: {path}")
    # Reuse the exact suite mapping validation used by the existing sampler.
    from vlade.data.episode_data import _read_suite_mapping, _episode_suite
    by_episode, by_instruction = _read_suite_mapping(data, suite_map)
    counts = dict.fromkeys(("spatial", "goal", "object", "libero_10"), 0)
    for entry in data.entries:
        if entry["split"] == "train":
            queries = read(root / "episodes" / entry["episode_id"] / "queries.json")
            counts[_episode_suite(entry, queries, by_episode, by_instruction)] += entry["queries"]
    if not all(counts.values()):
        raise ValueError("An empty suite cannot be balanced")
    return {"counts": data.manifest["counts"], "train_queries": query_count(data),
            "suite_unique_queries": counts, "target_proportions": {k: .25 for k in counts},
            "sampling": "suite_balanced", "validation_sampling": "episode", "seed": 42,
            "replacement": "underrepresented_suites", "normalization": "train_only_BOUNDS_Q99",
            "action_chunk": 8, "action_dim": 7, "proprio_dim": 8,
            "files_sha256": {n: sha256(root / n) for n in ("manifest.json", "splits.json", "normalization.json")},
            "suite_map_sha256": sha256(suite_map), "payloads_verified": True,
            "test_usage": "integrity_only_never_training_or_selection"}


def plan_budget(*, profile: dict, current_cu: float | None, target_cu: float,
                cu_before: float | None, cu_after: float | None, cu_elapsed: float,
                session_seconds: float | None, explicit_seconds: float | None,
                free_bytes: int, checkpoint_bytes: int, remaining_steps: int,
                save_steps: int = 2000, margin: float = 1.35, resume_step: int = 0) -> dict:
    """Budget all remaining saves and keep checkpoint storage append-only."""
    def positive(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    if (not positive(target_cu) or not positive(cu_elapsed) or not positive(margin) or margin < 1
            or any(not positive(profile[k]) for k in
                   ("step_seconds", "checkpoint_seconds", "startup_seconds", "reload_seconds"))
            or type(resume_step) is not int or resume_step < 0
            or any(type(v) is not int or v < 1 for v in
                   (free_bytes, checkpoint_bytes, remaining_steps, save_steps))):
        raise ValueError("Invalid pilot, storage, or budget input")
    rate = None
    if cu_before is not None or cu_after is not None:
        if not positive(cu_before) or not positive(cu_after) or cu_before <= cu_after:
            raise ValueError("CU readings must show positive measured consumption; extend measurement if rounded")
        rate = (cu_before - cu_after) / cu_elapsed
    limits = []
    if rate is not None:
        if not positive(current_cu) or current_cu <= target_cu:
            raise ValueError("Enter current CU above the target reserve")
        limits.append((current_cu - target_cu) / (rate * margin))
    elif explicit_seconds is None:
        raise ValueError("No measured CU rate: enter an explicit time cap; no rate is assumed")
    for value in (session_seconds, explicit_seconds):
        if value is not None:
            if not positive(value):
                raise ValueError("Time caps must be positive and unexpired")
            limits.append(value)
    seconds = min(limits)
    validation_seconds = profile.get("validation_seconds")
    if not positive(validation_seconds):
        raise ValueError("Invalid validation measurement")
    reserve = max(300., margin * (profile["checkpoint_seconds"] + profile["reload_seconds"] + validation_seconds))
    startup = profile["startup_seconds"] * margin
    slot = math.ceil(checkpoint_bytes * margin)
    # One extra slot covers an incomplete save without deleting earlier checkpoints.
    slots = free_bytes // slot - 1
    def save_count(steps: int) -> int:
        final_step = resume_step + steps
        return final_step // save_steps - resume_step // save_steps + int(final_step % save_steps != 0)
    def cost(steps: int) -> float:
        saves = save_count(steps)
        return (startup + steps * profile["step_seconds"] * margin
                + saves * (profile["checkpoint_seconds"] + validation_seconds) * margin + reserve)
    lo, hi = 0, remaining_steps
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if save_count(middle) <= slots and cost(middle) <= seconds:
            lo = middle
        else:
            hi = middle - 1
    if lo < 1:
        raise ValueError("Insufficient measured time/CU or Drive quota for a resumable FFT step")
    return {"steps_this_session": lo, "remaining_training_steps": remaining_steps,
            "budget_limited": lo < remaining_steps, "wall_seconds": seconds,
            "save_steps": save_steps, "save_reserve_seconds": reserve,
            "measured_cu_per_second": rate, "estimated_cu_per_step": rate * profile["step_seconds"] if rate else None,
            "current_cu": current_cu, "target_remaining_cu": target_cu,
            "cu_limit_is_estimated": True, "cu_target_supported": rate is not None,
            "safety_factor": margin, "estimated_main_seconds": cost(lo),
            "estimated_full_training_seconds": cost(remaining_steps),
            "estimated_checkpoint_bytes": checkpoint_bytes, "free_drive_bytes": free_bytes,
            "planned_checkpoint_count": save_count(lo), "profile": profile,
            "resume_step": resume_step}


def install_hooks(suite_map: Path) -> dict:
    """Bind the existing loader and strengthen fingerprints in this process only."""
    import vlade.training.fft as fft
    import vlade.training.prepared_fft as prepared
    import vlade.training.prepared_fft_suite_balanced as balanced
    from types import SimpleNamespace
    original = prepared.make_loader
    fingerprint = fft._code_fingerprint
    def make_loader(cfg: Any, processor: Any, *, train: bool = True) -> Any:
        values = asdict(cfg)
        values["suite_map_path"] = suite_map
        return balanced.make_loader(SimpleNamespace(**values), processor, train=train)
    def code_fingerprint() -> dict:
        result = fingerprint()
        result["suite_map.json"] = sha256(suite_map)
        # Model provenance is pinned by Git; do not scan a source checkout or emit per-file hashes.
        result["ava_origin_commit"] = fft.ORIGIN_REVISIONS["ava_vla"]
        result["source_commit"] = fft._git_revision(fft.REPO_ROOT)

        return result
    prepared.make_loader = make_loader
    fft._code_fingerprint = code_fingerprint
    return {"previous_loader": original, "fingerprint": code_fingerprint()}


def execute(settings: dict) -> dict:
    """Run one explicitly selected phase, preserving the existing FFT loop."""
    import torch
    import vlade.training.fft as fft
    import vlade.training.loop as loop
    import vlade.training.run_control as control
    runtime_api = runtime_api_report(fft.REPO_ROOT)
    from vlade.storage.assets import resolve_input
    suite_map = resolve_input(settings["suite_map"]) / "suite_map.json"
    install_hooks(suite_map)
    cfg = fft.FFTConfig.from_json(Path(settings["config"]), "ava_vla")
    cfg = replace(cfg, resume=Path(settings["resume"]) if settings.get("resume") else None)
    phase = settings["phase"]
    if phase not in {"pilot4", "main"} or not settings.get("authorized"):
        raise ValueError("Explicit phase execution flag required")
    if cfg.seed != 42 or cfg.image_aug or not cfg.save_optimizer_state:
        raise ValueError("Require seed=42, image_aug=false and optimizer state saving")
    if phase == "pilot4" and (cfg.max_steps != 4 or cfg.resume):
        raise ValueError("Pilot must start independently from T0 and run four optimizer steps")
    run_root = Path(settings["run_root"])
    started = time.time()
    started_monotonic = time.monotonic()
    deadline = float(settings["deadline_unix"])
    reserve = float(settings["reserve_seconds"])
    stop_step = int(settings["stop_step"])
    if not math.isfinite(deadline) or not math.isfinite(reserve) or reserve <= 0 or deadline <= started + reserve:
        raise ValueError("Expired/invalid execution budget")
    deadline_monotonic = started_monotonic + deadline - started
    # Only the adapter owns budget termination. Older FFT catches every Exception as failure;
    # newer FFT has a different optional budget module. Never send our stop into either path.
    budget_env = {key: os.environ.get(key) for key in
                  ("VLADE_TRAIN_DEADLINE_UNIX", "VLADE_DEADLINE_UNIX", "VLADE_SAVE_RESERVE_SECONDS", "VLADE_HUB_RUN_ID")}
    resume_step = 0
    if cfg.resume:
        if not cfg.resume.resolve().is_relative_to((run_root / "checkpoints").resolve()):
            raise ValueError("Resume must explicitly name this run's main checkpoint")
        resume_step = fft.validate_resume(cfg)["step"]
        validate_suite_resume(cfg, suite_map, fft._code_fingerprint(), control.verify_checkpoint)
    if not resume_step < stop_step <= cfg.max_steps:
        raise ValueError("Invalid session target optimizer step")
    context: dict = {}
    saves: list[dict] = []
    times: list[float] = []
    evaluations: list[float] = []
    budget_stopped = False
    old_model, old_loader = fft._make_model_and_components, fft._make_loader
    old_save, old_loop, old_seal = fft.save_checkpoint, loop.run_loop, control.seal_checkpoint
    old_evaluate = loop.evaluate
    def synchronize() -> None:
        if str(cfg.device).startswith("cuda"):
            torch.cuda.synchronize(cfg.device)
    def evaluate_hook(*args: Any, **kwargs: Any) -> Any:
        synchronize()
        before = time.monotonic()
        try:
            return old_evaluate(*args, **kwargs)
        finally:
            synchronize()
            elapsed = time.monotonic() - before
            context.setdefault("validation_timings", []).append(elapsed)
            if context.get("step_metrics") is not None:
                context["step_metrics"]["validation_seconds"] = elapsed
    def model_hook(*args: Any, **kwargs: Any) -> Any:
        value = old_model(*args, **kwargs)
        context["model"] = value
        return value
    def loader_hook(*args: Any, **kwargs: Any) -> Any:
        value = old_loader(*args, **kwargs)
        if kwargs.get("train", True):
            context["dataset"] = value[1]
        return value
    def seal_hook(directory: Path, step: int, microsteps: int) -> dict:
        for name in ("config.json", "trainer_state.pt", "vlade_fft.json"):
            if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
                raise ValueError(f"Cannot publish an incomplete training checkpoint: {name}")
        write(directory / "suite_sampling.json", {
            "sampling": "suite_balanced", "validation_sampling": "episode", "seed": 42,
            "target_proportions": {k: .25 for k in ("spatial", "goal", "object", "libero_10")},
            "query_order": "independent_prefer_distinct_trajectory_groups",
            "queries_per_update": cfg.grad_accumulation_steps * cfg.batch_size,
            "suite_map_sha256": sha256(suite_map), "completed_optimizer_step": step,
            "query_cursor": step * cfg.grad_accumulation_steps * cfg.batch_size,
            "cursor_method": "deterministic_replay_skip_num_workers_0_no_image_augmentation",
            "bitwise_gpu_reproduction_guaranteed": False, "fingerprint": fft._code_fingerprint()})
        return old_seal(directory, step, microsteps)
    def save_hook(*args: Any, **kwargs: Any) -> Path:
        if kwargs.get("hub") is not None:
            context["hub"] = kwargs["hub"]
        if settings.get("free_bytes_at_start") is not None:
            used = sum(p.stat().st_size for p in run_root.rglob("*") if p.is_file())
            if settings["free_bytes_at_start"] - used < settings["checkpoint_bytes"] * 1.35:
                raise RuntimeError("Insufficient user-confirmed Drive quota for another checkpoint")
        before = time.monotonic()
        path = old_save(*args, **kwargs)
        control.verify_checkpoint(path, accumulation=cfg.grad_accumulation_steps)
        elapsed = time.monotonic() - before
        saves.append({"step": read(path / "vlade_fft.json")["step"], "path": str(path),
                      "save_verify_seconds": elapsed,
                      "bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file())})
        return path
    def loop_hook(**kwargs: Any) -> Any:
        nonlocal budget_stopped
        state = kwargs["state"]
        original_step = kwargs["on_step"]
        original_batches = kwargs["batches"]
        max_step_seconds = max(float(settings.get("step_seconds", 0)),
                               float(settings.get("first_step_seconds", 0)))
        def save_boundary() -> None:
            if state.microsteps != state.step * cfg.grad_accumulation_steps:
                raise ValueError("Cannot checkpoint a partial accumulation window")
            if state.step < 1:
                raise RuntimeError("Budget expired before the first optimizer step; no checkpoint saved")
            path = cfg.output_dir / f"step-{state.step:07d}"
            if path.exists():
                control.verify_checkpoint(path, accumulation=cfg.grad_accumulation_steps)
            else:
                model, processor, head, proprio, generator = context["model"]
                from vlade.training.hub import HubCheckpoints
                hub = context.get("hub")
                if hub is None:
                    hub = HubCheckpoints(cfg.output_dir, resume=cfg.resume, metric="val_l1")
                    context["hub"] = hub
                save_hook(cfg, state.step, model, processor, head, proprio, generator,
                          kwargs["optimizer"], context["dataset"], scheduler=kwargs["scheduler"], loop_state=state,
                          hub=hub, metrics=context.get("last_metrics", {}))
                hub.publish(path, final=state.step == cfg.max_steps)
        def guarded_batches() -> Any:
            # Check before fetching the first microbatch of each accumulation window.
            iterator = iter(original_batches)
            micro = 0
            while True:
                dynamic_save = max([reserve] + [1.35 * s["save_verify_seconds"] + 300 for s in saves])
                if micro % cfg.grad_accumulation_steps == 0:
                    if (state.step >= stop_step or
                            time.monotonic() + dynamic_save + max_step_seconds * 1.35 >= deadline_monotonic):
                        save_boundary()
                        raise _SavedBudgetStop("Saved before the next accumulation window")
                    synchronize()
                    context["step_started"] = time.monotonic()
                    context.setdefault("startup_seconds", context["step_started"] - started_monotonic)
                try:
                    batch = next(iterator)
                except StopIteration:
                    return  # Preserve the loop's truncated-accumulation failure policy (PEP 479).
                yield batch
                micro += 1
        def on_step(state: Any, metrics: dict) -> None:
            nonlocal max_step_seconds
            synchronize()
            seconds = time.monotonic() - context["step_started"]
            metrics["optimizer_step_seconds"] = seconds
            context["step_metrics"] = metrics
            validation_start = len(context.get("validation_timings", []))
            try:
                original_step(state, metrics)
                context["last_metrics"] = dict(metrics)
            finally:
                context.pop("step_metrics", None)
            # A newer FFT may overwrite its own timing key; preserve our independent measurement.
            metrics["optimizer_step_seconds"] = seconds
            times.append(seconds)
            measured_validation = context.get("validation_timings", [])[validation_start:]
            if measured_validation:
                metrics["validation_seconds"] = sum(measured_validation)
                evaluations.append(metrics["validation_seconds"])
            max_step_seconds = max(max_step_seconds, seconds)
            with (run_root / "logs/metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"phase": phase, "step": state.step, **metrics}) + "\n")
        kwargs.update(batches=guarded_batches(), on_step=on_step)
        try:
            return old_loop(**kwargs)
        except _SavedBudgetStop:
            # save_boundary already verified the seal. Return normally to legacy fft.train;
            # the adapter reports budget_exhausted after it independently verifies final state.
            budget_stopped = True
            return state
    fft._make_model_and_components, fft._make_loader = model_hook, loader_hook
    fft.save_checkpoint, loop.run_loop, control.seal_checkpoint = save_hook, loop_hook, seal_hook
    loop.evaluate = evaluate_hook
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    try:
        if os.environ.get("VLADE_HUB_REPO"):
            # Both phases end in /checkpoints; use distinct Hub groups.
            group = os.environ.get("VLADE_HUB_RUN_ID", "")
            if not group:
                raise ValueError("Set VLADE_HUB_RUN_ID for Suite Hub uploads")
            os.environ["VLADE_HUB_RUN_ID"] = group + "-" + phase
        os.environ.update(VLADE_TRAIN_DEADLINE_UNIX="0", VLADE_DEADLINE_UNIX="0",
                          VLADE_SAVE_RESERVE_SECONDS=str(reserve))
        fft.train(cfg)
        runs = sorted(cfg.output_dir.glob("run-*.json"), key=lambda p: p.stat().st_mtime_ns)
        run = read(runs[-1])
        if run["status"] not in {"completed", "budget_exhausted"}:
            raise RuntimeError(f"FFT did not finish safely: {run['status']}")
        run.update(sampler="suite_balanced", validation_sampling="episode",
                   suite_map_sha256=sha256(suite_map))
        if budget_stopped:
            run.update(status="budget_exhausted", budget_stop_verified=True)
        write(runs[-1], run)
        path = cfg.output_dir / f"step-{run['step']:07d}"
        control.verify_checkpoint(path, accumulation=cfg.grad_accumulation_steps)
        payload = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=True)
        if payload["loop"]["step"] != run["step"] or any(k not in payload for k in ("optimizer", "scheduler", "rng")):
            raise ValueError("Final training state is incomplete")
        result = {"status": "completed" if run["step"] == cfg.max_steps else "budget_exhausted",
                  "phase": phase, "step": run["step"], "checkpoint": str(path),
                  "elapsed_seconds": time.monotonic() - started_monotonic,
                  "startup_seconds": context.get("startup_seconds", run.get("startup_seconds", 0)),
                  "step_seconds": max(times[1:] or times, default=None),
                  "first_step_seconds": times[0] if times else None, "saves": saves,
                  "validation_seconds": max(evaluations, default=0),
                  "fingerprint": fft._code_fingerprint(), "checkpoint_verified": True,
                  "runtime_api": runtime_api,
                  "gpu": run["hardware"], "sampler": "suite_balanced",
                  "validation_sampling": "episode", "training_complete": run["step"] == cfg.max_steps}
        plan = settings.get("budget", {})
        rate = plan.get("measured_cu_per_second")
        cu_start = plan.get("current_cu")
        result.update(cu_usage_is_estimated=True,
                      estimated_cu_used=rate * result["elapsed_seconds"] if rate else None,
                      estimated_remaining_cu=cu_start - rate * result["elapsed_seconds"] if rate and cu_start else None,
                      budget=plan)
        report = run_root / ("pilot4/pilot_report.json" if phase == "pilot4" else "reports/run_summary.json")
        write(report, result)
        return result
    except BaseException:
        with (run_root / "logs/errors.log").open("a", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        write(run_root / "reports/failure.json", {"status": "failed", "phase": phase,
              "elapsed_seconds": time.monotonic() - started_monotonic, "auto_unassign_allowed": False})
        raise
    finally:
        fft._make_model_and_components, fft._make_loader = old_model, old_loader
        fft.save_checkpoint, loop.run_loop, control.seal_checkpoint = old_save, old_loop, old_seal
        loop.evaluate = old_evaluate
        for key, value in budget_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path)
    args = parser.parse_args()
    execute(read(args.settings))


if __name__ == "__main__":
    main()
