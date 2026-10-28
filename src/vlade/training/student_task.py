"""Task-only Qwen BC using the same episode queries and train statistics as T1."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import random
import time
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from vlade.data.colab_assets import sha256
from vlade.data.episode_data import EpisodeData, query_stream, query_count, write_json
from vlade.models.contracts import masked_action_l1
from vlade.models.students.qwen import HistoryFrame, QwenStudent, StudentConfig, load_processor, prepare_inputs
from vlade.training.loop import LoopState, make_scheduler, restore_training_state, run_loop, training_state
from vlade.training.run_control import checkpoint_space, gradient_diagnostics, seal_checkpoint, verify_checkpoint
from vlade.utils.filesystem import publish_directory


def samples(data: EpisodeData, split: str, *, repeat: bool = False,
            episode_limit: int = 0, seed: int = 42, skip: int = 0, limit: int = 0) -> Iterator[tuple[dict, list[dict]]]:
    """Deterministic causal windows; this sampler is also the future KD boundary."""
    yield from query_stream(data, split, repeat=repeat, episode_limit=episode_limit,
                            seed=seed, skip=skip, limit=limit)


def batches(data: EpisodeData, processor, config: StudentConfig, *, split: str,
            skip: int = 0, episode_limit: int = 0, seed: int = 42, limit: int = 0):
    for sample, history in samples(data, split, repeat=split == "train", episode_limit=episode_limit,
                                   seed=seed, skip=skip, limit=limit):
        frames = [HistoryFrame(q["timestep"], data.image(q, "primary"), data.image(q, "wrist")) for q in history]
        inputs = prepare_inputs(processor, config, instruction=sample["instruction"], frames=frames,
                                current_timestep=sample["timestep"])
        yield inputs, torch.from_numpy(sample["proprio"])[None], torch.from_numpy(sample["gt_action"])[None], torch.from_numpy(sample["action_valid_mask"])[None]


def identity(settings: dict, data: EpisodeData) -> dict:
    paths = [Path(__file__), Path(__file__).with_name("loop.py"),
             Path(__file__).with_name("run_control.py"),
             Path(__file__).parents[1] / "data/episode_data.py",
             Path(__file__).parents[1] / "models/contracts.py",
             Path(__file__).parents[1] / "models/students/qwen.py"]
    return {"settings": settings, "data_manifest_sha256": sha256(data.root / "manifest.json"),
            "code_sha256": {path.name: sha256(path) for path in paths},
            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "initial_download_manifest_sha256": sha256(Path(settings["initial_checkpoint"]) / "download_manifest.json")}


def run(settings: dict, *, resume: Path | None = None, preflight: bool = False) -> None:
    if settings.get("batch_size") != 1 or settings.get("method") != "task_only":
        raise ValueError("Task-only path currently supports batch_size=1; use gradient accumulation")
    if min(settings["max_steps"], settings["save_steps"], settings["accumulation_steps"]) < 1:
        raise ValueError("Training budgets must be positive")
    if settings.get("eval_steps", 100) < 1 or settings.get("validation_queries", 32) < 1:
        raise ValueError("Periodic validation and a positive validation query budget are required")
    data = EpisodeData(Path(settings["data"]))
    config = StudentConfig(**settings["student"])
    config.validate()
    if config.history_queries != 2 or config.query_interval_steps != 8:
        raise ValueError("This shared sampler requires two queries at interval eight")
    checkpoint = Path(settings["initial_checkpoint"])
    if not (checkpoint / "model.safetensors").is_file() and not (checkpoint / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("Download the pinned Qwen model before Student training")
    downloaded = json.loads((checkpoint / "download_manifest.json").read_text(encoding="utf-8"))
    if (downloaded["repository"], downloaded["revision"]) != (config.model_id, config.revision):
        raise ValueError("Student initial checkpoint revision mismatch")
    for item in downloaded["files"]:
        if sha256(checkpoint / item["path"]) != item["sha256"]:
            raise ValueError(f"Student initial checkpoint changed: {item['path']}")
    experiment = identity(settings, data)
    if resume:
        if not (resume / "COMPLETE").is_file():
            raise ValueError("Incomplete Student checkpoint")
        if json.loads((resume / "identity.json").read_text(encoding="utf-8")) != experiment:
            raise ValueError("Student resume configuration/data/code/environment mismatch")
        sealed = verify_checkpoint(resume, accumulation=settings["accumulation_steps"])
    if preflight:
        print(json.dumps({"status": "preflight_ok", "counts": data.manifest["counts"],
                          "model": config.model_id, "gpu_forward_tested": False}))
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 CUDA GPU is required")
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.glob("step-*")):
        raise FileExistsError("Choose a fresh run name or explicitly resume a completed checkpoint")
    model = (QwenStudent.load(resume / "model") if resume else
             QwenStudent.from_pretrained(config, directory=checkpoint, dtype=torch.float32)).to("cuda:0")
    model.backbone.gradient_checkpointing_enable()
    processor = load_processor(config, directory=checkpoint)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Student BC expects the full backbone and head to be trainable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = make_scheduler(optimizer, settings["max_steps"], schedule="constant")
    state = LoopState()
    if resume:
        payload = torch.load(resume / "trainer_state.pt", map_location="cpu", weights_only=True)
        state = restore_training_state(payload, optimizer, scheduler)
        if (state.step, state.microsteps) != (sealed["step"], sealed["microsteps"]):
            raise ValueError("Student model/optimizer/query cursor step mismatch")
    run_id = uuid.uuid4().hex[:12]
    manifest = {"status": "running", "identity": experiment, "seed": seed,
                "initialization": "pinned_pretrained_backbone_and_seeded_continuous_head",
                "mode": "single_episode_overfit_pilot" if settings.get("episode_limit", 0) else "task_only",
                "gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "dtype": "float32_parameters_bfloat16_autocast", "rollout_success_rate": None,
                "kd_enabled": False, "resume_sample_order": "deterministic_query_cursor",
                "sampler": "seeded_episode_shuffle_causal_queries_v1",
                "train_queries_per_epoch": query_count(data, episode_limit=settings.get("episode_limit", 0)),
                "validation_policy": "uniform_query_positions_across_validation_split",
                "normalization_sha256": data.manifest["normalization_sha256"]}
    manifest_path = output / f"run-{run_id}.json"
    write_json(manifest_path, manifest)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def loss_fn(batch, step):
        inputs, proprio, target, valid = batch
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(inputs, proprio.to("cuda:0"))
        return masked_action_l1(prediction, target.to("cuda:0"), valid.to("cuda:0"))

    def validation_loss(step):
        from vlade.training.loop import evaluate
        budget = settings.get("validation_queries", 32)
        return evaluate([model], batches(data, processor, config, split="validation", seed=seed, limit=budget),
                        loss_fn, step, budget)

    def before_step(current):
        if "gradient_diagnostics" not in manifest:
            manifest["gradient_diagnostics"] = gradient_diagnostics({
                "vision": model.backbone.visual, "language": model.backbone.language_model,
                "proprio": model.head.proprio, "action": model.head.action})

    try:
        with (output / f"metrics-{run_id}.jsonl").open("x", encoding="utf-8") as log:
            def on_step(current, metrics):
                metrics.update(step=current.step, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               elapsed_seconds=time.monotonic() - started)
                if current.step % settings.get("eval_steps", 100) == 0 or current.step == settings["max_steps"]:
                    metrics["validation_l1"] = validation_loss(current.step)
                    manifest["validation_l1"] = metrics["validation_l1"]
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                print(json.dumps(metrics), flush=True)
                if current.step % settings["save_steps"] == 0 or current.step == settings["max_steps"]:
                    manifest["checkpoint_estimated_bytes"] = checkpoint_space(output, [model], optimizer)
                    stage = output / f".step-{current.step:07d}-{uuid.uuid4().hex}.incomplete"
                    stage.mkdir()
                    model.save(stage / "model")
                    processor.save_pretrained(stage / "processor")
                    torch.save(training_state(current, optimizer, scheduler), stage / "trainer_state.pt")
                    write_json(stage / "identity.json", experiment)
                    seal_checkpoint(stage, current.step, current.microsteps)
                    (stage / "COMPLETE").write_text("ok\n", encoding="utf-8")
                    publish_directory(stage, output / f"step-{current.step:07d}")
                manifest["step"] = current.step
                write_json(manifest_path, manifest)

            run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler,
                     batches=batches(data, processor, config, split="train", skip=state.microsteps,
                                     episode_limit=settings.get("episode_limit", 0), seed=seed),
                     loss_fn=loss_fn, max_steps=settings["max_steps"], accumulation_steps=settings["accumulation_steps"],
                     max_grad_norm=1.0, state=state, on_before_step=before_step, on_step=on_step)
        # A bounded held-out loss is recorded separately from rollout success.
        if "validation_l1" not in manifest:
            manifest["validation_l1"] = validation_loss(state.step)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest.update(step=state.step, invocation_seconds=time.monotonic() - started)
        write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), resume=args.resume, preflight=args.preflight)


if __name__ == "__main__":
    main()
