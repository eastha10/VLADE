"""Task-only training for native QwenVLA with causal episode state."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from vlade.data.colab_assets import sha256
from vlade.data.episode_data import EpisodeData, query_stream, write_json
from vlade.models.contracts import masked_action_l1
from vlade.models.students.backbones.qwen import load_processor, prepare_observation
from vlade.models.students.configuration_qwen_vla import QwenVLAConfig
from vlade.models.students.qwen_vla import QwenVLA
from vlade.storage.assets import resolve_input
from vlade.training.hub import HubCheckpoints
from vlade.training.loop import LoopState, evaluate, make_scheduler, restore_training_state, run_loop, training_state
from vlade.training.run_control import checkpoint_space, seal_checkpoint, verify_checkpoint
from vlade.utils.filesystem import publish_directory
from vlade.utils.source_identity import source_identity

ROOT = Path(__file__).resolve().parents[3]


def batches(data: EpisodeData, processor, config: QwenVLAConfig, split: str, seed: int, **kwargs):
    for sample, history in query_stream(data, split, seed=seed, **kwargs):
        current = prepare_observation(
            processor, config, instruction=sample["instruction"],
            primary=data.image(sample, "primary"), wrist=data.image(sample, "wrist"),
        )
        previous = history[0] if len(history) == 2 else None
        yield sample, current, previous


def action_loss(model: QwenVLA, data: EpisodeData, processor, config: QwenVLAConfig, device):
    """Carry detached state across adjacent queries, reconstruct after a gap or resume."""
    last = None
    state = None

    def forward(sample, inputs, state, step):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        proprio = torch.from_numpy(sample["proprio"])[None].to(device)
        return model(inputs, proprio, state, batch_idx=step)

    def loss(batch, step):
        nonlocal last, state
        sample, current, previous = batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if previous is None:
                state = None
            elif last != (previous["episode_id"], previous["timestep"]):
                inputs = prepare_observation(
                    processor, config, instruction=previous["instruction"],
                    primary=data.image(previous, "primary"), wrist=data.image(previous, "wrist"),
                )
                with torch.no_grad():
                    state = forward(previous, inputs, None, step).recurrent_state
            prediction = forward(sample, current, state, step)
        last = sample["episode_id"], sample["timestep"]
        state = prediction.recurrent_state.detach()
        target = torch.from_numpy(sample["gt_action"])[None].to(device)
        valid = torch.from_numpy(sample["action_valid_mask"])[None].to(device)
        return masked_action_l1(prediction, target, valid)

    def save_state():
        return last, state

    def load_state(value):
        nonlocal last, state
        last, state = value
        if state is not None:
            state = state.to(device)

    return loss, save_state, load_state


def check_adamw_capacity(model: QwenVLA, device: torch.device) -> None:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    required = parameter_count * 16
    free, _ = torch.cuda.mem_get_info(device)
    if required >= free:
        raise RuntimeError(f"FP32 AdamW needs at least {required / 1024**3:.2f} GiB before activations; {free / 1024**3:.2f} GiB free")


def run(settings: dict, output: Path, *, resume: Path | None = None, stop_after: int | None = None) -> None:
    """Run full-model behavior cloning. A bounded pilot keeps the main run's schedule."""
    target = settings["max_steps"]
    limit = stop_after if stop_after is not None else target
    accumulation = settings["accumulation_steps"]
    if not 1 <= limit <= target or min(accumulation, settings["save_steps"], settings["validation_queries"]) < 1:
        raise ValueError("Invalid step, accumulation, save or validation budget")
    source = source_identity(ROOT, require_clean=True)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 CUDA GPU is required")
    initial = resolve_input(settings["initial_checkpoint"])
    config_path = ROOT / settings["model_config"]
    config = QwenVLAConfig.from_json(config_path)
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = QwenVLA.from_pretrained(config, qwen_initial=initial) if resume is None else None
    if model is not None:
        check_adamw_capacity(model, device)
    data = EpisodeData(resolve_input(settings["data"]))
    identity = {
        "settings": settings, **source,
        "data_sha256": sha256(data.root / "manifest.json"),
        "initial_sha256": sha256(initial / "download_manifest.json"),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
    }
    if resume is not None:
        seal = verify_checkpoint(resume, accumulation=accumulation)
        if json.loads((resume / "identity.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("Checkpoint training identity differs from this run")
        if seal["step"] >= limit:
            raise ValueError("The requested stop must follow the resumed checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(output.glob("step-*"))
    if checkpoints and (resume is None or checkpoints[-1].resolve() != resume.resolve()):
        raise FileExistsError("Resume the latest checkpoint in this output or choose a fresh output")

    if resume is not None:
        model = QwenVLA.load(resume / "model")
        check_adamw_capacity(model, device)
    model.to(device)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    processor = load_processor(config, qwen_initial=initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"], foreach=False)
    scheduler = make_scheduler(optimizer, target)
    loss, get_history, set_history = action_loss(model, data, processor, config, device)
    state = LoopState()
    if resume is not None:
        payload = torch.load(resume / "trainer_state.pt", map_location="cpu", weights_only=True)
        state = restore_training_state(payload, optimizer, scheduler)
        if (state.step, state.microsteps) != (seal["step"], seal["microsteps"]):
            raise ValueError("Trainer cursor differs from checkpoint seal")
        set_history(payload["episode_history"])
        del payload

    hub = HubCheckpoints(output, resume=resume, metric="validation_l1")
    started = time.monotonic()
    write_json(output / f"invocation-{uuid.uuid4().hex[:12]}.json", {
        "identity": identity, "stop_after": limit, "resume_step": state.step,
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "dtype": "fp32_parameters_bf16_autocast", "batch_size": 1,
        "normalization_sha256": data.manifest["normalization_sha256"],
    })
    torch.cuda.reset_peak_memory_stats()
    with (output / f"metrics-{uuid.uuid4().hex[:12]}.jsonl").open("x", encoding="utf-8") as log:
        def on_step(current, metrics):
            metrics.update(step=current.step, gpu=torch.cuda.get_device_name(),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           elapsed_seconds=time.monotonic() - started)
            if current.step % settings["save_steps"] == 0 or current.step == limit:
                held_out = settings["validation_queries"]
                validation, _, _ = action_loss(model, data, processor, config, device)
                metrics["validation_l1"] = evaluate(
                    [model], batches(data, processor, config, "validation", seed, limit=held_out),
                    validation, current.step, held_out,
                )
                checkpoint_space(output, [model], optimizer)
                stage = output / f".step-{current.step:07d}-{uuid.uuid4().hex}.incomplete"
                stage.mkdir()
                model.save(stage / "model")
                processor.save_pretrained(stage / "processor")
                payload = training_state(current, optimizer, scheduler)
                payload["episode_history"] = get_history()
                torch.save(payload, stage / "trainer_state.pt")
                write_json(stage / "identity.json", identity)
                write_json(stage / "metrics.json", metrics)
                hub.prepare(stage, current.step, metrics)
                seal_checkpoint(stage, current.step, current.microsteps)
                (stage / "COMPLETE").write_text("ok\n", encoding="utf-8")
                checkpoint = output / f"step-{current.step:07d}"
                publish_directory(stage, checkpoint)
                hub.publish(checkpoint, final=current.step == target)
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps(metrics), flush=True)

        try:
            run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler,
                     batches=batches(data, processor, config, "train", seed, repeat=True, skip=state.microsteps),
                     loss_fn=loss, max_steps=limit, accumulation_steps=accumulation, state=state, on_step=on_step)
        finally:
            hub.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding="utf-8")), args.output,
        resume=args.resume, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
