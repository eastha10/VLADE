"""Behavior cloning and action distillation for native QwenVLA."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

from vlade.data.colab_assets import sha256
from vlade.data.episode_data import EpisodeData, query_stream, write_json
from vlade.distillation.action import action_distillation_loss, validate_weights
from vlade.legacy.plan_cot.distillation.teacher_plan import plan_token_budget
from vlade.distillation.representation import (ActionRepresentationKD, representation_settings,
    representation_contract, save_representation, load_representation)
from vlade.models.contracts import masked_action_l1
from vlade.models.students.backbones.qwen import load_processor, load_saved_processor, prepare_observation
from vlade.models.students.configuration_qwen_vla import QwenVLAConfig
from vlade.models.students.qwen_vla import QwenVLA
from vlade.storage.assets import resolve_input
from vlade.training.cached_actions import CachedActions
from vlade.training.hub import HubCheckpoints
from vlade.training.loop import LoopState, evaluate, make_scheduler, restore_training_state, run_loop, training_state
from vlade.training.run_control import checkpoint_space, seal_checkpoint, verify_checkpoint
from vlade.training.teacher_inference import InferenceActions, teacher_identity
from vlade.training.scopes import configure_trainable_scope
from vlade.training.memory import (adamw_storage, configure_qwen_memory, cuda_memory,
                                   record_memory, validate_memory_profile)
from vlade.training.residency import StudentOffload, validate_teacher_residency
from vlade.utils.filesystem import publish_directory
from vlade.utils.source_identity import source_identity
from vlade.utils.timing import RunTimings

ROOT = Path(__file__).resolve().parents[3]


def observation_inputs(data: EpisodeData, processor, config: QwenVLAConfig, sample: dict,
                       plan: str | None = None) -> dict[str, torch.Tensor]:
    return prepare_observation(
        processor, config, instruction=sample["instruction"],
        primary=data.image(sample, "primary"), wrist=data.image(sample, "wrist"),
        timestep=sample["timestep"], teacher_plan=plan,
    )


def batches(data: EpisodeData, processor, config: QwenVLAConfig, split: str, seed: int, *,
            with_plan: bool = False, **kwargs):
    for sample, history in query_stream(data, split, seed=seed, **kwargs):
        current = None if with_plan else observation_inputs(data, processor, config, sample)
        previous = history[0] if len(history) == 2 else None
        yield sample, current, previous


def action_loss(model: QwenVLA, data: EpisodeData, processor, config: QwenVLAConfig, device, *,
                teacher: CachedActions | InferenceActions | None = None,
                plan_teacher: InferenceActions | None = None,
                plan_log: TextIO | None = None,
                task_weight: float = 1.0, action_weight: float = 1.0,
                representation: ActionRepresentationKD | None = None,
                representation_weight: float = 0.0, representation_layer: int = -2,
                step_offset: int = 0, loss_terms: list[dict[str, float]] | None = None,
                timings: RunTimings | None = None, phase: str = "train"):
    """Carry detached state across adjacent queries, reconstruct after a gap or resume."""
    last = None
    state = None
    forward_calls = 0

    def forward(sample, inputs, state, step, *, replay=False):
        nonlocal forward_calls
        inputs = {key: value.to(device) for key, value in inputs.items()}
        proprio = torch.from_numpy(sample["proprio"])[None].to(device)
        with (timings.measure("s0_forward", step=step + 1, phase=phase,
                              episode_id=sample["episode_id"], timestep=sample["timestep"],
                              is_warmup=forward_calls < 2, replay=replay)
              if timings is not None else nullcontext()):
            # Each forward owns its autocast cache. Replay under no_grad must not
            # retain BF16 parameter casts across a Teacher offload or a grad forward.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(inputs, proprio, state, batch_idx=step + step_offset)
        forward_calls += 1
        return result

    def replay_validation(sample, previous, step):
        episode = sample["episode_id"]
        _, queries = next(data.episodes("validation", order=[episode]))
        start = last[1] + 8 if last is not None and last[0] == episode else 0
        replay = state if start else None
        with torch.no_grad():
            for query in queries[start // 8:previous["timestep"] // 8 + 1]:
                plan = plan_teacher.get_plan(query, split="validation") if plan_teacher is not None else None
                inputs = observation_inputs(data, processor, config, query, plan)
                replay = forward(query, inputs, replay, step, replay=True).recurrent_state
        return replay

    def loss(batch, step):
        nonlocal last, state
        sample, current, previous = batch
        teacher_target = None
        teacher_hidden = None
        plan_row = None
        # Exclusive residency must finish T1 inference before retaining a Student
        # autograd graph. The loss, causal teacher prefix and query stay identical.
        if teacher is not None:
            if representation is not None:
                signal = teacher.get_representation(sample,
                    dimension=representation.projection.out_features, layer=representation_layer)
                teacher_target, teacher_hidden = signal.action, signal.hidden
            else:
                teacher_target = teacher.get(sample)
        if plan_teacher is not None:
            if plan_log is not None:
                plan_row = {key: sample[key] for key in ("episode_id", "timestep", "instruction")}
                plan_row.update(split=phase, optimizer_step=step + (phase == "train"))
            try:
                if phase == "train":
                    signal = plan_teacher.get_signal(sample)
                    plan, teacher_target = signal.plan, signal.action
                else:
                    plan = plan_teacher.get_plan(sample, split=phase)
            except (RuntimeError, ValueError, TimeoutError) as error:
                if plan_row is not None:
                    plan_row.update(status="plan_error", error_type=type(error).__name__, error=str(error))
                    print(json.dumps(plan_row, ensure_ascii=False), file=plan_log, flush=True)
                raise
            current = observation_inputs(data, processor, config, sample, plan)
            if plan_row is not None:
                plan_row.update(status="ok", plan=plan)
        if previous is None:
            state = None
        elif last != (previous["episode_id"], previous["timestep"]):
            if phase != "validation":
                plan = plan_teacher.get_plan(previous, split=phase) if plan_teacher is not None else None
                inputs = observation_inputs(data, processor, config, previous, plan)
                with torch.no_grad():
                    state = forward(previous, inputs, None, step, replay=True).recurrent_state
            else:
                state = replay_validation(sample, previous, step)
        prediction = forward(sample, current, state, step)
        last = sample["episode_id"], sample["timestep"]
        state = prediction.recurrent_state.detach()
        target = torch.from_numpy(sample["gt_action"])[None].to(device)
        valid = torch.from_numpy(sample["action_valid_mask"])[None].to(device)
        if teacher_target is not None:
            total, terms = action_distillation_loss(
                prediction, target, teacher_target[None].to(device), valid,
                task_weight=task_weight, action_weight=action_weight,
            )
            if representation is not None:
                rep_loss = representation(prediction.recurrent_state,
                    teacher_hidden[None].to(device), valid)
                total = total + representation_weight * rep_loss
                terms["representation_mse"] = float(rep_loss.detach())
            if loss_terms is not None:
                loss_terms.append(terms)
        else:
            total = masked_action_l1(prediction, target, valid)
            terms = None
        if plan_row is not None:
            plan_row.update(terms or {"task_l1": float(total.detach()), "action_kd_l1": None})
            print(json.dumps(plan_row, ensure_ascii=False), file=plan_log, flush=True)
        return total

    def save_state():
        return last, state

    def load_state(value):
        nonlocal last, state
        last, state = value
        if state is not None:
            state = state.to(device)

    return loss, save_state, load_state


def check_adamw_capacity(model: QwenVLA, device: torch.device, *, reserve_gib: float = 0) -> dict:
    storage = adamw_storage(model)
    required = storage["minimum_training_bytes"]
    free, _ = torch.cuda.mem_get_info(device)
    if required + reserve_gib * 1024**3 >= free:
        raise RuntimeError(f"FP32 AdamW needs at least {required / 1024**3:.2f} GiB plus "
                           f"{reserve_gib:.2f} GiB activation reserve; {free / 1024**3:.2f} GiB free")
    return storage


def verified_s0(initial: Path, config: QwenVLAConfig, data_sha256: str) -> int:
    """Check native S0 architecture and training data before initializing a new KD run."""
    seal = verify_checkpoint(initial)
    saved_config = json.loads((initial / "model/student.json").read_text(encoding="utf-8"))
    if saved_config != asdict(config):
        raise ValueError("Initial S0 must match the selected native QwenVLA configuration")
    saved_identity = json.loads((initial / "identity.json").read_text(encoding="utf-8"))
    if saved_identity.get("data_sha256") != data_sha256:
        raise ValueError("S0 and action distillation must use the same prepared data manifest")
    backbone = initial / "model/backbone"
    if (not (initial / "model/components.pt").is_file() or not (initial / "processor").is_dir()
            or not (backbone / "config.json").is_file()
            or not any((backbone / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json"))):
        raise ValueError("S0 must include a complete backbone, policy components and processor")
    return seal["step"]


def run(settings: dict, output: Path, *, resume: Path | None = None, stop_after: int | None = None,
        local_s0: Path | None = None, local_s0_manifest_sha256: str | None = None,
        preflight: bool = False) -> None:
    """Train QwenVLA. A bounded pilot keeps the configured run schedule."""
    target = settings["max_steps"]
    limit = stop_after if stop_after is not None else target
    accumulation = settings["accumulation_steps"]
    profile = validate_memory_profile(settings)
    if not 1 <= limit <= target or min(accumulation, settings["save_steps"], settings["validation_queries"]) < 1:
        raise ValueError("Invalid step, accumulation, save or validation budget")
    kd = settings.get("distillation")
    rep_settings = representation_settings(kd or {})
    plan_tokens = plan_token_budget(kd or {})
    initialization = settings.get("initialization", "student_checkpoint" if kd is not None else "pretrained_backbone")
    if initialization not in {"pretrained_backbone", "student_checkpoint"}:
        raise ValueError("Unknown QwenVLA initialization")
    from_s0 = initialization == "student_checkpoint"
    if kd is not None and not from_s0:
        raise ValueError("QwenVLA Action KD requires a verified S0 checkpoint")
    if (local_s0 is None) != (local_s0_manifest_sha256 is None) or (local_s0 is not None and not from_s0):
        raise ValueError("Local S0 requires student_checkpoint initialization, a path, and its manifest SHA256")
    episode_limit = 0
    teacher_mode = None
    if kd is not None:
        validate_teacher_residency(kd)
        validate_weights(kd["task_weight"], kd["action_kd_weight"])
        teacher_mode = kd.get("teacher_mode", "cache")
        if teacher_mode not in {"cache", "inference"}:
            raise ValueError("Teacher mode must be cache or inference")
        if teacher_mode == "inference" and not Path(kd.get("teacher_python", "")).is_file():
            raise FileNotFoundError("Set up the AVA inference runtime and pass --teacher-python")
        episode_limit = kd.get("episode_limit", 0)
        if type(episode_limit) is not int or episode_limit < 0:
            raise ValueError("Teacher episode_limit must be a nonnegative integer")
    source = source_identity(ROOT, require_clean=True)
    device = torch.device("cuda:0")
    if not preflight and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("A BF16 CUDA GPU is required")
    if local_s0 is not None:
        initial = resolve_input(local_s0)
        if sha256(initial / "checkpoint_manifest.json") != local_s0_manifest_sha256.lower():
            raise ValueError("Local S0 checkpoint manifest does not match the pinned SHA256")
    else:
        initial = resolve_input(settings["initial_checkpoint"])
    config_path = ROOT / settings["model_config"]
    config = QwenVLAConfig.from_json(config_path)
    data = EpisodeData(resolve_input(settings["data"]))
    data_sha256 = sha256(data.root / "manifest.json")
    s0_step = verified_s0(initial, config, data_sha256) if from_s0 else 0
    identity = {
        "settings": settings, **source,
        "data_sha256": data_sha256,
        "initial_sha256": sha256(initial / ("checkpoint_manifest.json" if from_s0 else "download_manifest.json")),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
    }
    if from_s0:
        identity["initial_student_step"] = s0_step
    teacher = None
    if teacher_mode == "cache":
        teacher = CachedActions(resolve_input(kd["teacher_cache"]), data, episode_limit=episode_limit)
        identity["teacher_cache"] = teacher.signature
    elif teacher_mode == "inference":
        teacher_root, identity["teacher_inference"] = teacher_identity(kd["teacher_checkpoint"], data)
        if rep_settings is not None:
            identity["representation"] = representation_contract(teacher_root,
                student_dim=config.hidden_size, student_layer=config.temporal_feature_layer,
                teacher_layer=identity["teacher_inference"]["temporal_feature_layer"],
                teacher_python=Path(kd["teacher_python"]))
        if plan_tokens:
            from vlade.legacy.plan_cot.distillation.teacher_plan import plan_generation_provenance
            signature = identity["teacher_inference"]
            plan_generation_provenance({"step": signature["step"],
                                      "plan_supervision": signature.get("plan_supervision")})
    if resume is not None:
        seal = verify_checkpoint(resume, accumulation=accumulation)
        if json.loads((resume / "identity.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("Checkpoint training identity differs from this run")
        if rep_settings is not None and "representation.pt" not in seal["files"]:
            raise ValueError("S2 resume requires its sealed training projection")
        if seal["step"] > target or (not preflight and seal["step"] >= limit):
            raise ValueError("The requested stop must follow the resumed checkpoint")
    if preflight:
        print(json.dumps({"status": "preflight_ok", "counts": data.manifest["counts"],
                          "initial_student_step": s0_step, "teacher_mode": teacher_mode,
                          "teacher_plan_max_new_tokens": plan_tokens,
                          "representation": identity.get("representation"),
                          "memory_profile": profile, "effective_batch_size": accumulation,
                          "gpu_forward_tested": False,
                          "teacher_inference": identity.get("teacher_inference"),
                          "teacher_cache": identity.get("teacher_cache")}), flush=True)
        return
    checkpoints = sorted(output.glob("step-*"))
    if checkpoints and (resume is None or checkpoints[-1].resolve() != resume.resolve()):
        raise FileExistsError("Resume the latest checkpoint in this output or choose a fresh output")

    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if from_s0:
        checkpoint = resume or initial
        model = QwenVLA.load(checkpoint / "model").float()
        processor = load_saved_processor(config, checkpoint / "processor")
    else:
        model = (QwenVLA.load(resume / "model") if resume is not None
                 else QwenVLA.from_pretrained(config, qwen_initial=initial))
        processor = load_processor(config, qwen_initial=initial)
    scope_summary = configure_trainable_scope(model, settings, from_s0=from_s0)
    print(json.dumps(scope_summary), flush=True)
    representation = None
    if rep_settings is not None:
        contract = identity["representation"]
        representation = ActionRepresentationKD(contract["student_dim"], contract["teacher_dim"])
        if resume is not None:
            load_representation(representation, resume / "representation.pt", contract)
    modules = [model] + ([representation] if representation is not None else [])
    training_modules = torch.nn.ModuleList(modules)
    storage = check_adamw_capacity(training_modules, device, reserve_gib=profile["reserve_gib"] if profile else 0)
    runtime_summary = configure_qwen_memory(model, settings)
    training_modules.to(device)
    optimizer = torch.optim.AdamW((p for p in training_modules.parameters() if p.requires_grad), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"], foreach=False)
    scheduler = make_scheduler(optimizer, target)
    loss_terms: list[dict[str, float]] = []
    weights = ({"task_weight": kd["task_weight"], "action_weight": kd["action_kd_weight"]}
               if kd is not None else {})
    state = LoopState()
    if resume is not None:
        payload = torch.load(resume / "trainer_state.pt", map_location="cpu", weights_only=True)
        state = restore_training_state(payload, optimizer, scheduler)
        if (state.step, state.microsteps) != (seal["step"], seal["microsteps"]):
            raise ValueError("Trainer cursor differs from checkpoint seal")
        history = payload["episode_history"]
        del payload

    output.mkdir(parents=True, exist_ok=True)
    hub = HubCheckpoints(output, resume=resume, metric="validation_l1")
    started = time.monotonic()
    invocation_id = uuid.uuid4().hex[:12]
    memory_path = output / f"memory-{invocation_id}.jsonl" if profile else None
    if memory_path is not None:
        record_memory(memory_path, device, "student_ready", **storage, **runtime_summary)
    plan_log_path = output / f"teacher-plans-{invocation_id}.jsonl"
    write_json(output / f"invocation-{invocation_id}.json", {
        "identity": identity, "stop_after": limit, "resume_step": state.step,
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "dtype": "fp32_parameters_bf16_autocast", "batch_size": 1,
        **scope_summary,
        **runtime_summary, "storage_estimate": storage,
        "memory_log": memory_path.name if memory_path else None,
        "normalization_sha256": data.manifest["normalization_sha256"],
        **({"teacher_plan_log": plan_log_path.name} if plan_tokens else {}),
    })
    torch.cuda.reset_peak_memory_stats()
    with (RunTimings(output, role="s0", device=str(device), metadata={
        "gpu": torch.cuda.get_device_name(), "dtype": "fp32_parameters_bf16_autocast",
        "batch_size": 1, "action_chunk_size": 8, "seed": seed,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "initial_sha256": identity["initial_sha256"], "data_sha256": data_sha256,
        **scope_summary, **runtime_summary,
          }) as timings,
          (output / f"metrics-{invocation_id}.jsonl").open("x", encoding="utf-8") as log,
          (plan_log_path.open("x", encoding="utf-8") if plan_tokens else nullcontext()) as plan_log):
        def on_step(current, metrics):
            if loss_terms:
                metrics.update({key: sum(row[key] for row in loss_terms) / len(loss_terms)
                                for key in loss_terms[0]})
                loss_terms.clear()
            elif from_s0:
                metrics["task_l1"] = metrics["loss"]
            metrics.update(step=current.step, gpu=torch.cuda.get_device_name(),
                           **cuda_memory(device),
                           elapsed_seconds=time.monotonic() - started)
            if current.step % settings["save_steps"] == 0 or current.step == limit:
                held_out = settings["validation_queries"]
                validation, _, _ = action_loss(model, data, processor, config, device,
                                               plan_teacher=online_teacher if plan_tokens else None,
                                               plan_log=plan_log, timings=timings, phase="validation")
                with timings.measure("s0_validation", step=current.step, phase="validation"):
                    metrics["validation_l1"] = evaluate(
                        [model], batches(data, processor, config, "validation", seed,
                                         limit=held_out, with_plan=bool(plan_tokens)),
                        validation, current.step, held_out,
                    )
                save_started = timings.start()
                checkpoint_space(output, modules, optimizer)
                stage = output / f".step-{current.step:07d}-{uuid.uuid4().hex}.incomplete"
                stage.mkdir()
                model.save(stage / "model")
                if representation is not None:
                    save_representation(representation, stage / "representation.pt", identity["representation"])
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
                metrics["checkpoint_save_seconds"] = timings.finish(
                    save_started, "s0_checkpoint_save", step=current.step, phase="save")
            if memory_path is not None:
                record_memory(memory_path, device, "optimizer_boundary", step=current.step,
                              validation_and_save=current.step % settings["save_steps"] == 0 or current.step == limit)
                metrics.update(cuda_memory(device))
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps(metrics), flush=True)

        online_teacher = None
        try:
            if teacher_mode == "inference":
                with timings.measure("t1_worker_startup", phase="setup"):
                    offload = (StudentOffload(training_modules, optimizer, device, timings=timings, memory_path=memory_path)
                               if kd.get("teacher_residency") == "exclusive" else None)
                    online_teacher = InferenceActions(kd, data, timings=timings, request_context=offload)
                if online_teacher.signature != identity["teacher_inference"]:
                    raise ValueError("T1 identity changed before training")
                teacher = online_teacher
            loss, get_history, set_history = action_loss(
                model, data, processor, config, device, teacher=None if plan_tokens else teacher,
                plan_teacher=online_teacher if plan_tokens else None, step_offset=s0_step,
                representation=representation,
                representation_weight=rep_settings["weight"] if rep_settings else 0.0,
                representation_layer=identity.get("representation", {}).get("teacher_layer", -2),
                plan_log=plan_log, loss_terms=loss_terms, timings=timings, **weights,
            )
            if resume is not None:
                set_history(history)
            run_loop(modules=modules, optimizer=optimizer, scheduler=scheduler,
                     batches=batches(data, processor, config, "train", seed, repeat=True,
                                     skip=state.microsteps, episode_limit=episode_limit, with_plan=bool(plan_tokens)),
                     loss_fn=loss, max_steps=limit, accumulation_steps=accumulation, state=state,
                     on_step=on_step, timings=timings)
        except torch.cuda.OutOfMemoryError:
            if memory_path is not None:
                record_memory(memory_path, device, "cuda_out_of_memory", step=state.step)
            raise
        finally:
            try:
                if online_teacher is not None:
                    online_teacher.close()
            finally:
                hub.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--preflight", action="store_true", help="Verify input identity without loading models")
    parser.add_argument("--local-s0", type=Path, help="Verified HF-cached QwenVLA checkpoint for Student initialization")
    parser.add_argument("--local-s0-manifest-sha256", help="Expected SHA256 of the local checkpoint manifest")
    parser.add_argument("--teacher-python", type=Path, help="Python executable from the prepared AVA runtime")
    parser.add_argument("--action-kd-weight", type=float, help="Override lambda for a new KD run")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    settings.pop("workflow", None)
    if args.action_kd_weight is not None:
        if settings.get("distillation") is None:
            parser.error("--action-kd-weight requires a distillation configuration")
        settings["distillation"]["action_kd_weight"] = args.action_kd_weight
    if args.teacher_python is not None:
        if settings.get("distillation", {}).get("teacher_mode") != "inference":
            parser.error("--teacher-python requires distillation.teacher_mode=inference")
        # Invoking the venv through its own Python path preserves pyvenv.cfg.
        settings["distillation"]["teacher_python"] = str(args.teacher_python.absolute())
    run(settings, args.output,
        resume=args.resume, stop_after=args.stop_after,
        local_s0=args.local_s0, local_s0_manifest_sha256=args.local_s0_manifest_sha256,
        preflight=args.preflight)


if __name__ == "__main__":
    main()
