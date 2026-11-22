"""Experimental S0 fake-quantization comparison. Outputs an evaluation report."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from .quantization_comparison import REFERENCE_FIELDS, atomic_json, source_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="HF asset URI or verified HF cache path")
    parser.add_argument("--data", required=True, help="HF asset URI or verified HF cache path")
    parser.add_argument("--registry", type=Path, help="Pinned VLADE HF asset registry JSON")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    for method in ("actquant", "qvla", "quant"):
        parser.add_argument(f"--{method}", choices=("on", "off"), default="off")
    parser.add_argument("--calibration-queries", type=int, default=8)
    parser.add_argument("--eval-queries", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=0, help="LIBERO+ Spatial variants, one trial each")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-bpw", type=float, default=4.0, help="ActQuant and QVLA weight budget")
    parser.add_argument("--condition", choices=("bf16", "uniform_w4", "uniform_w8", "actquant", "qvla", "quant"),
                        help="Run exactly one independent condition; old switches retain legacy combination behavior")
    parser.add_argument("--scope", choices=("language", "language_vision"))
    parser.add_argument("--reference-report", type=Path, help="Completed BF16 report from the identical source/input/seed")
    args = parser.parse_args()
    if not 2 <= args.target_bpw <= 16:
        parser.error("--target-bpw must be in [2, 16]")
    if args.calibration_queries < 3 or args.eval_queries < 1 or args.warmup < 0 or not 0 <= args.rollouts <= 2402:
        parser.error("Use at least 3 calibration queries, 1 evaluation query, nonnegative warmup, and 0–2402 rollouts")
    if args.output.exists():
        parser.error("--output already exists")
    if args.condition and any(getattr(args, m) == "on" for m in ("actquant", "qvla", "quant")):
        parser.error("Independent --condition cannot be combined with legacy switches")
    if args.condition == "quant" and (args.scope not in (None, "language") or args.target_bpw != 8):
        parser.error("QuantVLA pilot requires language scope and --target-bpw 8")
    if args.quant == "on" and args.scope == "language_vision":
        parser.error("Legacy QuantVLA also supports language-only scope")
    if args.condition in {"uniform_w4", "uniform_w8", "bf16"}:
        expected = {"uniform_w4": 4, "uniform_w8": 8, "bf16": 16}[args.condition]
        if args.target_bpw != expected:
            parser.error("Fixed-precision condition must declare its actual bit budget")
    if args.condition not in (None, "bf16") and args.reference_report is None:
        parser.error("Independent quantized condition requires --reference-report")
    return args


def select_layers(model: Any, scope: str) -> dict[str, Any]:
    """Share exactly the same declared target modules across independent methods."""
    from torch import nn
    if scope not in {"language", "language_vision"}:
        raise ValueError("Unknown quantization scope")
    layers = {name: layer for name, layer in model.named_modules()
              if type(layer) is nn.Linear and (name.startswith("backbone.language_model.")
                  or (scope == "language_vision" and name.startswith("backbone.visual.blocks.")))}
    if not layers:
        raise ValueError("No target linear weights found in the QwenVLA checkpoint")
    return layers


def reference_identity(report: dict) -> dict:
    return {key: report[key] for key in REFERENCE_FIELDS}


def evaluation_device() -> Any:
    """The research worker requires native CUDA BF16 support."""
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("Use the CUDA PyTorch environment with native BF16 support.")
    return torch.device("cuda:0")


def observations(data: Any, split: str, seed: int, limit: int) -> Iterator[dict]:
    entries = [entry for entry in data.entries if entry["split"] == split]
    if split == "train":
        entries = [entry for entry in entries if entry.get("is_calibration")]
        random.Random(seed).shuffle(entries)
    remaining = limit
    for _, queries in data.episodes(split, order=[entry["episode_id"] for entry in entries]):
        for sample in queries[:remaining]:
            yield sample
            remaining -= 1
        if not remaining:
            return
    if remaining:
        raise ValueError(f"{split} needs {limit} queries, found {limit - remaining}")


def run(args: argparse.Namespace) -> None:
    from vlade.storage.assets import resolve_input, asset_record, read_registry

    registry = read_registry(args.registry)
    def record(value):
        key = str(value)[len("hfasset://"):] if str(value).startswith("hfasset://") else Path(value).resolve().parent.parent.name
        entry = asset_record(key, args.registry)
        return {"asset_key": key, **{n: entry[n] for n in ("repo_id", "repo_type", "revision")}}
    hf_inputs = {"checkpoint": record(args.checkpoint), "data": record(args.data)}
    hf_inputs["registry_revision"] = registry.get("registry_revision")
    if args.rollouts:
        hf_inputs["benchmark_assets"] = record("hfasset://libero_plus_assets")
    args.checkpoint = resolve_input(args.checkpoint, registry=args.registry)
    args.data = resolve_input(args.data, registry=args.registry)
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.data, args.checkpoint)):
        raise SystemExit("Choose an output path outside the input directories.")

    import torch
    from transformers import AutoProcessor

    from vlade.data.colab_assets import sha256
    from vlade.data.episode_data import EpisodeData
    from vlade.evaluation.qwen_quantization import calibrate
    from vlade.models.students.backbones.qwen import prepare_observation
    from vlade.models.students.qwen_vla import QwenVLA
    from vlade.training.run_control import CHECKPOINT_MANIFEST, verify_checkpoint
    from vlade.evaluation.s0_libero_rollout import prepare_rollouts, run_rollouts

    if os.environ.get("VLADE_LOCAL_INPUTS"):
        hf_inputs["local_bindings_sha256"] = sha256(Path(os.environ["VLADE_LOCAL_INPUTS"]))

    device = evaluation_device()
    torch.manual_seed(args.seed)
    seal = verify_checkpoint(args.checkpoint)
    identity = json.loads((args.checkpoint / "identity.json").read_text(encoding="utf-8"))
    data = EpisodeData(args.data)
    data_digest = sha256(data.root / "manifest.json")
    if data_digest != identity["data_sha256"]:
        raise ValueError("The prepared data manifest differs from the checkpoint training data.")
    selected = [name for name in ("quant", "actquant", "qvla") if getattr(args, name) == "on"]
    modes = {args.condition: [] if args.condition == "bf16" else [args.condition]} if args.condition else {"bf16": []}
    if selected and args.condition is None:
        modes["+".join(selected)] = selected
    rollout_plan, rollout_error, benchmark_identity = None, None, None
    if args.rollouts:
        try:
            from vlade.runtime.benchmark_source import verify
            from vlade.runtime.benchmark import prepare
            root = Path(__file__).resolve().parents[3]
            spec = json.loads((root / "configs/runtime/student_eval.json").read_text(encoding="utf-8"))
            verify(spec, Path(sys.executable))
            runtime_record = json.loads((Path(sys.executable).parent.parent / "benchmark-source.json").read_text())
            prepare(args.output.parent / "benchmark-runtime")
            from libero.libero import benchmark
            if not Path(benchmark.__file__).resolve().is_relative_to(Path(runtime_record["source"]).resolve()):
                raise ValueError("Imported benchmark is outside the verified pinned source")
            benchmark_identity = spec["benchmark_source"]
            rollout_plan = prepare_rollouts(args.rollouts, args.seed)
        except Exception as exc:
            rollout_error = str(exc)
            print(f"Rollout unavailable: {rollout_error}", flush=True)
    report = {
        "schema_version": 2,
        "status": "running",
        "source": source_identity(),
        "hf_inputs": hf_inputs,
        "benchmark_source": benchmark_identity,
        "checkpoint_step": seal["step"],
        "checkpoint_seal_sha256": sha256(args.checkpoint / CHECKPOINT_MANIFEST),
        "data_manifest_sha256": data_digest,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "batch_size": 1,
        "seed": args.seed,
        "calibration_queries": args.calibration_queries,
        "eval_queries": args.eval_queries,
        "normalization_sha256": data.manifest["normalization_sha256"],
        "split_sha256": data.manifest["split_sha256"],
        "warmup": args.warmup,
        "target_bpw": args.target_bpw,
        "quantization_order": (modes[args.condition] if args.condition else selected),
        "rollout_count_requested": args.rollouts,
        "measurement": "synchronized model forward, excluding observation preprocessing and transfers",
        "runs": {},
    }
    reference = []
    reference_ids = []
    if args.reference_report:
        saved = json.loads(args.reference_report.read_text(encoding="utf-8"))
        if saved.get("status") != "completed" or reference_identity(saved) != reference_identity(report):
            raise ValueError("BF16 reference is incomplete or belongs to another source/input/seed")
        reference = [torch.tensor(row["actions"], dtype=torch.float32) for row in saved["reference_actions"]]
        reference_ids = [row["query"] for row in saved["reference_actions"]]
        if len(reference) != args.eval_queries:
            raise ValueError("BF16 reference query count differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the file before the first model load; every completed mode survives later failures.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)

    def run_mode(mode, stages):
        gc.collect()
        torch.cuda.empty_cache()
        model = QwenVLA.load(args.checkpoint / "model").to(dtype=torch.bfloat16).to(device).eval()
        config = model.student_config
        processor = AutoProcessor.from_pretrained(
            args.checkpoint / "processor", local_files_only=True, trust_remote_code=False,
            use_fast=False, size={"shortest_edge": config.image_size**2, "longest_edge": config.image_size**2},
        )
        samples = {}

        def evaluate(sample, state):
            inputs = prepare_observation(
                processor, config, instruction=sample["instruction"],
                primary=data.image(sample, "primary"), wrist=data.image(sample, "wrist"),
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            proprio = torch.as_tensor(sample["proprio"], device=device)[None]
            torch.cuda.synchronize()
            start = perf_counter()
            with torch.autocast(device.type, dtype=torch.bfloat16):
                output = model(inputs, proprio, state)
            torch.cuda.synchronize()
            return output, (perf_counter() - start) * 1_000

        def run(split, limit, consume, *, gradients):
            episode, state = None, None
            samples[split] = []
            with torch.set_grad_enabled(gradients):
                for sample in observations(data, split, args.seed, limit):
                    if episode != sample["episode_id"]:
                        episode, state = sample["episode_id"], None
                    output, elapsed = evaluate(sample, state)
                    target = torch.as_tensor(sample["gt_action"], dtype=torch.float32, device=device)[None]
                    valid = torch.as_tensor(sample["action_valid_mask"], dtype=torch.bool, device=device)[None]
                    if not valid.any():
                        raise ValueError("Query has no valid action targets.")
                    if not torch.isfinite(output.actions).all():
                        raise FloatingPointError("Non-finite model actions")
                    consume(output.actions, target, valid, elapsed, sample)
                    state = output.recurrent_state.detach()
                    samples[split].append([episode, sample["timestep"]])
            return len(samples[split])

        handles, allocations = [], {}
        for stage in stages:
            scope = args.scope or ("language" if stage == "quant" else "language_vision")
            layers = select_layers(model, scope)
            model.requires_grad_(False)
            if stage != "quant":
                for layer in layers.values():
                    layer.weight.requires_grad_(True)

            def collect(callback, *, gradients):
                return run("train", args.calibration_queries,
                            lambda actions, target, valid, _elapsed, _sample: callback(actions, target, valid),
                           gradients=gradients)

            if stage in {"uniform_w4", "uniform_w8"}:
                from .qwen_quantization import round_weights
                bits = 4 if stage == "uniform_w4" else 8
                with torch.no_grad():
                    for layer in layers.values():
                        layer.weight.copy_(round_weights(layer.weight.float(), bits))
                allocation = {"calibration_queries": 0, "weight_bpw_excluding_metadata": bits,
                              "quantized_scope_parameters": sum(m.weight.numel() for m in layers.values()),
                              "bit_allocation": dict.fromkeys(layers, bits)}
                stage_handles = []
            else:
                allocation, stage_handles = calibrate(stage, layers, collect, args.target_bpw, args.seed)
            total = sum(p.numel() for p in model.parameters())
            count = allocation["quantized_scope_parameters"]
            allocation.update(quantization_scope=scope, target_bpw=8 if stage == "quant" else args.target_bpw,
                target_modules={n: m.weight.numel() for n, m in layers.items()}, total_model_parameters=total,
                full_model_weight_bpw_excluding_metadata=(16 * (total - count) + count * allocation["weight_bpw_excluding_metadata"]) / total)
            allocations[stage] = allocation
            handles.extend(stage_handles)
            model.requires_grad_(False)

        latency, deviation, errors = [], [], []
        try:
            if args.warmup:
                run("validation", args.warmup, lambda *_: None, gradients=False)
            torch.cuda.reset_peak_memory_stats()

            def measure(actions, target, valid, elapsed, sample):
                actions = actions[valid].float().cpu()
                errors.append((actions - target[valid].cpu()).abs())
                index = len(latency)
                if mode == "bf16":
                    reference.append(actions)
                    reference_ids.append([sample["episode_id"], sample["timestep"]])
                if reference_ids[index] != [sample["episode_id"], sample["timestep"]] or reference[index].shape != actions.shape:
                    raise ValueError("BF16 query order/action layout mismatch")
                deviation.append((actions - reference[index]).abs())
                latency.append(elapsed)

            run("validation", args.eval_queries, measure, gradients=False)
            result = {
                **(allocations[stages[0]] if len(stages) == 1 else {}),
                "quantization_stages": allocations,
                "scope": "bf16" if mode == "bf16" else "fake_quantization_pilot",
                "samples": samples,
                "action_mae_vs_ground_truth": torch.cat(errors).mean().item(),
                "action_mae_vs_bf16": torch.cat(deviation).mean().item(),
                "forward_p50_ms": torch.tensor(latency).quantile(0.5).item(),
                "forward_p95_ms": torch.tensor(latency).quantile(0.95).item(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            if args.rollouts:
                if rollout_error is not None:
                    result["rollout"] = {"status": "unavailable", "reason": rollout_error}
                else:
                    def predict(instruction, primary, wrist, proprio, state):
                        inputs = prepare_observation(processor, config, instruction=instruction,
                                                     primary=primary, wrist=wrist)
                        inputs = {key: value.to(device) for key, value in inputs.items()}
                        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                            output = model(inputs, torch.as_tensor(proprio, device=device)[None], state)
                        return output.actions[0].float().cpu().numpy(), output.recurrent_state

                    result["rollout"] = run_rollouts(rollout_plan, predict, data.stats)
            return result
        finally:
            for handle in handles:
                handle.remove()

    try:
        for mode, stages in modes.items():
            report["current_condition"] = mode
            atomic_json(args.output, report)
            print(f"Running {mode}", flush=True)
            result = run_mode(mode, stages)
            report["runs"][mode] = result
            if mode == "bf16":
                report["reference_actions"] = [{"query": key, "actions": action.tolist()}
                                                for key, action in zip(reference_ids, reference)]
            atomic_json(args.output, report)
            print(f"{mode}: action MAE {result['action_mae_vs_ground_truth']:.6f}", flush=True)
        report["status"] = "completed"
        atomic_json(args.output, report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        atomic_json(args.output, report)
        raise
    print(args.output)


def main() -> None:
    args = parse_args()
    previous = os.environ.get("VLADE_ASSET_REGISTRY")
    if args.registry is not None:
        os.environ["VLADE_ASSET_REGISTRY"] = str(args.registry.resolve())
    try:
        run(args)
    finally:
        if previous is None:
            os.environ.pop("VLADE_ASSET_REGISTRY", None)
        else:
            os.environ["VLADE_ASSET_REGISTRY"] = previous


if __name__ == "__main__":
    main()
