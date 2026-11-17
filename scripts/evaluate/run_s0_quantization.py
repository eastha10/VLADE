"""Experimental S0 fake-quantization comparison. Outputs an evaluation report."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
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
    args = parser.parse_args()
    if not 2 <= args.target_bpw <= 16:
        parser.error("--target-bpw must be in [2, 16]")
    if args.calibration_queries < 3 or args.eval_queries < 1 or args.warmup < 0 or not 0 <= args.rollouts <= 2402:
        parser.error("Use at least 3 calibration queries, 1 evaluation query, nonnegative warmup, and 0–2402 rollouts")
    if args.output.exists():
        parser.error("--output already exists")
    return args


def observations(data, split, seed, limit):
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


def main():
    args = parse_args()
    from vlade.storage.assets import resolve_input

    args.checkpoint = resolve_input(args.checkpoint, registry=args.registry)
    args.data = resolve_input(args.data, registry=args.registry)
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.data, args.checkpoint)):
        raise SystemExit("Choose an output path outside the input directories.")

    import torch
    from torch import nn
    from transformers import AutoProcessor

    from vlade.data.colab_assets import sha256
    from vlade.data.episode_data import EpisodeData
    from vlade.evaluation.qwen_quantization import calibrate
    from vlade.models.students.backbones.qwen import prepare_observation
    from vlade.models.students.qwen_vla import QwenVLA
    from vlade.training.run_control import CHECKPOINT_MANIFEST, verify_checkpoint
    from vlade.evaluation.s0_libero_rollout import prepare_rollouts, run_rollouts

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("Use the local CUDA PyTorch environment with BF16 support.")
    torch.manual_seed(args.seed)
    seal = verify_checkpoint(args.checkpoint)
    identity = json.loads((args.checkpoint / "identity.json").read_text(encoding="utf-8"))
    data = EpisodeData(args.data)
    data_digest = sha256(data.root / "manifest.json")
    if data_digest != identity["data_sha256"]:
        raise ValueError("The prepared data manifest differs from the checkpoint training data.")
    device = torch.device("cuda:0")
    selected = [name for name in ("quant", "actquant", "qvla") if getattr(args, name) == "on"]
    modes = {"bf16": []}
    if selected:
        modes["+".join(selected)] = selected
    rollout_plan, rollout_error = None, None
    if args.rollouts:
        try:
            rollout_plan = prepare_rollouts(args.rollouts, args.seed)
        except Exception as exc:
            rollout_error = str(exc)
            print(f"Rollout unavailable: {rollout_error}", flush=True)
    report = {
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
        "warmup": args.warmup,
        "target_bpw": args.target_bpw,
        "quantization_order": selected,
        "rollout_count_requested": args.rollouts,
        "measurement": "synchronized model forward, excluding observation preprocessing and transfers",
        "runs": {},
    }
    reference = []

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
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
                    consume(output.actions, target, valid, elapsed)
                    state = output.recurrent_state.detach()
                    samples[split].append([episode, sample["timestep"]])
            return len(samples[split])

        handles, allocations = [], {}
        for stage in stages:
            layers = {
                name: layer for name, layer in model.named_modules()
                if type(layer) is nn.Linear and (
                    name.startswith("backbone.language_model.")
                    or (stage != "quant" and name.startswith("backbone.visual.blocks."))
                )
            }
            if not layers:
                raise ValueError("No target linear weights found in the QwenVLA checkpoint.")
            model.requires_grad_(False)
            if stage != "quant":
                for layer in layers.values():
                    layer.weight.requires_grad_(True)

            def collect(callback, *, gradients):
                return run("train", args.calibration_queries,
                           lambda actions, target, valid, _: callback(actions, target, valid),
                           gradients=gradients)

            allocation, stage_handles = calibrate(stage, layers, collect, args.target_bpw, args.seed)
            allocations[stage] = allocation
            handles.extend(stage_handles)
            model.requires_grad_(False)

        latency, deviation, errors = [], [], []
        try:
            if args.warmup:
                run("validation", args.warmup, lambda *_: None, gradients=False)
            torch.cuda.reset_peak_memory_stats()

            def measure(actions, target, valid, elapsed):
                actions = actions[valid].float().cpu()
                errors.append((actions - target[valid].cpu()).abs())
                index = len(latency)
                if mode == "bf16":
                    reference.append(actions)
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
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            output = model(inputs, torch.as_tensor(proprio, device=device)[None], state)
                        return output.actions[0].float().cpu().numpy(), output.recurrent_state

                    result["rollout"] = run_rollouts(rollout_plan, predict, data.stats)
            return result
        finally:
            for handle in handles:
                handle.remove()

    for mode, stages in modes.items():
        print(f"Running {mode}", flush=True)
        result = run_mode(mode, stages)
        report["runs"][mode] = result
        print(f"{mode}: action MAE {result['action_mae_vs_ground_truth']:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
