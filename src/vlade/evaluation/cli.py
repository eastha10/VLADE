"""Compatibility CLI for the registered, project-owned LIBERO evaluation entry."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", choices=("ava_vla", "openvla_oft", "qwen"), default="ava_vla")
    parser.add_argument("--pretrained_checkpoint")
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local_log_dir", type=Path, required=True)
    parser.add_argument("--num_open_loop_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    # Only the maintained LIBERO/L1 baseline options are accepted.
    for name in ("use_l1_regression", "use_diffusion", "use_film", "use_proprio", "center_crop"):
        parser.add_argument("--" + name)
    parser.add_argument("--num_images_in_input", type=int, default=2)
    for name in (
        "temporal_strategy",
        "temporal_feature_source",
        "temporal_feature_layer",
        "attn_weight_force_eager_attn",
        "attn_weight_extra_layer_ids",
        "attn_weight_head_type",
        "attn_weight_score_config",
        "attn_weight_sink_ids",
        "attn_weight_sink_weight",
    ):
        parser.add_argument("--multi_frame." + name)
    args = parser.parse_args(argv)
    if args.num_trials_per_task < 1 or not 1 <= args.num_open_loop_steps <= 8:
        parser.error("Rollout count must be positive; open-loop steps must be between 1 and 8")
    if (
        args.use_diffusion == "True"
        or args.use_film == "True"
        or args.num_images_in_input != 2
        or args.use_l1_regression == "False"
        or args.use_proprio == "False"
    ):
        parser.error("This research path uses native two-camera continuous L1 only")
    expected = {
        "temporal_strategy": "attn_weight",
        "temporal_feature_source": "action",
        "attn_weight_force_eager_attn": "True",
        "attn_weight_extra_layer_ids": "(i for i in range(0, 32))",
        "attn_weight_head_type": "softmaxscore",
        "attn_weight_score_config": "(1.9, 0.1, 0.0)",
        "attn_weight_sink_ids": "[68,75,180,187,324,331,436,443]",
        "attn_weight_sink_weight": "1.0",
    }
    for name, value in expected.items():
        actual = getattr(args, "multi_frame." + name)
        if actual is not None and actual != value:
            parser.error("Unsupported active AVA baseline setting: " + name)
    layer = getattr(args, "multi_frame.temporal_feature_layer")
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    else:
        if not args.pretrained_checkpoint:
            parser.error("A verified checkpoint input is required")
        config = {
            "model": args.model,
            "seed": args.seed,
            "checkpoints": {"policy": args.pretrained_checkpoint},
            "evaluation": {
                "task_suites": [args.task_suite_name],
                "tasks_per_suite": None,
                "rollouts_per_task": args.num_trials_per_task,
                "open_loop_steps": args.num_open_loop_steps,
                "max_steps": args.max_steps,
                "center_crop": args.center_crop != "False",
                "temporal_feature_layer": int(layer) if layer is not None else -2,
                "assets": "hfasset://libero_plus_assets",
                "expected_task_counts": {args.task_suite_name: 10},
                "benchmark_revision": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
            },
        }
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    from vlade.evaluation.rollouts import run

    run(config, args.local_log_dir, args.workspace or args.local_log_dir / "runtime")
    return 0
