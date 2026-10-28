"""Bounded post-save debug actions tied to the exact saved checkpoint bytes."""
from __future__ import annotations
import uuid
import json
from pathlib import Path
from typing import Any

import torch
from vlade.data.episode_data import CONVENTION, EpisodeData
from vlade.data.teacher_records import TeacherProvenance, TeacherRecordWriter
from vlade.training.loop import capture_rng, restore_rng


def emit_checkpoint_debug(cfg: Any, checkpoint: Path, step: int, modules: list[Any],
                          batch: dict, predict: Any) -> None:
    from vlade.training.fixed_cache import bundle_hash
    root = cfg.output_dir / "training_debug"
    if len(list(root.glob("*/manifest.json"))) >= cfg.debug_records:
        return
    if cfg.prepared_data_root is None or "sample_metadata" not in batch:
        raise ValueError("Checkpoint debug needs real episode metadata from prepared data")
    data = EpisodeData(cfg.prepared_data_root)
    experiment_path = getattr(cfg, "config_path", None) or Path(__file__).resolve().parents[3] / "configs/experiments/libero_plus_fft.json"
    initial = json.loads(experiment_path.read_text(encoding="utf-8"))["initial_checkpoints"][getattr(cfg, "model", "ava_vla")]
    provenance = TeacherProvenance(
        purpose="training_debug", checkpoint_id=str(checkpoint), checkpoint_sha256=bundle_hash(checkpoint),
        dataset_revision=data.manifest["dataset_revision"], split_sha256=data.manifest["split_sha256"],
        normalization_sha256=data.manifest["normalization_sha256"], tokenizer_id=initial["repository"],
        tokenizer_revision=initial["revision"], action_convention=CONVENTION,
        temporal_mode="zero_context", query_interval_steps=8, frozen=False, quantized=False)
    modes = [(child, child.training) for module in modules for child in module.modules()]
    rng = capture_rng()
    result = {}
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            predict(result.update)
        sample = batch["sample_metadata"][0]
        episode = sample["episode_id"]
        entries = {entry["episode_id"]: entry for entry in data.entries}
        record = {"episode_id": episode, "timestep": sample["timestep"], "instruction": sample["instruction"],
                  "teacher_step": step, "episode_start": sample["timestep"] == 0,
                  "episode_end": sample["timestep"] + 8 >= entries[episode]["steps"],
                  "observation_refs": {key: f"episodes/{episode}/{sample[key]}" for key in ("primary", "wrist")},
                  "checkpoint_sha256": provenance.checkpoint_sha256,
                  "normalization_sha256": provenance.normalization_sha256,
                  "tokenizer_id": provenance.tokenizer_id, "tokenizer_revision": provenance.tokenizer_revision,
                  "action_valid_mask": batch["action_valid_mask"][0].cpu(),
                  **{key: value[0] for key, value in result.items()}}
        with TeacherRecordWriter(root / f"step-{step:07d}-{uuid.uuid4().hex[:8]}", provenance,
                                 max_records=1, max_bytes=2 * 1024**2) as writer:
            writer.write(record)
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
