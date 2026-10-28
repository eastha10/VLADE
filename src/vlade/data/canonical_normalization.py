"""Compute exact train-only statistics for the canonical LIBERO+ split."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

from vlade.data.canonical_split import (CANONICAL_SHA256, assert_observed_assignments,
    load_canonical_split)
from vlade.data.episode_data import CONVENTION, write_json


def tfrecord_records(path: Path) -> Iterator[bytes]:
    with path.open("rb") as stream:
        while length_bytes := stream.read(8):
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated TFRecord length in {path.name}")
            length = struct.unpack("<Q", length_bytes)[0]
            if len(stream.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord length CRC in {path.name}")
            payload = stream.read(length)
            if len(payload) != length or len(stream.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord payload in {path.name}")
            yield payload


def statistics(values: np.ndarray, mask: list[bool]) -> dict:
    return {
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "mask": mask,
    }


def compute(data_dir: Path, manifest_path: Path, output: Path,
            expected_sha256: str = CANONICAL_SHA256) -> dict:
    from tfrecord import example_pb2

    canonical = load_canonical_split(manifest_path, expected_sha256)
    shards = sorted(data_dir.glob("libero_mix-train.tfrecord-*-of-01024"))
    expected_names = [f"libero_mix-train.tfrecord-{index:05d}-of-01024" for index in range(1024)]
    if [path.name for path in shards] != expected_names:
        raise FileNotFoundError("Canonical normalization requires all 1,024 TFRecord shards")
    train_actions: list[np.ndarray] = []
    train_proprio: list[np.ndarray] = []
    observed: list[tuple[str, bool, str]] = []
    total_episodes = total_transitions = train_transitions = 0
    for shard_number, shard in enumerate(shards, 1):
        for payload in tfrecord_records(shard):
            example = example_pb2.Example()
            example.ParseFromString(payload)
            feature = example.features.feature
            action = np.asarray(feature["steps/action"].float_list.value, dtype=np.float32).reshape(-1, 7)
            state = np.asarray(feature["steps/observation/state"].float_list.value, dtype=np.float32).reshape(-1, 8)
            if len(action) != len(state) or not np.isfinite(action).all() or not np.isfinite(state).all():
                raise ValueError(f"Invalid trajectory in {shard.name}")
            split, is_calibration, group_id = canonical.assignment(action, state)
            observed.append((split, is_calibration, group_id))
            total_episodes += 1
            total_transitions += len(action)
            if split == "train":
                converted = action.copy()
                if not np.allclose(np.abs(converted[:, 6]), 1.0, atol=1e-5):
                    raise ValueError(f"Unexpected gripper convention in {shard.name}")
                converted[:, 6] = np.where(converted[:, 6] < 0, 1.0, 0.0)
                train_actions.append(converted)
                train_proprio.append(state.copy())
                train_transitions += len(action)
        if shard_number % 64 == 0:
            print(f"normalization_shards={shard_number}/1024 episodes={total_episodes}", flush=True)
    assert_observed_assignments(observed, canonical)
    if total_episodes != 14_347 or total_transitions != 2_238_036 or train_transitions != 1_791_269:
        raise ValueError("Canonical dataset/transition totals do not match the selected source")
    action_values = np.concatenate(train_actions, axis=0)
    proprio_values = np.concatenate(train_proprio, axis=0)
    result = {
        "schema_version": 1,
        "action": statistics(action_values, [True] * 6 + [False]),
        "proprio": statistics(proprio_values, [True] * 8),
        "num_transitions": train_transitions,
        "num_trajectories": 11_477,
        "normalization_scope": "canonical_train_only",
        "normalization_type": "bounds_q99",
        "convention": CONVENTION,
        "canonical_manifest_sha256": canonical.sha256,
        "candidate": "D_80_10_10",
        "seed": 42,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-sha256", default=CANONICAL_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compute(args.data_dir, args.canonical_manifest, args.output, args.canonical_sha256)
    print(json.dumps({key: result[key] for key in ("num_trajectories", "num_transitions",
                                                    "canonical_manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
