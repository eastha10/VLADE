"""Canonical D/seed-42 split, normalization and shared FFT configuration checks."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlade.data.canonical_split import (CANONICAL_CALIBRATION_EPISODES, CANONICAL_COUNTS,
    CANONICAL_GROUPS, CANONICAL_SHA256, file_sha256, load_canonical_split)
from vlade.data.colab_assets import PART_METADATA
from vlade.training.fft import FFTConfig
from vlade.training.prepared_fft import loader_split


MANIFEST = ROOT / "experiments/manifests/libero_plus_rlds/canonical_d_seed42.jsonl.gz"
NORMALIZATION = ROOT / "experiments/manifests/libero_plus_rlds/normalization_train_d_seed42.json"
NORMALIZATION_SHA256 = "eac95bcc81b6182b90a2bd030096df6e3e408cb88923d0a1d57dc6c81e297c67"


def test_canonical_candidate_identity_counts_and_no_group_leakage():
    canonical = load_canonical_split(MANIFEST)
    assert canonical.sha256 == CANONICAL_SHA256
    assert len(canonical.episode_ids) == sum(CANONICAL_COUNTS.values()) == 14_347
    assert len(canonical.group_assignments) == CANONICAL_GROUPS
    assert {episode_id for values in canonical.group_episode_ids.values() for episode_id in values} == canonical.episode_ids
    counts = Counter()
    calibration = 0
    for group_id, (split, is_calibration) in canonical.group_assignments.items():
        weight = canonical.group_episode_counts[group_id]
        counts[split] += weight
        calibration += weight * is_calibration
    assert dict(counts) == CANONICAL_COUNTS
    assert calibration == CANONICAL_CALIBRATION_EPISODES


def test_canonical_manifest_hash_is_enforced_before_parsing(tmp_path):
    damaged = tmp_path / "canonical.jsonl.gz"
    damaged.write_bytes(MANIFEST.read_bytes() + b"damage")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_canonical_split(damaged)


def test_canonical_normalization_is_train_only_and_bound_to_manifest():
    normalization = json.loads(NORMALIZATION.read_text(encoding="utf-8"))
    normalized = NORMALIZATION.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized).hexdigest() == NORMALIZATION_SHA256
    assert normalization["canonical_manifest_sha256"] == CANONICAL_SHA256
    assert normalization["normalization_scope"] == "canonical_train_only"
    assert normalization["num_trajectories"] == CANONICAL_COUNTS["train"]
    assert normalization["num_transitions"] == 1_791_269


def test_ava_and_oft_share_canonical_manifest_normalization_and_prepared_data():
    config = ROOT / "configs/experiments/libero_plus_fft.json"
    ava = FFTConfig.from_json(config, "ava_vla")
    oft = FFTConfig.from_json(config, "openvla_oft")
    for field in ("prepared_data_root", "canonical_split_manifest", "canonical_split_sha256",
                  "canonical_normalization", "canonical_normalization_sha256"):
        assert getattr(ava, field) == getattr(oft, field)
    assert ava.canonical_split_sha256 == CANONICAL_SHA256
    assert ava.canonical_normalization_sha256 == NORMALIZATION_SHA256
    assert not ava.image_aug and not oft.image_aug


def test_fft_loader_cannot_select_validation_or_test_for_training():
    assert loader_split(train=True) == "train"
    assert loader_split(train=False) == "validation"
    assert "test" not in {loader_split(train=True), loader_split(train=False)}


def test_pinned_dataset_parts_have_explicit_size_and_sha256():
    assert sum(size for size, _ in PART_METADATA.values()) == 75_544_257_300
    assert all(size > 0 and len(digest) == 64 for size, digest in PART_METADATA.values())
