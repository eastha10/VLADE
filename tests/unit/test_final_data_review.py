"""Reject coherently rehashed but semantically inconsistent prepared datasets."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.data.colab_assets import sha256
from vlade.data.episode_data import EpisodeData, prepare, query_count, write_json
from test_colab_pipeline import episode


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    prepare(iter([episode(0), episode(0)] + [episode(i) for i in range(1, 50)]), root)
    return root


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reseal(root, manifest):
    """Update hashes too, so failures must come from the semantic invariant."""
    keys = ("episode_id", "content_sha256", "split", "steps", "queries")
    write_json(root / "splits.json", [{key: row[key] for key in keys} for row in manifest["episodes"]])
    manifest["split_sha256"] = sha256(root / "splits.json")
    stats = read(root / "normalization.json")
    stats["split_sha256"] = manifest["split_sha256"]
    write_json(root / "normalization.json", stats)
    manifest["normalization_sha256"] = sha256(root / "normalization.json")
    write_json(root / "manifest.json", manifest)


def first_training_entry(manifest):
    return next(row for row in manifest["episodes"] if row["split"] == "train")


def rewrite_queries(root, manifest, entry, queries):
    path = root / "episodes" / entry["episode_id"] / "queries.json"
    write_json(path, queries)
    entry["files"]["queries.json"] = sha256(path)
    write_json(root / "manifest.json", manifest)


def test_original_prepared_data_still_has_all_training_queries(prepared):
    data = EpisodeData(prepared)
    actual = sum(len(queries) for _, queries in data.episodes("train"))
    assert actual == query_count(data) > 0


def test_manifest_split_edit_cannot_bypass_the_sealed_split(prepared):
    manifest = read(prepared / "manifest.json")
    entry = first_training_entry(manifest)
    entry["split"] = "validation"
    write_json(prepared / "manifest.json", manifest)
    with pytest.raises(ValueError, match="sealed split manifest"):
        EpisodeData(prepared)


def test_manifest_counts_must_describe_the_actual_entries(prepared):
    manifest = read(prepared / "manifest.json")
    manifest["counts"]["train"] += 1
    write_json(prepared / "manifest.json", manifest)
    with pytest.raises(ValueError, match="split counts"):
        EpisodeData(prepared)


def test_normalization_must_bind_to_the_same_split_even_after_rehash(prepared):
    stats = read(prepared / "normalization.json")
    stats["split_sha256"] = "a" * 64
    write_json(prepared / "normalization.json", stats)
    manifest = read(prepared / "manifest.json")
    manifest["normalization_sha256"] = sha256(prepared / "normalization.json")
    write_json(prepared / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Normalization.*bound"):
        EpisodeData(prepared)


def test_rehashed_query_count_must_match_step_count(prepared):
    manifest = read(prepared / "manifest.json")
    manifest["episodes"][0]["queries"] += 1
    reseal(prepared, manifest)
    with pytest.raises(ValueError, match="step/query count"):
        EpisodeData(prepared)


def test_rehashed_duplicate_episode_identity_is_rejected(prepared):
    manifest = read(prepared / "manifest.json")
    manifest["episodes"][1]["episode_id"] = manifest["episodes"][0]["episode_id"]
    reseal(prepared, manifest)
    with pytest.raises(ValueError, match="duplicate episode identity"):
        EpisodeData(prepared)


def test_rehashed_duplicate_content_cannot_cross_splits(prepared):
    manifest = read(prepared / "manifest.json")
    first, duplicate = manifest["episodes"][:2]
    assert first["content_sha256"] == duplicate["content_sha256"]
    duplicate["split"] = "validation" if first["split"] != "validation" else "train"
    manifest["counts"] = dict(Counter(row["split"] for row in manifest["episodes"]))
    reseal(prepared, manifest)
    with pytest.raises(ValueError, match="content crosses splits"):
        EpisodeData(prepared)


@pytest.mark.parametrize("mutation", ["drop", "reverse", "duplicate_timestep"])
def test_query_length_and_order_checked_after_valid_file_hash(prepared, mutation):
    manifest = read(prepared / "manifest.json")
    entry = first_training_entry(manifest)
    queries = read(prepared / "episodes" / entry["episode_id"] / "queries.json")
    assert [row["timestep"] for row in queries] == [0, 8]
    if mutation == "drop":
        queries.pop()
    elif mutation == "reverse":
        queries.reverse()
    else:
        queries[1]["timestep"] = queries[0]["timestep"]
    rewrite_queries(prepared, manifest, entry, queries)
    data = EpisodeData(prepared)
    with pytest.raises(ValueError, match="Query length/order"):
        list(data.episodes("train", order=[entry["episode_id"]]))


@pytest.mark.parametrize("field", ["actions", "proprio"])
def test_trajectory_length_checked_after_valid_file_hash(prepared, field):
    manifest = read(prepared / "manifest.json")
    entry = first_training_entry(manifest)
    path = prepared / "episodes" / entry["episode_id"] / "trajectory.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in ("actions", "proprio")}
    arrays[field] = arrays[field][:-1]
    np.savez(path, **arrays)
    entry["files"]["trajectory.npz"] = sha256(path)
    write_json(prepared / "manifest.json", manifest)
    data = EpisodeData(prepared)
    with pytest.raises(ValueError, match="Trajectory shape/length"):
        list(data.episodes("train", order=[entry["episode_id"]]))


@pytest.mark.parametrize("reference", ["../outside.png", "missing.png"])
def test_camera_reference_checked_before_consumption(prepared, reference):
    manifest = read(prepared / "manifest.json")
    entry = first_training_entry(manifest)
    queries = read(prepared / "episodes" / entry["episode_id"] / "queries.json")
    queries[0]["primary"] = reference
    rewrite_queries(prepared, manifest, entry, queries)
    data = EpisodeData(prepared)
    with pytest.raises(ValueError, match="camera observation reference"):
        list(data.episodes("train", order=[entry["episode_id"]]))
