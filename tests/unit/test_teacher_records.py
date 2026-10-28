"""Storage contract, capacity, provenance and temporal integrity checks."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.data.teacher_records import TeacherProvenance, TeacherRecordWriter, load_teacher_records


def provenance():
    return TeacherProvenance("fixed_kd", "T1-test", "a" * 64, "dataset-rev", "b" * 64,
                             "c" * 64, "teacher-tokenizer", "tokenizer-rev",
                             "normalized-libero-delta-gripper", "episode_context", 8, True, False)


def record(time=0, end=True):
    return {"episode_id": "episode-1", "timestep": time, "teacher_step": 500,
            "episode_start": time == 0, "episode_end": end,
            "instruction": "pick the bowl", "observation_refs": {"primary": f"ep1/{time}/primary", "wrist": f"ep1/{time}/wrist"},
            "checkpoint_sha256": "a" * 64, "normalization_sha256": "c" * 64,
            "proprio": torch.randn(8), "gt_action": torch.randn(8, 7),
            "predicted_action": torch.randn(8, 7, requires_grad=True),
            "action_valid_mask": torch.ones(8, 7, dtype=torch.bool),
            "gt_token_ids": torch.arange(56), "tokenizer_id": "teacher-tokenizer", "tokenizer_revision": "tokenizer-rev",
            "action_hidden_states": torch.randn(56, 4, requires_grad=True),
            "hidden_layer": "last_decoder", "hidden_layout": "horizon_major_action_dimension_minor"}


def test_round_trip_separates_targets_and_detaches(tmp_path):
    first, second = record(end=False), record(8)
    with TeacherRecordWriter(tmp_path / "cache", provenance()) as writer:
        assert writer.write(first)
        assert writer.write(second)
    loaded = list(load_teacher_records(tmp_path / "cache", expected_provenance=provenance()))
    for key in ("proprio", "gt_action", "predicted_action", "gt_token_ids", "action_hidden_states", "action_valid_mask"):
        torch.testing.assert_close(loaded[0][key], first[key])
        assert not loaded[0][key].requires_grad
    assert len(loaded) == 2


@pytest.mark.parametrize("change", [{"quantized": True}, {"frozen": False}, {"temporal_mode": "zero_context"}])
def test_base_kd_provenance_rejects_invalid_teacher(tmp_path, change):
    with pytest.raises(ValueError):
        TeacherRecordWriter(tmp_path / "cache", replace(provenance(), **change))


@pytest.mark.parametrize("key,value", [
    ("predicted_action", torch.zeros(7, 8)), ("gt_action", torch.full((8, 7), float("nan"))),
    ("normalization_sha256", "d" * 64), ("checkpoint_sha256", "d" * 64),
    ("tokenizer_revision", "different"), ("gt_token_ids", torch.arange(56).float()),
    ("action_hidden_states", torch.zeros(8, 7, 4)), ("action_valid_mask", torch.zeros(8, 7, dtype=torch.bool)),
])
def test_bad_record_not_sealed(tmp_path, key, value):
    bad = record()
    bad[key] = value
    with pytest.raises(ValueError):
        with TeacherRecordWriter(tmp_path / "cache", provenance()) as writer:
            writer.write(bad)
    assert not (tmp_path / "cache/manifest.json").exists()


@pytest.mark.parametrize("change", [{"timestep": 16}, {"teacher_step": 501}, {"episode_id": "another"}, {"episode_start": True}])
def test_kd_rejects_temporal_or_step_mix(tmp_path, change):
    with pytest.raises(ValueError):
        with TeacherRecordWriter(tmp_path / "cache", provenance()) as writer:
            writer.write(record(end=False))
            writer.write({**record(8), **change})


def test_debug_sampling_budget_and_kd_guard(tmp_path):
    with TeacherRecordWriter(tmp_path / "debug", replace(provenance(), purpose="training_debug", frozen=False),
                             every_n=2, max_records=2) as writer:
        assert [writer.write(record()) for _ in range(5)] == [True, False, True, False, False]
    with pytest.raises(ValueError, match="debug"):
        list(load_teacher_records(tmp_path / "debug"))
    assert len(list(load_teacher_records(tmp_path / "debug", for_kd=False))) == 2


def test_capacity_never_silently_truncates_kd(tmp_path):
    with pytest.raises(ValueError, match="capacity"):
        with TeacherRecordWriter(tmp_path / "cache", provenance(), max_bytes=10) as writer:
            writer.write(record())
    assert not (tmp_path / "cache/manifest.json").exists()


def test_incomplete_and_tampered_stream_rejected(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        with TeacherRecordWriter(tmp_path / "incomplete", provenance()) as writer:
            writer.write(record(end=False))
    with TeacherRecordWriter(tmp_path / "cache", provenance()) as writer:
        writer.write(record())
    path = tmp_path / "cache/records.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["instruction"] = "modified"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        list(load_teacher_records(tmp_path / "cache"))


def test_reopened_episode_and_overwrite_rejected(tmp_path):
    with pytest.raises(ValueError, match="reappear"):
        with TeacherRecordWriter(tmp_path / "cache", provenance()) as writer:
            writer.write(record())
            writer.write(record())
    with pytest.raises(FileExistsError):
        TeacherRecordWriter(tmp_path / "cache", provenance())


def test_caught_write_error_cannot_publish_a_partial_cache(tmp_path):
    writer = TeacherRecordWriter(tmp_path / "cache", provenance(), max_records=1)
    writer.write(record())
    with pytest.raises(ValueError, match="capacity"):
        writer.write({**record(), "episode_id": "ep2"})
    with pytest.raises(ValueError, match="rejected write"):
        writer.close()
    assert not (tmp_path / "cache/manifest.json").exists()
