"""Regressions for sealed-part substitution and final cache/report accounting.

The synthetic episodes deliberately share numerical targets, so checking only
timestep and GT action cannot distinguish a copied part from its real source.
No pretrained model, GPU, TFDS or external download is used.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.training.fixed_cache import generate
from vlade.training.run_control import validate_teacher_report
from vlade.data.episode_data import CONVENTION
from vlade.data.teacher_records import TeacherProvenance, TeacherRecordWriter


class SyntheticQueryData:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir()
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        self.manifest = {"dataset_revision": "synthetic-cache-regression",
                         "split_sha256": "b" * 64, "normalization_sha256": "c" * 64}
        self.entries = [{"episode_id": f"episode-{index}", "split": "train", "steps": 10, "queries": 2}
                        for index in range(4)]

    def episodes(self, split: str):
        assert split == "train"
        for entry in self.entries:
            queries = []
            for timestep in (0, 8):
                valid = np.ones((8, 7), dtype=bool)
                if timestep:
                    valid[2:] = False
                queries.append({"episode_id": entry["episode_id"], "timestep": timestep,
                                "instruction": f"move for {entry['episode_id']}",
                                "primary": f"primary-{timestep}.png", "wrist": f"wrist-{timestep}.png",
                                "proprio": np.zeros(8, dtype=np.float32),
                                "gt_action": np.zeros((8, 7), dtype=np.float32), "action_valid_mask": valid})
            yield entry, queries


class SyntheticSequentialTeacher:
    feature_dimension = 4

    def __init__(self):
        self.last_timestep = None

    def reset(self):
        self.last_timestep = None

    def predict(self, sample, data):
        assert sample["timestep"] == (0 if self.last_timestep is None else self.last_timestep + 8)
        self.last_timestep = sample["timestep"]
        return torch.zeros(8, 7), torch.full((56, 4), float(self.last_timestep))


def write_sealed_fixture(data, selection, output: Path, *, hidden: bool) -> None:
    """Create public-writer sealed inputs for tests specifically about cache reuse.

    Initial directory publication is covered by the original pipeline tests.
    Here the exercised generate() path must validate existing parts, including
    non-replayed parts; no publication/validation function is mocked.
    """
    output.mkdir()
    provenance = TeacherProvenance(
        purpose="fixed_kd", checkpoint_id=selection["checkpoint"], checkpoint_sha256=selection["checkpoint_sha256"],
        dataset_revision=data.manifest["dataset_revision"], split_sha256=data.manifest["split_sha256"],
        normalization_sha256=data.manifest["normalization_sha256"],
        tokenizer_id="LiAuto-DSR/avavla-libero-4in1", tokenizer_revision="090bf68d234ea13bc78476236f82458b25b797b8",
        action_convention=CONVENTION, temporal_mode="episode_context", query_interval_steps=8,
        frozen=True, quantized=False)
    teacher = SyntheticSequentialTeacher()
    for entry, queries in data.episodes("train"):
        teacher.reset()
        with TeacherRecordWriter(output / entry["episode_id"], provenance) as writer:
            for index, sample in enumerate(queries):
                action, features = teacher.predict(sample, data)
                record = {"episode_id": entry["episode_id"], "timestep": sample["timestep"],
                          "instruction": sample["instruction"], "teacher_step": selection["teacher_step"],
                          "episode_start": index == 0, "episode_end": index == len(queries) - 1,
                          "observation_refs": {camera: f"episodes/{entry['episode_id']}/{sample[camera]}"
                                               for camera in ("primary", "wrist")},
                          "checkpoint_sha256": provenance.checkpoint_sha256,
                          "normalization_sha256": provenance.normalization_sha256,
                          "proprio": torch.from_numpy(sample["proprio"]),
                          "gt_action": torch.from_numpy(sample["gt_action"]),
                          "action_valid_mask": torch.from_numpy(sample["action_valid_mask"]),
                          "predicted_action": action}
                if hidden:
                    record.update(action_hidden_states=features, hidden_layer="language_model.hidden_states[-2]",
                                  hidden_layout="horizon_major_action_dimension_minor")
                writer.write(record)


@pytest.fixture
def cache_case(tmp_path):
    data = SyntheticQueryData(tmp_path / "data")
    selection = {"checkpoint": "synthetic-selected-T1", "checkpoint_sha256": "a" * 64, "teacher_step": 3}
    output = tmp_path / "cache"
    write_sealed_fixture(data, selection, output, hidden=True)
    report = generate(data, SyntheticSequentialTeacher(), selection, output, hidden=True)
    assert "episode-2" not in report["replayed_episodes"]
    assert (output / "COMPLETE").is_file()
    return data, selection, output


def rewrite_sealed_part(part: Path, mutate) -> None:
    """Keep record/manifest integrity valid while changing the tested semantics."""
    path = part / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    raw = ("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n").encode("utf-8")
    path.write_bytes(raw)
    manifest_path = part / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), count=len(rows))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_non_replayed_part_from_other_episode_is_rejected(cache_case):
    data, selection, output = cache_case
    for filename in ("manifest.json", "records.jsonl"):
        shutil.copyfile(output / "episode-0" / filename, output / "episode-2" / filename)
    with pytest.raises(ValueError, match="target/source mismatch"):
        generate(data, SyntheticSequentialTeacher(), selection, output, hidden=True)
    assert not (output / "COMPLETE").exists()


@pytest.mark.parametrize("field", ["instruction", "proprio", "action_valid_mask", "observation_refs", "teacher_step"])
def test_every_non_replayed_part_is_checked_against_source(cache_case, field):
    data, selection, output = cache_case

    def mutate(rows):
        for row in rows:
            if field == "instruction":
                row[field] = "another valid instruction"
            elif field == "proprio":
                row[field][0] += 1.0
            elif field == "action_valid_mask":
                row[field][0][0] = False
            elif field == "observation_refs":
                row[field]["primary"] = "episodes/another/primary.png"
            else:
                row[field] += 1

    rewrite_sealed_part(output / "episode-2", mutate)
    with pytest.raises(ValueError, match="target/source mismatch"):
        generate(data, SyntheticSequentialTeacher(), selection, output, hidden=True)
    assert not (output / "COMPLETE").exists()


def test_non_replayed_action_only_part_cannot_fill_hidden_cache(cache_case, tmp_path):
    data, selection, output = cache_case
    action_only = tmp_path / "action-only"
    write_sealed_fixture(data, selection, action_only, hidden=False)
    generate(data, SyntheticSequentialTeacher(), selection, action_only, hidden=False)
    for filename in ("manifest.json", "records.jsonl"):
        shutil.copyfile(action_only / "episode-2" / filename, output / "episode-2" / filename)
    with pytest.raises(ValueError, match="feature storage mode"):
        generate(data, SyntheticSequentialTeacher(), selection, output, hidden=True)
    assert not (output / "COMPLETE").exists()


@pytest.mark.parametrize("field", ["hidden_layer", "action_hidden_states"])
def test_non_replayed_feature_layer_and_dimension_are_checked(cache_case, field):
    data, selection, output = cache_case

    def mutate(rows):
        for row in rows:
            if field == "hidden_layer":
                row[field] = "language_model.hidden_states[-1]"
            else:
                row[field] = [features[:-1] for features in row[field]]

    rewrite_sealed_part(output / "episode-2", mutate)
    with pytest.raises(ValueError, match="feature layer/shape"):
        generate(data, SyntheticSequentialTeacher(), selection, output, hidden=True)
    assert not (output / "COMPLETE").exists()


def test_incomplete_record_bytes_are_separate_from_completed_cache(cache_case):
    data, selection, output = cache_case
    for index, length in enumerate((37, 13)):
        orphan = output / f".orphan-{index}.incomplete"
        orphan.mkdir()
        (orphan / "records.jsonl").write_bytes(b"x" * length)
    report = generate(data, SyntheticSequentialTeacher(), selection, output, hidden=True)
    sealed_bytes = sum((output / entry["episode_id"] / "records.jsonl").stat().st_size for entry in data.entries)
    assert report["bytes"] == sealed_bytes
    assert report["incomplete_bytes"] == 50
    assert (output / ".orphan-0.incomplete" / "records.jsonl").is_file()
    assert (output / "COMPLETE").is_file()


def comparison_report(tmp_path):
    checkpoint = tmp_path / "selected-T1"
    common = {"validation_success_rate": .6, "validation_episodes": 10,
              "original_libero_success_rate": .7, "original_libero_episodes": 10}
    report = {"report_type": "teacher_validation_comparison", "selection_split": "validation",
              "used_final_test": False, "evaluation_protocol_sha256": "a" * 64, "data_split_sha256": "b" * 64,
              "candidates": [{**common, "checkpoint": str(tmp_path / "original-T0"), "role": "T0"},
                             {**common, "checkpoint": str(checkpoint), "role": "T1"}]}
    return checkpoint, report


def test_t0_t1_report_with_same_normalized_checkpoint_is_rejected(tmp_path):
    checkpoint, report = comparison_report(tmp_path)
    report["candidates"][0]["checkpoint"] = str(checkpoint.parent / "unused" / ".." / checkpoint.name)
    with pytest.raises(ValueError, match="distinct checkpoint"):
        validate_teacher_report(report, checkpoint, "b" * 64)


@pytest.mark.parametrize("empty", ["", "   "])
def test_t0_t1_report_with_empty_checkpoint_identity_is_rejected(tmp_path, empty):
    checkpoint, report = comparison_report(tmp_path)
    report["candidates"][0]["checkpoint"] = empty
    with pytest.raises(ValueError, match="nonempty"):
        validate_teacher_report(report, checkpoint, "b" * 64)


def test_distinct_t0_t1_comparison_report_remains_accepted(tmp_path):
    checkpoint, report = comparison_report(tmp_path)
    validate_teacher_report(report, checkpoint, "b" * 64)
