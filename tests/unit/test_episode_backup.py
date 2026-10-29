"""Small interruption, archive damage, restore and resume coverage."""
from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import vlade.data.episode_data as episode_data_module
from vlade.data.episode_backup import (EpisodeArchiveBackup, file_sha256,
    import_legacy_incomplete, restore_episode_archives)
from vlade.data.episode_data import EpisodeData, prepare


def episode(index: int) -> list[dict]:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (index + 1, 20, 30)).save(buffer, format="PNG")
    image = buffer.getvalue()
    return [
        {"action": np.array([index + t / 10] * 6 + [-1 + t], dtype=np.float32),
         "observation": {"image": image, "wrist_image": image,
                         "state": np.arange(8, dtype=np.float32) + index + t},
         "language_instruction": f"fixture {index}".encode()}
        for t in range(2)
    ]


def install_splitter(monkeypatch):
    state = {"index": 0}

    def split(_content, _seed=42):
        value = ("train", "validation", "test")[state["index"] % 3]
        state["index"] += 1
        return value

    monkeypatch.setattr(episode_data_module, "group_split", split)
    return state


def test_periodic_backup_damage_restore_legacy_and_resume(tmp_path, monkeypatch):
    split_state = install_splitter(monkeypatch)
    output = tmp_path / "local/data"
    backup_dir = tmp_path / "drive/episode-backups"
    backup = EpisodeArchiveBackup(backup_dir, every=2, min_free_gib=0)

    def interrupted():
        yield episode(0)
        yield episode(1)
        yield episode(2)
        raise RuntimeError("runtime terminated")

    with pytest.raises(RuntimeError, match="runtime terminated"):
        prepare(interrupted(), output, source={"fixture": 1}, backup=backup)

    stage = output.with_name(output.name + ".incomplete")
    metadata_paths = list(backup_dir.glob("episodes-*.json"))
    archive_paths = list(backup_dir.glob("episodes-*.tar"))
    assert len(metadata_paths) == len(archive_paths) == 1
    assert not list(backup_dir.glob("*.partial"))
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    assert metadata["episode_ids"] == ["episode-0000000", "episode-0000001"]
    assert file_sha256(archive_paths[0]) == metadata["archive_sha256"]

    damaged_backup = tmp_path / "drive/damaged"
    shutil.copytree(backup_dir, damaged_backup)
    damaged_archive = next(damaged_backup.glob("*.tar"))
    damaged_archive.write_bytes(damaged_archive.read_bytes() + b"damage")
    damaged_stage = tmp_path / "damaged/data.incomplete"
    contract = json.loads((stage / "resume.json").read_text(encoding="utf-8"))
    damaged_report = restore_episode_archives(damaged_backup, damaged_stage, contract)
    assert damaged_report["restored_episodes"] == 0
    assert len(damaged_report["skipped_archives"]) == 1
    assert not list((damaged_stage / ".checkpoints").glob("*.json"))

    legacy_drive = tmp_path / "drive/libero_plus_episode80.incomplete"
    legacy_episode = legacy_drive / "episodes/episode-0000002"
    shutil.copytree(stage / "episodes/episode-0000002", legacy_episode)
    shutil.rmtree(stage)
    stage.mkdir(parents=True)
    assert import_legacy_incomplete(legacy_drive, stage) == 1
    restore_report = restore_episode_archives(backup_dir, stage, contract)
    assert restore_report["restored_episodes"] == 2
    assert not (stage / ".checkpoints/episode-0000002.json").exists()

    written = []
    original_write = episode_data_module._write_episode

    def observe_write(folder, checkpoint, expected, actions, proprio, queries, payloads):
        written.append(expected["episode_id"])
        return original_write(folder, checkpoint, expected, actions, proprio, queries, payloads)

    monkeypatch.setattr(episode_data_module, "_write_episode", observe_write)
    split_state["index"] = 0
    resumed_backup = EpisodeArchiveBackup(backup_dir, every=2, min_free_gib=0)
    manifest = prepare(iter(episode(index) for index in range(6)), output,
                       source={"fixture": 1}, resume=True, backup=resumed_backup)

    assert written == ["episode-0000003", "episode-0000004", "episode-0000005"]
    assert manifest["counts"] == {"train": 2, "validation": 2, "test": 2}
    assert (output / "COMPLETE").is_file()
    assert len(list(backup_dir.glob("episodes-*.json"))) == 3
    EpisodeData(output)


def test_backup_stops_before_writing_when_free_space_is_below_threshold(tmp_path):
    stage = tmp_path / "data.incomplete"
    stage.mkdir()
    backup = EpisodeArchiveBackup(tmp_path / "drive", every=1, min_free_gib=10**9)
    with pytest.raises(OSError, match="free space"):
        backup.check_space(stage)
