"""Checkpoint/pilot identity and recoverable extraction regression checks."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.data.episode_data import write_json
from vlade.data.colab_assets import DATA_REPO, DATA_REV, PARTS, OWNER, extract, sha256
from vlade.training.run_control import (seal_checkpoint, verify_checkpoint, pilot_context,
                                        require_reports, configure_tensorflow_cpu)
from vlade.utils.filesystem import publish_directory


def checkpoint(path, step):
    path.mkdir(parents=True)
    (path / "model.bin").write_bytes(f"model-step-{step}".encode())
    (path / "trainer_state.pt").write_bytes(f"optimizer-step-{step}".encode())
    seal_checkpoint(path, step, step * 8)
    (path / "COMPLETE").write_text("ok")
    return path


def test_checkpoint_rejects_optimizer_from_another_step(tmp_path):
    first = checkpoint(tmp_path / "step-0000001", 1)
    second = checkpoint(tmp_path / "step-0000002", 2)
    verify_checkpoint(first, accumulation=8)
    (first / "trainer_state.pt").write_bytes((second / "trainer_state.pt").read_bytes())
    with pytest.raises(ValueError, match="integrity"):
        verify_checkpoint(first, accumulation=8)


@pytest.mark.parametrize("change", ["model", "extra", "label", "microsteps"])
def test_checkpoint_seal_covers_files_step_and_boundary(tmp_path, change):
    path = checkpoint(tmp_path / "step-0000001", 1)
    if change == "model":
        (path / "model.bin").write_bytes(b"wrong")
    elif change == "extra":
        (path / "unknown.bin").write_bytes(b"unexpected")
    elif change == "label":
        moved = tmp_path / "step-0000002"
        path.rename(moved)
        path = moved
    else:
        with pytest.raises(ValueError, match="microstep"):
            verify_checkpoint(path, accumulation=4)
        return
    with pytest.raises(ValueError):
        verify_checkpoint(path)


def test_pilot_reports_are_bound_to_data_code_config_and_checkpoint(tmp_path):
    cp = checkpoint(tmp_path / "step-0000001", 1)
    data = tmp_path / "data"
    write_json(data / "manifest.json", {"data": "one"})
    config = tmp_path / "pilot.json"
    write_json(config, {"lr": 1})
    expected = pilot_context(config, cp, data, "student")
    report = tmp_path / "passed.json"
    write_json(report, {"status": "passed", "context": expected})
    require_reports([report], [expected])
    for key in ("code_sha256", "data_manifest_sha256", "configuration_sha256", "checkpoint_manifest_sha256"):
        changed = {**expected, key: "wrong"}
        with pytest.raises(ValueError, match="context mismatch"):
            require_reports([report], [changed])
    write_json(config, {"lr": 2})
    with pytest.raises(ValueError, match="context mismatch"):
        require_reports([report], [pilot_context(config, cp, data, "student")])


def test_tensorflow_is_disabled_before_upstream_use(monkeypatch):
    calls = []
    tf = ModuleType("tensorflow")
    tf.config = SimpleNamespace(set_visible_devices=lambda devices, kind: calls.append((devices, kind)))
    tf.random = SimpleNamespace(set_seed=lambda value: calls.append(value))
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    configure_tensorflow_cpu(17)
    assert calls == [([], "GPU"), 17]


def test_directory_publish_retries_transient_lock_and_never_overwrites(tmp_path, monkeypatch):
    stage, destination = tmp_path / ".stage", tmp_path / "published"
    stage.mkdir()
    original = Path.rename
    calls = []
    def locked(path, target):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError("temporary scanner lock")
        return original(path, target)
    monkeypatch.setattr(Path, "rename", locked)
    publish_directory(stage, destination)
    assert destination.is_dir() and len(calls) == 3
    stage.mkdir()
    with pytest.raises(FileExistsError):
        publish_directory(stage, destination)


def extraction_fixture(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    for name in PARTS:
        (parts / name).write_bytes(b"small synthetic part")
    write_json(parts / "download_manifest.json", {"repository": DATA_REPO, "revision": DATA_REV,
        "files": [{"path": name, "sha256": sha256(parts / name)} for name in PARTS]})
    return parts, tmp_path / "output", tmp_path / "scratch"


def test_extraction_restarts_after_interruption_and_preserves_unowned_files(tmp_path, monkeypatch):
    import vlade.data.colab_assets as assets
    parts, output, scratch = extraction_fixture(tmp_path)
    scratch.mkdir()
    legacy = scratch / "libero_plus_combined.zip"
    legacy.write_bytes(b"unowned previous file")
    legacy_stage = output.with_name(output.name + ".incomplete")
    legacy_stage.mkdir()
    (legacy_stage / "user-note.txt").write_text("preserve")
    monkeypatch.setattr(assets.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    calls = []
    def join(command, check):
        calls.append(command)
        combined = Path(command[-1])
        if len(calls) == 1:
            combined.write_bytes(b"interrupted")
            raise KeyboardInterrupt()
        with zipfile.ZipFile(combined, "w") as archive:
            archive.writestr("libero_mix/1.0.0/dataset_info.json", "{}")
            archive.writestr("libero_mix/1.0.0/data.tfrecord-00000", b"unit-test-not-real-RLDS")
    monkeypatch.setattr(assets.subprocess, "run", join)
    with pytest.raises(KeyboardInterrupt):
        extract(parts, output, scratch)
    assert not output.exists()
    assert not list(scratch.glob("vlade-libero-*.incomplete"))
    extract(parts, output, scratch)
    assert (output / "COMPLETE").is_file()
    assert legacy.read_bytes() == b"unowned previous file"
    assert (legacy_stage / "user-note.txt").read_text() == "preserve"


def test_notebook_rejects_changed_existing_t1_selection(tmp_path):
    from test_colab_pipeline import notebook_cell
    selection = tmp_path / "selections/T1.json"
    write_json(selection, {"checkpoint": str(tmp_path / "old"), "validation_report": str(tmp_path / "old-report"),
                          "selection_reason": "old", "validation_report_sha256": "a" * 64})
    code = notebook_cell("colab_02_ava_fft_and_fixed_cache.ipynb", "select-t1")
    code = code.replace("T1_CHECKPOINT = ''", "T1_CHECKPOINT = str(ROOT / 'new')")
    code = code.replace("VALIDATION_REPORT = ''", "VALIDATION_REPORT = str(ROOT / 'new-report')")
    code = code.replace("SELECTION_REASON = ''", "SELECTION_REASON = 'new'")
    with pytest.raises(ValueError, match="기존 T1 선택"):
        exec(code, {"ROOT": tmp_path, "Path": Path, "json": json, "hashlib": hashlib,
                    "run": lambda *_: pytest.fail("Do not silently reuse/overwrite selection")})
