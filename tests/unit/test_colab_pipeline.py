"""Real small episode preparation, split isolation, cache resume and tamper checks."""
import io
import json
import sys
import zipfile
from itertools import islice
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import vlade.data.episode_data as episode_data_module
from vlade.data.colab_assets import safe_members, sha256
from vlade.data.episode_data import EpisodeData, group_split, prepare
from vlade.training.fixed_cache import generate, validate_selection
from vlade.training.student_task import samples
from vlade.data.episode_data import query_stream, query_count, write_json
from vlade.training.run_control import (epoch_budget, overfit_report, require_reports,
    gradient_diagnostics, validate_teacher_report, planned_checkpoint_space)
from vlade.training.debug_records import emit_checkpoint_debug


def episode(index):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (index % 256, 10, 20)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    return [{"action": np.array([index + t / 10] * 6 + [-1 if t < 7 else 1], dtype=np.float32),
             "observation": {"image": raw, "wrist_image": raw,
                             "state": np.arange(8, dtype=np.float32) + index + t},
             "language_instruction": f"move item {index}".encode()}
            for t in range(10)]


@pytest.fixture
def data(tmp_path):
    # Duplicate content is deliberately included, irrespective of episode identity.
    prepare(iter([episode(0), episode(0)] + [episode(i) for i in range(1, 50)]), tmp_path / "data")
    return EpisodeData(tmp_path / "data")


def test_group_split_and_train_only_stats(data):
    entries = data.entries
    assert entries[0]["split"] == entries[1]["split"]
    assert entries[0]["content_sha256"] == entries[1]["content_sha256"]
    groups = {}
    for entry in entries:
        groups.setdefault(entry["content_sha256"], set()).add(entry["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    values = []
    for entry in entries:
        if entry["split"] == "train":
            with np.load(data.root / "episodes" / entry["episode_id"] / "trajectory.npz") as raw:
                values.append(raw["actions"][:, :6])
    expected = np.quantile(np.concatenate(values), .99, axis=0)
    np.testing.assert_allclose(data.stats["action"]["q99"][:6], expected)


def test_causal_history_tail_mask_and_gripper(data):
    previous_id = None
    seen = 0
    for sample, history in samples(data, "train"):
        assert history[-1]["timestep"] == sample["timestep"]
        assert all(q["episode_id"] == sample["episode_id"] for q in history)
        if previous_id != sample["episode_id"]:
            assert sample["timestep"] == 0 and len(history) == 1
        else:
            assert sample["timestep"] == 8 and len(history) == 2
            assert sample["action_valid_mask"].sum() == 14
            assert not sample["action_valid_mask"][2:].any()
            assert np.all(sample["gt_action"][:, 6] == 0)
        previous_id = sample["episode_id"]
        seen += 1
    assert seen > 0


def test_modified_image_is_rejected(data):
    _, queries = next(data.episodes("train"))
    sample = queries[0]
    (sample["folder"] / sample["primary"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Image integrity"):
        data.image(sample, "primary")


class FakeSequentialTeacher:
    def __init__(self):
        self.resets = 0
        self.time = None

    def reset(self):
        self.resets += 1
        self.time = None

    def predict(self, sample, data):
        assert sample["timestep"] == (0 if self.time is None else self.time + 8)
        self.time = sample["timestep"]
        return torch.from_numpy(sample["gt_action"] * .5), torch.full((56, 4), float(self.time))


def test_cache_seals_replays_and_resumes_without_mixing(data, tmp_path):
    selection = {"checkpoint": "selected-T1", "checkpoint_sha256": "a" * 64, "teacher_step": 12}
    teacher = FakeSequentialTeacher()
    result = generate(data, teacher, selection, tmp_path / "cache", hidden=True, max_episodes=2)
    assert result["queries"] == 4 and result["sample_replay_max_abs_error"] == 0
    assert teacher.resets == 4
    teacher = FakeSequentialTeacher()
    resumed = generate(data, teacher, selection, tmp_path / "cache", hidden=True, max_episodes=2)
    assert resumed["queries"] == 4 and teacher.resets == 2
    assert len(resumed["replayed_episodes"]) == 2
    with pytest.raises(ValueError, match="different Teacher"):
        generate(data, teacher, {**selection, "teacher_step": 13}, tmp_path / "cache", hidden=True, max_episodes=2)


def test_t0_cannot_be_used_as_selected_t1(data, tmp_path):
    with pytest.raises(ValueError, match="T1 must"):
        validate_selection({"checkpoint": str(tmp_path)}, data)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/drive"])
def test_archive_rejects_path_escape(name):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, "x")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive, pytest.raises(ValueError, match="Unsafe"):
        safe_members(archive)


def test_archive_rejects_literal_backslash():
    # Windows ZipFile.writestr normalizes backslashes; exercise a foreign member directly.
    item = zipfile.ZipInfo("safe")
    item.filename = "folder\\escape"
    class ForeignArchive:
        def infolist(self):
            return [item]
    with pytest.raises(ValueError, match="Unsafe"):
        safe_members(ForeignArchive())


def test_data_rejects_nonfinite_and_never_publishes_complete(tmp_path):
    steps = episode(0)
    steps[0]["action"][0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        prepare(iter([steps]), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    assert not list((tmp_path / "bad.incomplete").rglob("COMPLETE"))


def test_prepare_resumes_verified_legacy_episode_and_rewrites_damaged_episode(tmp_path, monkeypatch):
    output = tmp_path / "data"

    def interrupted():
        yield episode(0)
        yield episode(1)
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare(interrupted(), output)
    stage = tmp_path / "data.incomplete"
    assert not (stage / "COMPLETE").exists()
    checkpoints = stage / ".checkpoints"
    (checkpoints / "episode-0000000.json").unlink()
    damaged = stage / "episodes/episode-0000001/primary-0000000.png"
    damaged.write_bytes(b"damaged")

    rewritten = []
    original_write = episode_data_module._write_episode

    def observe_write(folder, checkpoint, expected, actions, proprio, queries, payloads):
        rewritten.append(expected["episode_id"])
        return original_write(folder, checkpoint, expected, actions, proprio, queries, payloads)

    monkeypatch.setattr(episode_data_module, "_write_episode", observe_write)
    manifest = prepare(iter([episode(0), episode(1)] + [episode(i) for i in range(2, 51)]),
                       output, resume=True)

    assert "episode-0000000" not in rewritten
    assert "episode-0000001" in rewritten
    assert len(manifest["episodes"]) == 51
    assert (output / "COMPLETE").is_file()
    assert not (tmp_path / "data.incomplete").exists()
    EpisodeData(output)


def test_prepare_resume_rejects_a_different_source_contract(tmp_path):
    output = tmp_path / "data"

    def interrupted():
        yield episode(0)
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        prepare(interrupted(), output, source={"revision": "one"})
    with pytest.raises(ValueError, match="different source or canonical contract"):
        prepare(iter([episode(i) for i in range(51)]), output, source={"revision": "two"}, resume=True)
    assert not list((tmp_path / "data.incomplete").rglob("COMPLETE"))


def keys(stream):
    return [(sample["episode_id"], sample["timestep"]) for sample, _ in stream]


def test_epoch_budget_covers_every_query_and_rounding(data):
    budget = epoch_budget(data, 2, batch_size=1, accumulation=8)
    assert budget["query_presentations"] >= query_count(data) * 2
    assert 0 <= budget["rounding_extra_queries"] < 8
    for epochs in (0, -1, 1.5):
        with pytest.raises(ValueError):
            epoch_budget(data, epochs)


def test_seeded_episode_shuffle_preserves_history_and_restores_cursor(data):
    count = query_count(data)
    full = keys(islice(query_stream(data, "train", repeat=True, seed=17), 2 * count + 8))
    assert len(set(full[:count])) == count
    assert full[:count] != full[count:2 * count]
    assert full[:count] == keys(query_stream(data, "train", seed=17))
    resumed = keys(islice(query_stream(data, "train", repeat=True, seed=17, skip=count + 3), 5))
    assert resumed == full[count + 3:count + 8]
    for sample, history in query_stream(data, "train", seed=17):
        assert all(item["episode_id"] == sample["episode_id"] for item in history)
        assert [item["timestep"] for item in history] == ([0] if sample["timestep"] == 0 else [0, 8])


def test_validation_subset_spans_split_and_test_is_sealed(data):
    all_keys = keys(query_stream(data, "validation"))
    subset = keys(query_stream(data, "validation", limit=3))
    assert subset == [all_keys[i * len(all_keys) // 3] for i in range(3)]
    assert subset == keys(query_stream(data, "validation", limit=3))
    with pytest.raises(ValueError, match="sealed"):
        list(query_stream(data, "test"))


def test_atomic_json_failure_preserves_existing_report(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"status": "passed"})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"loss": float("nan")})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_overfit_gate_rejects_flat_nonfinite_and_incomplete_history(tmp_path):
    declining = [{"step": i, "loss": 2 if i < 5 else .2} for i in range(10)]
    assert overfit_report(declining)["status"] == "passed"
    assert overfit_report([{"step": i, "loss": 1.} for i in range(10)])["status"] == "failed"
    assert overfit_report(declining[:3])["status"] == "failed"
    declining[-1]["loss"] = float("inf")
    assert overfit_report(declining)["status"] == "failed"
    path = tmp_path / "failed.json"
    write_json(path, {"status": "failed"})
    with pytest.raises(ValueError, match="prerequisite"):
        require_reports([path])


def test_gradient_diagnostic_failure_prevents_optimizer_update():
    from vlade.training.loop import run_loop, make_scheduler
    used, unused = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    original = used.weight.detach().clone()
    optimizer = torch.optim.SGD([*used.parameters(), *unused.parameters()], lr=.1)
    with pytest.raises(RuntimeError, match="unused"):
        run_loop(modules=[used, unused], optimizer=optimizer, scheduler=make_scheduler(optimizer, 1),
                 batches=[torch.ones(1, 1)], loss_fn=lambda x, _: used(x).square().mean(), max_steps=1,
                 on_before_step=lambda _: gradient_diagnostics({"used": used, "unused": unused}))
    torch.testing.assert_close(used.weight, original, rtol=0, atol=0)


def report_fixture(tmp_path):
    checkpoint = tmp_path / "T1"
    report = {"report_type": "teacher_validation_comparison", "selection_split": "validation",
              "used_final_test": False, "evaluation_protocol_sha256": "a" * 64,
              "data_split_sha256": "b" * 64,
              "candidates": [{"checkpoint": str(tmp_path / role), "role": role,
                 "validation_success_rate": .7 if role == "T0" else .6,
                 "validation_episodes": 10, "original_libero_success_rate": .8,
                 "original_libero_episodes": 10} for role in ("T0", "T1")]}
    return checkpoint, report


def test_t1_report_requires_measured_validation_and_original_libero(tmp_path):
    checkpoint, report = report_fixture(tmp_path)
    # Negative adaptation is valid evidence; the validator never assumes improvement.
    validate_teacher_report(report, checkpoint, "b" * 64)
    with pytest.raises(ValueError, match="final test"):
        validate_teacher_report({**report, "used_final_test": True}, checkpoint, "b" * 64)
    report["candidates"][1]["validation_success_rate"] = None
    with pytest.raises(ValueError, match="measured"):
        validate_teacher_report(report, checkpoint, "b" * 64)


def test_cache_code_change_rejected_and_replay_failure_removes_complete(data, tmp_path, monkeypatch):
    import vlade.training.fixed_cache as cache
    selection = {"checkpoint": "selected-T1", "checkpoint_sha256": "a" * 64, "teacher_step": 12}
    output = tmp_path / "cache"
    generate(data, FakeSequentialTeacher(), selection, output, max_episodes=1)
    with monkeypatch.context() as patch:
        patch.setattr(cache, "code_hashes", lambda _: {"changed": "yes"})
        with pytest.raises(ValueError, match="different Teacher"):
            generate(data, FakeSequentialTeacher(), selection, output, max_episodes=1)
    assert (output / "COMPLETE").is_file()
    class ChangedTeacher(FakeSequentialTeacher):
        def predict(self, sample, dataset):
            action, features = super().predict(sample, dataset)
            return action + 1, features
    with pytest.raises(ValueError, match="replay"):
        generate(data, ChangedTeacher(), selection, output, max_episodes=1)
    assert not (output / "COMPLETE").exists()


def test_post_checkpoint_debug_is_bounded_not_kd_and_restores_modes_rng(data, tmp_path):
    from vlade.data.teacher_records import load_teacher_records
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"explicit_unit_test_weights")
    cfg = SimpleNamespace(output_dir=tmp_path / "run", prepared_data_root=data.root, debug_records=1)
    _, queries = next(data.episodes("train"))
    sample = queries[0]
    batch = {"sample_metadata": [sample], "action_valid_mask": torch.from_numpy(sample["action_valid_mask"])[None]}
    module = torch.nn.Linear(1, 1).train()
    rng = torch.get_rng_state().clone()
    def predict(capture):
        torch.rand(1)
        capture({"predicted_action": torch.zeros(1, 8, 7), "gt_action": torch.from_numpy(sample["gt_action"])[None],
                 "proprio": torch.from_numpy(sample["proprio"])[None], "gt_token_ids": torch.arange(56)[None]})
    emit_checkpoint_debug(cfg, checkpoint, 1, [module], batch, predict)
    emit_checkpoint_debug(cfg, checkpoint, 2, [module], batch, predict)
    manifests = list((cfg.output_dir / "training_debug").glob("*/manifest.json"))
    assert len(manifests) == 1
    assert len(list(load_teacher_records(manifests[0].parent, for_kd=False))) == 1
    with pytest.raises(ValueError, match="not a KD"):
        list(load_teacher_records(manifests[0].parent))
    assert module.training
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_planned_checkpoint_space_counts_only_future_saves(tmp_path, monkeypatch):
    import vlade.training.run_control as control
    reference = tmp_path / "pilot"
    reference.mkdir()
    (reference / "weights").write_bytes(b"x" * 100)
    monkeypatch.setattr(control.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    report = planned_checkpoint_space(tmp_path, reference, max_steps=10, save_steps=3, after_step=3)
    assert report["remaining_checkpoints"] == 3  # 6,9,10
    monkeypatch.setattr(control.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="planned checkpoints"):
        planned_checkpoint_space(tmp_path, reference, 10, 3)


def notebook_cell(filename, cell_id):
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    book = json.loads(path.read_text(encoding="utf-8"))
    return "".join(next(cell["source"] for cell in book["cells"] if cell["id"] == cell_id))


def test_notebook_run_all_does_not_launch_main_training(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Main training must remain opt-in")
    env = {"completed_checkpoints": lambda _: [], "OUTPUT": tmp_path, "RUN_MAIN_FFT": False,
           "run": forbidden}
    exec(notebook_cell("colab_02_ava_fft_and_fixed_cache.ipynb", "fft-main"), env)
    env = {"student_config": lambda _: (tmp_path / "config.json", tmp_path), "RUN_MAIN_STUDENT": False,
           "train_student": forbidden, "json": json}
    exec(notebook_cell("colab_03_qwen_task_only.ipynb", "student-main"), env)


def test_notebook_failed_overfit_cannot_launch_main_training(tmp_path):
    smoke, overfit = tmp_path / "smoke", tmp_path / "overfit"
    write_json(smoke / "reload_report.json", {"status": "passed", "context": {}})
    write_json(overfit / "overfit_report.json", {"status": "failed"})
    env = {"student_config": lambda _: (tmp_path / "config.json", tmp_path), "RUN_MAIN_STUDENT": True,
           "smoke_output": smoke, "overfit_output": overfit, "require_reports": require_reports,
           "smoke_config": tmp_path / "smoke.json", "overfit_config": tmp_path / "overfit.json",
           "PREPARED": tmp_path / "data", "OVERFIT_MAX_RATIO": .8, "pilot_context": lambda *a, **k: {}, "json": json,
           "train_student": lambda *_: pytest.fail("Failed pilot cannot train")}
    write_json(env["overfit_config"], {"max_steps": 50})
    with pytest.raises(ValueError, match="prerequisite"):
        exec(notebook_cell("colab_03_qwen_task_only.ipynb", "student-main"), env)
