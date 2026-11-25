"""Offline regression checks for the actual annotation and Teacher Plan clients."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))

from vlade.data import teacher_plan_annotations as labels
from vlade.research import teacher_plan_training as workflow
from vlade.storage.assets import sha256
from vlade.training.teacher_plan_supervision import PlanAnnotations

CONFIG = ROOT / "configs/experiments/teacher_plan_training.json"


def fake_data(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    manifest = {"split_sha256": "a" * 64, "normalization_sha256": "b" * 64}
    (root / "manifest.json").write_text(json.dumps(manifest))
    folder = root / "episodes/train"
    queries = [{"episode_id": "train", "timestep": step, "instruction": "Move cup.", "folder": folder,
                "primary": f"primary-{step}.png", "wrist": f"wrist-{step}.png",
                "proprio": SimpleNamespace(tolist=lambda: [0.] * 8), "gt_action": "future action excluded"}
               for step in (0, 8)]
    def episodes(split):
        assert split == "train", "Annotation export accessed held-out examples"
        yield {"episode_id": "train", "queries": 2}, queries
    return SimpleNamespace(root=root, manifest=manifest, episodes=episodes,
                           entries=[{"episode_id": "train", "queries": 2, "split": "train"}])


def reviewed(tmp_path, monkeypatch):
    data = fake_data(tmp_path)
    monkeypatch.setattr(labels, "_data", lambda reference: data)
    draft = tmp_path / "draft"
    labels.export_template("hfasset://prepared_d_seed42", draft)
    rows = [json.loads(line) for line in (draft / "plans.jsonl").read_text().splitlines()]
    for row in rows:
        row["plan"] = "Move toward the cup with the gripper open."
    (draft / "plans.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return data, draft, rows


def test_unregistered_external_data_rejected_before_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises((ValueError, FileNotFoundError)):
        labels.export_template(str(tmp_path / "unregistered"), output)
    assert not output.exists()


def test_template_has_no_fabricated_plan_or_future_observations(tmp_path, monkeypatch):
    data = fake_data(tmp_path)
    calls = []
    monkeypatch.setattr(labels, "_data", lambda reference: calls.append(reference) or data)
    output = tmp_path / "draft"
    result = labels.export_template("hfasset://prepared_d_seed42", output)
    assert result["queries"] == 2 and calls == ["hfasset://prepared_d_seed42"]
    rows = [json.loads(line) for line in (output / "plans.jsonl").read_text().splitlines()]
    assert [row["plan"] for row in rows] == ["", ""]
    observations = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert set(observations[0]) == {"episode_id", "timestep", "instruction", "data", "views", "proprio"}
    assert observations[0]["views"]["primary"] == "episodes/train/primary-0.png"
    with pytest.raises(ValueError):
        labels.validate_reviewed("hfasset://prepared_d_seed42", output, "human_reviewed")
    with pytest.raises(FileExistsError):
        labels.export_template("hfasset://prepared_d_seed42", output)


@pytest.mark.parametrize("damage", ["duplicate", "missing", "heldout", "extra", "instruction", "empty", "identity", "review"])
def test_bad_or_leaking_reviewed_labels_are_rejected(tmp_path, monkeypatch, damage):
    data, draft, rows = reviewed(tmp_path, monkeypatch)
    review = "human_reviewed"
    if damage == "duplicate": rows.append(rows[0])
    if damage == "missing": rows.pop()
    if damage == "heldout": rows[0]["split"] = "validation"
    if damage == "extra": rows.append({**rows[0], "timestep": 16})
    if damage == "instruction": rows[0]["instruction"] = "Different task"
    if damage == "empty": rows[0]["plan"] = ""
    if damage == "identity": data.manifest["split_sha256"] = "c" * 64
    if damage == "review": review = "unreviewed"
    (draft / "plans.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        labels.validate_reviewed("hfasset://prepared_d_seed42", draft, review)


def test_reviewed_package_connects_to_existing_plan_ce_loader(tmp_path, monkeypatch):
    data, draft, rows = reviewed(tmp_path, monkeypatch)
    manifest = labels.validate_reviewed("hfasset://prepared_d_seed42", draft, "externally_reviewed")
    assert json.loads((draft / "manifest.json").read_text())["annotation_source"] == "unreviewed"
    package = tmp_path / "hf-payload"
    package.mkdir()
    (package / "manifest.json").write_text(json.dumps(manifest))
    (package / "plans.jsonl").write_bytes((draft / "plans.jsonl").read_bytes())
    monkeypatch.setattr("vlade.storage.assets.resolve_input_record", lambda reference: (package, {
        "format": "directory", "revision": "c" * 40, "repo_id": "VLADE-konkuk/libero-plus-prepared",
        "files": {"manifest.json": {}, "plans.jsonl": {}}}))
    annotations = PlanAnnotations("hfasset://teacher_plans_d_seed42", data)
    assert annotations.target(rows[1]) == rows[1]["plan"]
    assert annotations.identity["annotations_revision"] == "c" * 40


def test_publication_uses_private_existing_uploader_without_review_assets(tmp_path, monkeypatch):
    _, draft, _ = reviewed(tmp_path, monkeypatch)
    captured = []
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: object())
    monkeypatch.setattr("vlade.storage.migrate.published_registry", lambda api: {"assets": {}})
    def publish(api, key, sources, **kwargs):
        captured.append((key, sources, kwargs))
        assert json.loads(sources["manifest.json"].read_text())["annotation_source"] == "human_reviewed"
        return {"assets": {key: {"revision": "c" * 40}}}
    monkeypatch.setattr("vlade.storage.migrate.publish_files", publish)
    labels.publish_reviewed("hfasset://prepared_d_seed42", draft, "human_reviewed", "teacher_plans_test")
    key, sources, options = captured[0]
    assert set(sources) == {"manifest.json", "plans.jsonl"}
    assert options["repo"] == "VLADE-konkuk/libero-plus-prepared" and options["kind"] == "dataset"


def test_publication_rejects_existing_different_key_before_write(tmp_path, monkeypatch):
    _, draft, _ = reviewed(tmp_path, monkeypatch)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: object())
    monkeypatch.setattr("vlade.storage.migrate.published_registry", lambda api: {"assets": {"teacher_plans_test": {}}})
    monkeypatch.setattr("vlade.storage.migrate.publish_files", lambda *args, **kwargs: pytest.fail("remote write"))
    with pytest.raises(ValueError, match="different content"):
        labels.publish_reviewed("hfasset://prepared_d_seed42", draft, "human_reviewed", "teacher_plans_test")


@pytest.mark.parametrize("override", [{"plan_loss_weight": 0}, {"plan_loss_weight": float("nan")},
    {"registry_revision": "main"}, {"initial_checkpoint": "local/T1"}, {"train.max_steps": 0},
    {"train.warmup_steps": 1000}, {"train.save_steps": 1001}, {"train.learning_rate": 0}, {"train.unknown": 1}])
def test_invalid_training_settings_are_rejected(override):
    with pytest.raises((ValueError, TypeError)):
        workflow.load_settings(CONFIG, override)


def test_new_supervised_run_initializes_from_t1_and_preserves_fft_contract():
    settings = workflow.load_settings(CONFIG)
    base = json.loads((ROOT / settings["base_config"]).read_text())
    main = workflow.build_fft_config(settings, Path("T1_plan"), pilot=False)
    pilot = workflow.build_fft_config(settings, Path("T1_plan_pilot"), pilot=True)
    assert main["fft_runs"]["ava_vla"]["checkpoint"] == "hfasset://T1"
    for key in ("sequence_length", "detach_every", "sequence_sampling", "temporal_feature_layer"):
        assert main["fft_runs"]["ava_vla"][key] == base["fft_runs"]["ava_vla"][key]
    assert main["fft_defaults"]["optimizer"] == "paged_adamw_8bit"
    assert main["fft_defaults"]["save_optimizer_state"] is True
    assert main["fft_defaults"]["plan_loss_weight"] == 1.
    assert "resume" not in main["fft_defaults"]
    assert pilot["fft_defaults"]["max_steps"] == 1 and pilot["fft_defaults"]["warmup_steps"] == 0
    assert main["fft_defaults"]["max_steps"] == 1000


def test_cli_plan_has_no_workspace_or_hf_side_effects(tmp_path, monkeypatch, capsys):
    work = tmp_path / "unused"
    monkeypatch.setattr(workflow, "prepare_inputs", lambda *args, **kwargs: pytest.fail("input resolution"))
    monkeypatch.setattr(workflow, "prepare", lambda *args: pytest.fail("install"))
    workflow.main(["--config", str(CONFIG), "--workspace", str(work), "--run-group", "offline", "--action", "plan"])
    assert not work.exists()
    value = json.loads(capsys.readouterr().out)
    assert value["pilot"]["hub_run"] == "offline-pilot/T1_plan_pilot"
    assert value["main"]["hub_run"] == "offline/T1_plan"


def test_missing_annotations_fail_before_model_download(tmp_path, monkeypatch):
    settings = workflow.load_settings(CONFIG)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "assets": {}}))
    monkeypatch.setattr("vlade.research.registry.prepare_registry", lambda *args: path)
    with pytest.raises(ValueError, match="Reviewed Plan annotations"):
        workflow.prepare_inputs(settings, tmp_path, tmp_path / "run", annotations=True)
    assert not (tmp_path / "run").exists()


def test_notebook_is_configuration_only_and_opt_in(tmp_path):
    from vlade.utils.teacher_plan_training_notebook import write_notebook
    output = write_notebook(tmp_path / "client.ipynb")
    assert output.read_bytes() == (ROOT / "experiments/colab_teacher_plan_training.ipynb").read_bytes()
    notebook = json.loads(output.read_text(encoding="utf-8"))
    source = "".join("".join(c["source"]) for c in notebook["cells"])
    for flag in ("RUN_SETUP", "EXPORT_TEMPLATE", "PUBLISH_REVIEWED", "RUN_PREFLIGHT", "RUN_PILOT", "RUN_TRAIN", "EXPORT_TEACHER_PIN"):
        assert flag + " = False" in source
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), c["id"], "exec")
            assert c["outputs"] == [] and c["execution_count"] is None
    assert "loss.backward" not in source and "create_commit" not in source
    assert "SOURCE_COMMIT" in source and "새 RUN_GROUP" in source


def setup_worker(tmp_path, monkeypatch):
    from vlade.training import fft
    settings = workflow.load_settings(CONFIG)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": 1, "assets": {"T1": {
        "repo_id": "VLADE-konkuk/VLADE-checkpoints", "repo_type": "model", "revision": "a" * 40}}}))
    monkeypatch.setattr(workflow, "source_identity", lambda *args, **kwargs: {"git_commit": "b" * 40})
    monkeypatch.setattr(workflow, "prepare_inputs", lambda *args, **kwargs: registry)
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda reference: tmp_path / str(reference).split("/")[-1])
    annotations = tmp_path / "teacher_plans_d_seed42"
    annotations.mkdir()
    (annotations / "manifest.json").write_text('{"annotation_source":"human_reviewed"}')
    monkeypatch.setattr(labels, "validate_reviewed", lambda *args: {})
    monkeypatch.setattr("vlade.data.episode_data.EpisodeData", lambda *args: object())
    monkeypatch.setattr("vlade.training.teacher_inference.teacher_identity", lambda *args: (tmp_path, {"step": 1975, "plan_supervision": None}))
    calls = []
    monkeypatch.setattr(fft, "preflight", lambda cfg: calls.append(("preflight", cfg)))
    monkeypatch.setattr("vlade.training.hub.HubCheckpoints", lambda *args, **kwargs: object())
    monkeypatch.setattr(fft, "train", lambda cfg: calls.append(("train", cfg)))
    monkeypatch.setattr(workflow, "export_pin", lambda *args: calls.append(("pin", args)))
    return settings, calls


def test_pilot_and_main_do_not_resume_each_other(tmp_path, monkeypatch):
    settings, calls = setup_worker(tmp_path, monkeypatch)
    run = tmp_path / "runs/test"
    workflow._worker(settings, tmp_path, run, "test", "pilot", None, None)
    pilot_cfg = next(item[1] for item in calls if item[0] == "train")
    assert pilot_cfg.max_steps == 1 and pilot_cfg.resume is None
    checkpoint = run / "T1_plan_pilot/step-0000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("ok")
    calls.clear()
    workflow._worker(settings, tmp_path, run, "test", "train", None, None)
    main_cfg = next(item[1] for item in calls if item[0] == "train")
    assert main_cfg.max_steps == 1000 and main_cfg.resume is None
    assert str(main_cfg.checkpoint).replace("\\", "/").endswith("T1")
    assert calls[0] == ("pin", (run / "T1_plan_pilot", "test-pilot", 1))


def test_main_requires_pilot_and_changed_contract_is_rejected(tmp_path, monkeypatch):
    settings, _ = setup_worker(tmp_path, monkeypatch)
    run = tmp_path / "runs/test"
    with pytest.raises(RuntimeError, match="separate one-step pilot"):
        workflow._worker(settings, tmp_path, run, "test", "train", None, None)
    changed = copy.deepcopy(settings)
    changed["plan_loss_weight"] = .5
    with pytest.raises(ValueError, match="Settings or source changed"):
        workflow._worker(changed, tmp_path, run, "test", "preflight", None, None)


def test_remote_main_restore_does_not_require_recreating_pilot(tmp_path, monkeypatch):
    settings, calls = setup_worker(tmp_path, monkeypatch)
    run = tmp_path / "runs/test"
    def download(repo, group, output, **kwargs):
        assert group == "test/T1_plan" and kwargs["revision"] == "c" * 40
        (output / "step-0000100").mkdir(parents=True)
    monkeypatch.setattr("vlade.training.hub.download_checkpoint", download)
    workflow._worker(settings, tmp_path, run, "test", "train", "c" * 40, None)
    cfg = next(item[1] for item in calls if item[0] == "train")
    assert cfg.resume == run / "T1_plan/step-0000100"
    assert not (run / "T1_plan_pilot").exists()


def test_final_handoff_checks_local_seal_remote_identity_and_pending(tmp_path, monkeypatch):
    from vlade.training.run_control import seal_checkpoint
    output = tmp_path / "T1_plan"
    checkpoint = output / "step-0000002"
    checkpoint.mkdir(parents=True)
    contract = {"prompt_version": "ava-approach-v1", "objective": "plan_ce", "trained_steps": 2,
                "weight": 1., "annotations_manifest_sha256": "a" * 64, "annotations_sha256": "b" * 64,
                "annotations_revision": "c" * 40, "annotations_repo_id": "VLADE-konkuk/libero-plus-prepared"}
    (checkpoint / "vlade_fft.json").write_text(json.dumps({"step": 2, "model": "ava_vla", "plan_supervision": contract}))
    (checkpoint / "trainer_state.pt").write_text("synthetic state")
    seal_checkpoint(checkpoint, 2, 16)
    (checkpoint / "COMPLETE").write_text("ok\n")
    prefix = "runs/main/T1_plan/checkpoints/step-0000002"
    pointer = tmp_path / "remote-final.json"
    pointer.write_text(json.dumps({"step": 2, "checkpoint": prefix}))
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: SimpleNamespace(model_info=lambda *args: SimpleNamespace(private=True, sha="d" * 40)))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(pointer))
    monkeypatch.setattr("vlade.storage.checkpoint_inputs.checkpoint_record", lambda *args: {
        "files": {"checkpoint_manifest.json": {"sha256": sha256(checkpoint / "checkpoint_manifest.json")}}})
    report = workflow.export_pin(output, "main", 2)
    assert report["TEACHER_CHECKPOINT_PIN"]["revision"] == "d" * 40
    assert report["pending"] is False
    with pytest.raises(RuntimeError, match="Target step is not complete"):
        workflow.export_pin(output, "main", 3)
    (output / "hub_pending.json").write_text('{"queue":[{}]}')
    with pytest.raises(RuntimeError, match="pending"):
        workflow.export_pin(output, "main", 2)
    (output / "hub_pending.json").write_text('{"queue":[]}')
    (checkpoint / "trainer_state.pt").write_text("tampered state")
    with pytest.raises(ValueError, match="integrity"):
        workflow.export_pin(output, "main", 2)
