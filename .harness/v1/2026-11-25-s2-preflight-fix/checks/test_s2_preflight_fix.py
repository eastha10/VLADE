"""Regressions for serialized T1 defaults and failed-preflight source transition."""
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))
from vlade.distillation import representation_config as rc
from vlade.research import distillation, representation_recovery as recovery
from vlade.utils import representation_kd_notebook as notebook

OLD, NEW = "a" * 40, "b" * 40


def test_pinned_t1_diff_config_restores_actual_llama_defaults(tmp_path):
    report = json.loads((Path(__file__).resolve().parents[1] / "reports/t1-config.json").read_text())
    raw = {**report["identifiers"], "text_config": report["text_config"]}
    assert "hidden_size" not in raw["text_config"] and "num_hidden_layers" not in raw["text_config"]
    (tmp_path / "config.json").write_text(json.dumps(raw))
    result = rc.representation_contract(tmp_path, student_dim=1024, student_layer=-2,
                                        teacher_layer=-2, teacher_python=Path(sys.executable))
    assert result["teacher_dim"] == 4096 and result["tokens"] == 56
    with pytest.raises(ValueError, match="outside"):
        rc.representation_contract(tmp_path, student_dim=1024, student_layer=-2, teacher_layer=-34)


@pytest.mark.parametrize("change", [
    {"llm_backbone_id": "vicuna-v15-7b"}, {"model_type": "qwen3_5"},
    {"text_config": None}, {"text_config": {} , "llm_backbone_id": None},
    {"text_config": {"model_type": "qwen3_5_text"}},
    *[{"text_config": {name: value}} for name in ("hidden_size", "num_hidden_layers")
      for value in (None, True, 0, -1, "4096", 4096.0)],
])
def test_invalid_explicit_config_is_not_repaired(tmp_path, change, monkeypatch):
    config = {"llm_backbone_id": "llama2-7b-pure", "text_config": {}}
    config.update(change)
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **kw: pytest.fail("invalid raw config reached runtime"))
    with pytest.raises(ValueError, match="Llama2"):
        rc.representation_contract(tmp_path, student_dim=1024, student_layer=-2, teacher_layer=-2)


def test_existing_ava_interpreter_and_local_only_loader_are_used(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"llm_backbone_id": "llama2-7b-pure",
        "text_config": {"model_type": "llama"}}))
    calls = []
    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(stdout='{"hidden_size":16,"num_hidden_layers":2}')
    monkeypatch.setattr(rc.subprocess, "run", run)
    value = rc.representation_contract(tmp_path, student_dim=32, student_layer=-2,
                                       teacher_layer=-2, teacher_python=Path("existing-ava-python"))
    assert value["teacher_dim"] == 16
    command, options = calls[0]
    assert command[0] == "existing-ava-python" and "local_files_only=True" in command[3]
    assert "from_pretrained" in command[3] and "trust_remote_code" not in command[3]
    assert options["timeout"] == 60 and options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_trainer_preflight_accepts_pinned_t1_diff_structure(tmp_path, monkeypatch, capsys):
    from vlade.training import qwen_vla_task as trainer
    report = json.loads((Path(__file__).resolve().parents[1] / "reports/t1-config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**report["identifiers"], "text_config": report["text_config"]}))
    for name in ("manifest.json", "checkpoint_manifest.json"):
        (tmp_path / name).write_text("{}")
    _, variants = distillation.load_settings(ROOT / "configs/experiments/qwen_vla_representation_kd.json", {})
    settings = variants["gt_kd"]
    settings["distillation"]["teacher_python"] = sys.executable
    monkeypatch.setattr(trainer, "source_identity", lambda *a, **kw: {"git_commit": NEW})
    monkeypatch.setattr(trainer, "resolve_input", lambda *a: tmp_path)
    monkeypatch.setattr(trainer, "EpisodeData", lambda *a: SimpleNamespace(root=tmp_path, manifest={"counts": {}}))
    monkeypatch.setattr(trainer, "verified_s0", lambda *a: 800)
    monkeypatch.setattr(trainer, "teacher_identity", lambda *a: (tmp_path, {"step": 1975, "temporal_feature_layer": -2}))
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    output = tmp_path / "not-created"
    trainer.run(settings, output, preflight=True)
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "preflight_ok" and result["representation"]["teacher_dim"] == 4096
    assert result["representation"]["student_dim"] == 1024 and not output.exists()
    assert result["teacher_plan_max_new_tokens"] == 0 and result["gpu_forward_tested"] is False


@pytest.fixture
def failed_run(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs/experiments/qwen_vla_representation_kd.json").read_text())
    config["max_steps"] = 125  # Preserve saved effective conditions, not fresh defaults.
    workflow, variants = distillation.load_settings(config, {})
    folder = tmp_path / "runs/failed-s2"
    folder.mkdir(parents=True)
    contract = {"workflow": workflow, "variants": variants, "run_group": "failed-s2", "git_commit": OLD}
    for name, value in (("representation-config.json", config), ("gt_kd.json", variants["gt_kd"]),
                        ("inputs.json", {"assets": {"T1": {"revision": "c" * 40}}}),
                        ("run-contract.json", contract)):
        (folder / name).write_text(json.dumps(value, indent=2) + "\n")
    monkeypatch.setattr(recovery, "source_identity", lambda *a, **kw: {"git_commit": NEW})
    monkeypatch.setattr(distillation, "prepare_inputs", lambda *a: folder / "inputs.json")
    import huggingface_hub
    remote = SimpleNamespace(private=True, siblings=[])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=lambda *a: remote))
    return tmp_path, folder, contract, remote


def test_preflight_transition_preserves_settings_pins_and_original_bytes(failed_run):
    work, folder, contract, _ = failed_run
    original = {p.name: p.read_bytes() for p in folder.iterdir()}
    recovery.recover_preflight_source(ROOT, work, "failed-s2", OLD)
    assert json.loads((folder / "run-contract.json").read_text()) == {**contract, "git_commit": NEW}
    assert (folder / f"preflight-source-{OLD}.json").read_bytes() == original["run-contract.json"]
    for name in ("inputs.json", "gt_kd.json", "representation-config.json"):
        assert (folder / name).read_bytes() == original[name]
    recovery.recover_preflight_source(ROOT, work, "failed-s2", OLD)


@pytest.mark.parametrize("extra", ["S2", "metrics.jsonl", "unknown.txt", "step-0000001", "hub_pending.json"])
def test_any_local_training_or_unknown_output_blocks_source_change(failed_run, extra):
    work, folder, _, _ = failed_run
    before = (folder / "run-contract.json").read_bytes()
    (folder / extra).mkdir() if extra in {"S2", "step-0000001"} else (folder / extra).write_text("{}")
    with pytest.raises(ValueError, match="local output"):
        recovery.recover_preflight_source(ROOT, work, "failed-s2", OLD)
    assert (folder / "run-contract.json").read_bytes() == before
    assert not (folder / f"preflight-source-{OLD}.json").exists()


def test_remote_run_payload_blocks_source_change(failed_run):
    work, folder, _, remote = failed_run
    remote.siblings = [SimpleNamespace(rfilename="runs/failed-s2/S2/last.json")]
    with pytest.raises(ValueError, match="remote output"):
        recovery.recover_preflight_source(ROOT, work, "failed-s2", OLD)
    assert json.loads((folder / "run-contract.json").read_text())["git_commit"] == OLD


def test_changed_input_registry_blocks_source_change(failed_run, monkeypatch):
    work, folder, _, _ = failed_run
    def changed(*a):
        raise FileExistsError("Input selection changed")
    monkeypatch.setattr(distillation, "prepare_inputs", changed)
    with pytest.raises(FileExistsError, match="Input selection"):
        recovery.recover_preflight_source(ROOT, work, "failed-s2", OLD)
    assert not (folder / f"preflight-source-{OLD}.json").exists()


def test_changed_effective_settings_blocks_source_change(failed_run):
    work, folder, _, _ = failed_run
    path = folder / "representation-config.json"
    value = json.loads(path.read_text()); value["max_steps"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="settings differ"):
        recovery.recover_preflight_source(ROOT, work, "failed-s2", OLD)


def test_new_client_keeps_saved_run_conditions_and_recovery_arguments(failed_run, monkeypatch):
    work, folder, _, _ = failed_run
    from vlade.runtime import environment
    from vlade.utils import source_identity
    monkeypatch.setattr(source_identity, "source_identity", lambda *a, **kw: {"git_commit": NEW})
    located, transitions, runs = [], [], []
    monkeypatch.setattr(environment, "locate", lambda code, work, name: located.append(name))
    monkeypatch.setattr(recovery, "recover_preflight_source", lambda *args: transitions.append(args))
    monkeypatch.setattr(distillation, "main", lambda args: runs.append(args))
    script = next(n.value.value for n in ast.parse(notebook.RUN_CELL).body
                  if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "CLIENT" for x in n.targets))
    monkeypatch.setattr(sys, "argv", ["CLIENT", str(ROOT), str(work), NEW, "failed-s2",
                                     "preflight", "", "1.0", "{}", OLD])
    exec(script, {})
    assert located == ["student", "ava"]
    assert transitions == [(ROOT, work, "failed-s2", OLD)]
    assert json.loads((folder / "representation-config.json").read_text())["max_steps"] == 125
    assert runs[0][runs[0].index("--action") + 1] == "preflight"


def test_recovery_cell_replaces_old_client_and_retains_kernel_source(tmp_path, monkeypatch):
    """Execute the complete standalone cell with fake paths/processes, no install/GPU."""
    real_path = Path
    workspace = tmp_path / "work"
    folder = workspace / "runs/s2-repr-20261125T081316Z-88f7798e"
    folder.mkdir(parents=True)
    config = json.loads((ROOT / "configs/experiments/qwen_vla_representation_kd.json").read_text())
    (folder / "representation-config.json").write_text(json.dumps(config))
    pins = {key: {"revision": revision} for key, revision in {
        "S0": "380f8fb1bd828d42e7f396d6979d699b0b7d0cff", "T1": "1fdc672416f2129624b22022e6977345a3b429e3",
        "prepared_d_seed42": "6d2542577647a9e291554524af4e1dd0928e9df6"}.items()}
    (folder / "inputs.json").write_text(json.dumps({"assets": pins}))
    code = tmp_path / "sources" / NEW
    nbpath = code / "experiments/colab_s2_representation_kd_existing_session.ipynb"
    nbpath.parent.mkdir(parents=True)
    nbpath.write_text(json.dumps({"cells": [{"id": "run", "source": ["CLIENT='new-client'\nS2_RECOVER_PREFLIGHT_FROM=''\n"]}]}))
    def mapped(value):
        if str(value) == "/content/vlade_distillation": return workspace
        if str(value) == "/content/vlade_sources": return code.parent
        return real_path(value)
    monkeypatch.setattr(notebook.__dict__.get("subprocess", __import__("subprocess")), "check_output",
                        lambda *a, **kw: NEW + "\n")
    scope = {"CODE": "kernel-original-code", "SOURCE_COMMIT": OLD, "CLIENT": "old-client", "Path": mapped}
    script = notebook.RECOVERY_CELL.replace("from pathlib import Path", "")
    exec(script.replace("S2_FIX_COMMIT = ''", f"S2_FIX_COMMIT = '{NEW}'"), scope)
    assert scope["CLIENT"] == "new-client" and scope["S2_SOURCE_COMMIT"] == NEW
    assert scope["S2_CODE"] == code and scope["S2_WORKSPACE"] == workspace
    assert scope["CODE"] == "kernel-original-code" and scope["SOURCE_COMMIT"] == OLD
    assert scope["ACTION"] == "preflight"


def test_notebook_generation_and_code_cells(tmp_path):
    target = notebook.write_notebook(tmp_path / "generated.ipynb")
    assert target.read_bytes() == (ROOT / "experiments/colab_s2_representation_kd_existing_session.ipynb").read_bytes()
    data = json.loads(target.read_text(encoding="utf-8"))
    import nbformat
    nbformat.validate(nbformat.reads(target.read_text(encoding="utf-8"), as_version=4))
    for item in data["cells"]:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), item["id"], "exec")
            assert item["outputs"] == []
    client = next(n.value.value for n in ast.parse(notebook.RUN_CELL).body
                  if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "CLIENT" for x in n.targets))
    compile(client, "CLIENT", "exec")
