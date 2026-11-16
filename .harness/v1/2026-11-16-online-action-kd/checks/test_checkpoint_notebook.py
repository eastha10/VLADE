"""HF metadata pinning, cached-input boundaries and dedicated notebook flow."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))

from vlade.storage.assets import REGISTRY_REPO, fetch_asset, resolve_input
from vlade.storage.checkpoint_inputs import checkpoint_record, pin_checkpoint_inputs


@pytest.fixture
def remote(tmp_path, monkeypatch):
    import huggingface_hub
    revision = "a" * 40
    prefix = "runs/teacher/output/checkpoints/step-0000004"
    payloads = {"weights.pt": b"weights", "identity.json": b"{}"}
    seal = {"schema_version": 1, "step": 4, "microsteps": 8, "files": {
        name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} for name, value in payloads.items()}}
    payloads["checkpoint_manifest.json"] = json.dumps(seal).encode()
    payloads["COMPLETE"] = b"ok\n"
    inventory = {prefix + "/" + name: SimpleNamespace(rfilename=prefix + "/" + name, size=len(value),
        lfs=SimpleNamespace(sha256=hashlib.sha256(value).hexdigest()) if name.endswith(".pt") else None)
        for name, value in payloads.items()}
    for name, value in payloads.items():
        (tmp_path / name).write_bytes(value)
    calls = []
    def download(repo, name, *, revision, **kwargs):
        assert repo == REGISTRY_REPO and revision == "a" * 40
        calls.append(name)
        return str(tmp_path / name[len(prefix) + 1:])
    info = SimpleNamespace(private=True, sha=revision, siblings=list(inventory.values()))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=lambda *a, **k: info))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return SimpleNamespace(revision=revision, prefix=prefix, info=info, files=inventory, calls=calls,
        selection={"repo_id": REGISTRY_REPO, "revision": revision, "prefix": prefix})


def test_pinning_only_reads_metadata_then_inputs_fetch_hashes_and_reject_tamper(remote, tmp_path, monkeypatch):
    output = tmp_path / "registry.json"
    registry = pin_checkpoint_inputs({"schema_version": 1, "assets": {}}, {"t1": remote.selection}, output)
    assert all(not name.endswith("weights.pt") for name in remote.calls)
    assert registry["assets"]["t1"]["format"] == "hub_checkpoint"
    monkeypatch.setenv("VLADE_ASSET_REGISTRY", str(output))
    monkeypatch.setenv("VLADE_ASSET_CACHE", str(tmp_path / "cache"))
    root = resolve_input("hfasset://t1")
    assert (root / "weights.pt").read_bytes() == b"weights"
    assert resolve_input(root) == root
    (root / "weights.pt").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        resolve_input(root)


@pytest.mark.parametrize("damage", ["revision", "repo", "private", "hash", "extra", "missing"])
def test_pin_rejects_wrong_origin_revision_or_remote_seal(remote, damage):
    spec = dict(remote.selection)
    if damage == "revision":
        spec["revision"] = "main"
    elif damage == "repo":
        spec["repo_id"] = "personal/other"
    elif damage == "private":
        remote.info.private = False
    elif damage == "hash":
        remote.files[remote.prefix + "/weights.pt"].lfs.sha256 = "b" * 64
    elif damage == "extra":
        remote.info.siblings.append(SimpleNamespace(rfilename=remote.prefix + "/extra.pt", size=1, lfs=None))
    else:
        remote.info.siblings = [item for item in remote.info.siblings if not item.rfilename.endswith("COMPLETE")]
    with pytest.raises(ValueError):
        checkpoint_record(**{("prefix" if k == "prefix" else k): v for k, v in spec.items()})


def notebook_cells():
    notebook = json.loads((ROOT / "experiments/colab_action_distillation.ipynb").read_text(encoding="utf-8"))
    return notebook, {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def test_notebook_defaults_never_connect_install_download_or_train(tmp_path):
    notebook, cells = notebook_cells()
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    ns = {"CODE": ROOT, "sys": sys, "Path": Path}
    with patch("vlade.research.client.job") as job, redirect_stdout(io.StringIO()):
        exec(cells["config"], ns)
        ns["WORKSPACE"] = str(tmp_path / "not-created")
        for name in ("client", "connect-inputs", "plan", "setup", "hub-check", "preflight", "pilot", "train", "download-resume"):
            exec(cells[name], ns)
    assert [call.kwargs["action"] for call in job.call_args_list] == ["plan"]
    assert not (tmp_path / "not-created").exists()
    ns["_CONNECTED"] = copy.deepcopy(ns["input_selection"]())
    ns["REGISTRY_PATH"] = tmp_path / "runtime-registry.json"
    ns["LAMBDA_KD"] = 2.
    with pytest.raises(RuntimeError, match="계획"):
        ns["require_current_inputs"]()


def test_notebook_pilot_and_main_keep_same_group_with_distinct_phase(tmp_path, monkeypatch):
    from vlade.research.cli import main
    _, cells = notebook_cells()
    ns = {"CODE": ROOT, "sys": sys, "Path": Path}
    work = tmp_path / "work"
    with patch("vlade.research.client.job"), redirect_stdout(io.StringIO()):
        exec(cells["config"], ns)
        ns["WORKSPACE"] = str(work)
        exec(cells["plan"], ns)
    # Exercise actual worker dispatch/config persistence without installing or training.
    monkeypatch.setattr("vlade.utils.source_identity.source_identity", lambda *a, **kw: {"git_commit": "b" * 40})
    monkeypatch.setattr("vlade.runtime.environment.runtime_environment", lambda *a: {})
    monkeypatch.setattr("vlade.research.execution.dispatch", lambda *a: None)
    args = ["--config", str(ROOT / "configs/research/qwen_kd.json"), "--workspace", str(work),
            "--action", "run", "--worker", "--set", 'run_group="test-student-phases"']
    main([*args, "--set", "pilot.only=true"])
    main([*args, "--set", "pilot.only=false"])
    main([*args, "--set", "pilot.only=false", "--set", 'resume="step-0000001"'])
    directory = work / "runs/test-student-phases"
    assert (directory / "student-pilot-settings.json").is_file()
    assert (directory / "student-main-settings.json").is_file()
    with pytest.raises(ValueError, match="differs"):
        main([*args, "--set", "train.learning_rate=0.001"])
