"""Registered read-only roots retain immutable HF identity and byte validation."""
import hashlib
import io
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))
from vlade.storage import assets, local_inputs


def fingerprint(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model = tmp_path / "existing-model"
    data = tmp_path / "existing-data"
    model.mkdir()
    data.mkdir()
    payload = {"COMPLETE": b"ok\n", "model/weights.bin": b"original-model"}
    for name, value in payload.items():
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    folder = data / "episodes/e1"
    folder.mkdir(parents=True)
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    files = {"trajectory.npz": b"trajectory", "queries.json": b"[]", "image.png": image.getvalue()}
    for name, value in files.items():
        (folder / name).write_bytes(value)
    manifest = {"schema_version": 2, "episodes": [{"episode_id": "e1", "split": "train", "is_calibration": True,
                "files": {name: fingerprint(value)["sha256"] for name, value in files.items()}}]}
    metadata = {"manifest.json": json.dumps(manifest).encode(), "normalization.json": b"{}", "splits.json": b"[]"}
    for name, value in metadata.items():
        (data / name).write_bytes(value)
    (data / "COMPLETE").write_text("ok\n")
    records = {
        "model": {"repo_id": "test/model", "repo_type": "model", "revision": "a" * 40, "format": "directory",
                  "prefix": "assets/model", "local_name": "model", "files": {n: fingerprint(v) for n, v in payload.items()}},
        "dataset": {"repo_id": "test/data", "repo_type": "dataset", "revision": "b" * 40, "format": "archive",
                    "archive_path": "data.tar", "archive_bytes": 1, "archive_sha256": "c" * 64, "local_name": "data",
                    "identity_files": {n: fingerprint(v) for n, v in metadata.items()}}}
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": 1, "assets": records}))
    bindings = tmp_path / "runtime/local_inputs.json"
    monkeypatch.setattr(local_inputs, "_remote_identity", lambda r: {k: r[k] for k in ("repo_id", "repo_type", "revision")})
    monkeypatch.setenv("VLADE_LOCAL_INPUTS", str(bindings))
    return model, data, registry, bindings, records


def register(setup, key):
    model, data, registry, bindings, _ = setup
    return local_inputs.register_local_input(key, model if key == "model" else data, bindings, registry=registry)


def test_registered_model_and_dataset_reuse_original_paths(setup, monkeypatch):
    model, data, registry, bindings, _ = setup
    before = {p: p.read_bytes() for root in (model, data) for p in root.rglob("*") if p.is_file()}
    register(setup, "model")
    entry = register(setup, "dataset")
    assert entry["verification"]["all_payload_files_verified"] == 3
    monkeypatch.setattr(assets, "fetch_asset", lambda *a, **kw: pytest.fail("Must not download/copy existing input"))
    for key, root in (("model", model), ("dataset", data)):
        assert assets.resolve_input("hfasset://" + key, registry=registry) == root
        assert assets.resolve_input(root, registry=registry) == root
        assert assets.resolve_path("hfasset://" + key, registry=registry) == root
    assert all(p.read_bytes() == original for p, original in before.items())
    assert not (model.parent / "verified.json").exists()
    assert bindings.is_file()


def test_unregistered_local_root_and_subdirectory_remain_rejected(setup):
    model, _, registry, _, _ = setup
    with pytest.raises(FileNotFoundError):
        assets.resolve_input(model, registry=registry)
    register(setup, "model")
    with pytest.raises(ValueError, match="External weights/data"):
        assets.resolve_input(model / "model", registry=registry)


@pytest.mark.parametrize("key,relative", [("model", "model/weights.bin"), ("dataset", "manifest.json")])
def test_mutated_registered_bytes_never_fall_back_to_download(setup, monkeypatch, key, relative):
    model, data, registry, _, _ = setup
    register(setup, key)
    ((model if key == "model" else data) / relative).write_bytes(b"modified")
    monkeypatch.setattr(assets, "fetch_asset", lambda *a, **kw: pytest.fail("No fallback on corrupted binding"))
    with pytest.raises(ValueError, match="integrity"):
        assets.resolve_input("hfasset://" + key, registry=registry)


def test_revision_change_rejected(setup):
    _, _, registry, _, records = setup
    register(setup, "model")
    records["model"]["revision"] = "d" * 40
    registry.write_text(json.dumps({"schema_version": 1, "assets": records}))
    with pytest.raises(ValueError, match="pinned registry"):
        assets.resolve_input("hfasset://model", registry=registry)


def test_registration_verifies_every_dataset_payload(setup):
    _, data, _, bindings, _ = setup
    (data / "episodes/e1/queries.json").write_text("tampered")
    with pytest.raises(ValueError, match="payload integrity"):
        register(setup, "dataset")
    assert not bindings.exists()


def test_native_image_reader_still_checks_bound_payload(setup):
    from vlade.data.episode_data import EpisodeData
    _, data, registry, _, _ = setup
    register(setup, "dataset")
    root = assets.resolve_input("hfasset://dataset", registry=registry)
    digest = json.loads((data / "manifest.json").read_text())["episodes"][0]["files"]["image.png"]
    sample = {"folder": root / "episodes/e1", "primary": "image.png", "image_hashes": {"primary": digest}}
    instance = EpisodeData.__new__(EpisodeData)
    assert instance.image(sample, "primary").size == (2, 2)
    (data / "episodes/e1/image.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Image integrity"):
        instance.image(sample, "primary")


def test_remote_identity_failure_does_not_register(setup, monkeypatch):
    _, _, _, bindings, _ = setup
    def fail(record):
        raise ValueError("remote fingerprint mismatch")
    monkeypatch.setattr(local_inputs, "_remote_identity", fail)
    with pytest.raises(ValueError, match="remote fingerprint"):
        register(setup, "model")
    assert not bindings.exists()


def test_receipt_must_stay_outside_existing_input(setup):
    model, _, registry, _, _ = setup
    with pytest.raises(ValueError, match="outside"):
        local_inputs.register_local_input("model", model, model / "receipt.json", registry=registry)
    assert not (model / "receipt.json").exists()


def test_failed_rebinding_preserves_previous_receipt(setup):
    model, _, _, bindings, _ = setup
    register(setup, "model")
    original = bindings.read_bytes()
    (model / "model/weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError):
        register(setup, "model")
    assert bindings.read_bytes() == original
    assert not bindings.with_name(bindings.name + ".lock").exists()


def test_existing_registration_lock_is_not_deleted(setup):
    _, _, _, bindings, _ = setup
    bindings.parent.mkdir()
    lock = bindings.with_name(bindings.name + ".lock")
    lock.write_text("another registration")
    with pytest.raises(FileExistsError):
        register(setup, "model")
    assert lock.read_text() == "another registration"


def test_receipt_path_or_provenance_change_is_rejected(setup):
    _, _, registry, bindings, _ = setup
    register(setup, "model")
    value = json.loads(bindings.read_text())
    value["bindings"]["model"]["remote"]["revision"] = "e" * 40
    bindings.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="provenance"):
        assets.resolve_input("hfasset://model", registry=registry)


def test_output_protection_includes_bound_roots(setup):
    from vlade.evaluation.quantization_comparison import load_config, protect_output
    model, _, _, _, _ = setup
    register(setup, "model")
    cfg = load_config(ROOT / "configs/quantization/s0_comparison.json")
    with pytest.raises(ValueError, match="outside dataset/checkpoint/HF caches"):
        protect_output(model / "new-output", cfg)
    assert not (model / "new-output").exists()


@pytest.mark.parametrize("key", ["model", "dataset"])
def test_missing_pinned_fingerprints_cannot_register(setup, key):
    _, _, registry, bindings, records = setup
    if key == "model":
        records[key]["files"] = {}
    else:
        del records[key]["identity_files"]["manifest.json"]
    registry.write_text(json.dumps({"schema_version": 1, "assets": records}))
    with pytest.raises(ValueError, match="fingerprints"):
        register(setup, key)
    assert not bindings.exists()


def test_invalid_previous_root_does_not_create_a_lock_inside_input(setup, monkeypatch):
    _, _, _, bindings, _ = setup
    register(setup, "model")
    previous = json.loads(bindings.read_text())
    previous["bindings"]["model"]["root"] = str(bindings.parent)
    bindings.write_text(json.dumps(previous))
    monkeypatch.setattr(local_inputs.os, "open", lambda *a, **kw: pytest.fail("Must reject before writing a lock"))
    with pytest.raises(ValueError, match="inside an existing registered input"):
        register(setup, "dataset")
