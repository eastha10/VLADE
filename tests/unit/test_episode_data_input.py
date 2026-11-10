"""HF provenance and TFDS checks without network or real preprocessing."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import vlade.data.episode_data as module
from vlade.storage import assets


@pytest.fixture
def local_tfds(tmp_path):
    root = tmp_path / "existing"
    directory = root / "nested" / "libero_mix" / "1.0.0"
    directory.mkdir(parents=True)
    info = {"name": "libero_mix", "version": "1.0.0", "fileFormat": "tfrecord",
            "splits": [{"name": "train", "shardLengths": ["14347"], "numBytes": "8"}]}
    (directory / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    (directory / "features.json").write_text("{}", encoding="utf-8")
    with (directory / "libero_mix-train.tfrecord-00000-of-00001").open("wb") as stream:
        stream.truncate(8 + 16 * 14347)
    return root, directory, info


def snapshot(root):
    return {p.relative_to(root).as_posix(): module.sha256(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def trusted_tfds(local_tfds, monkeypatch):
    """Isolate TFDS schema tests from the separately tested provenance boundary."""
    monkeypatch.setattr(assets, "resolve_input", lambda value: Path(value))
    return local_tfds


@pytest.fixture
def hf_tfds(local_tfds, tmp_path, monkeypatch):
    source, _, _ = local_tfds
    files = {path.relative_to(source).as_posix(): path.read_bytes()
             for path in source.rglob("*") if path.is_file()}
    record = {"repo_id": "fixture/raw", "repo_type": "dataset", "revision": "a" * 40,
              "format": "directory", "prefix": "assets/raw", "local_name": "tfds",
              "files": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                        for name, payload in files.items()}}
    cache = tmp_path / "cache"
    root = assets.cache_path("raw_rlds", record, cache)
    for name in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    (root.parent / "verified.json").write_text(json.dumps(record), encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": 1, "assets": {"raw_rlds": record}}), encoding="utf-8")
    monkeypatch.setenv("VLADE_ASSET_REGISTRY", str(registry))
    monkeypatch.setenv("VLADE_ASSET_CACHE", str(cache))
    monkeypatch.setattr(assets, "configure_http", lambda: pytest.fail("Cached tests must not use the network"))
    return root


@pytest.mark.parametrize("direct", [False, True])
def test_located_tfds_is_read_only(trusted_tfds, direct):
    root, directory, info = trusted_tfds
    before = snapshot(root)
    actual, source = module._preparation_source(directory if direct else root)
    assert actual == directory.resolve()
    assert source == {"dataset_info": info,
                      "dataset_info_sha256": module.sha256(directory / "dataset_info.json"),
                      "extraction_manifest": None}
    assert snapshot(root) == before
    assert not (root / "extraction_manifest.json").exists()


def test_extraction_manifest_keeps_legacy_source(trusted_tfds):
    root, directory, info = trusted_tfds
    extraction = {"source_revision": "archive-revision", "files": ["existing"]}
    (root / "extraction_manifest.json").write_text(json.dumps(extraction), encoding="utf-8")
    # The old path only located dataset_info and read extraction provenance.
    (directory / "features.json").unlink()
    before = snapshot(root)
    actual, source = module._preparation_source(root)
    assert actual == directory
    assert source == {"dataset_info": info,
                      "dataset_info_sha256": module.sha256(directory / "dataset_info.json"),
                      "extraction_manifest": extraction}
    assert snapshot(root) == before


@pytest.mark.parametrize("case, message", [
    ("name", "Expected local LIBERO"), ("version", "Expected local LIBERO"),
    ("split", "original train split"), ("count", "14,347"),
    ("missing_shard", "shard inventory"), ("extra_shard", "shard inventory"),
    ("truncated", "shard sizes"), ("features", "features.json"),
    ("ambiguous", "Exactly one"), ("path", "libero_mix/1.0.0"),
])
def test_invalid_local_tfds_is_rejected_before_output(trusted_tfds, case, message):
    root, directory, info = trusted_tfds
    shard = directory / "libero_mix-train.tfrecord-00000-of-00001"
    if case in {"name", "version"}:
        info[case] = "wrong"
    elif case == "split":
        info["splits"][0]["name"] = "validation"
    elif case == "count":
        info["splits"][0]["shardLengths"] = ["1"]
    elif case == "missing_shard":
        shard.unlink()
    elif case == "extra_shard":
        (directory / "unexpected.tfrecord").write_bytes(b"x")
    elif case == "truncated":
        shard.write_bytes(b"x")
    elif case == "features":
        (directory / "features.json").unlink()
    elif case == "ambiguous":
        (root / "dataset_info.json").write_text("{}", encoding="utf-8")
    (directory / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    if case == "path":
        directory.rename(directory.with_name("wrong"))
    before = snapshot(root)
    with pytest.raises(ValueError, match=message):
        module._preparation_source(root)
    assert snapshot(root) == before


@pytest.mark.parametrize("bad_hash", [None, "canonical", "normalization"])
def test_cli_preserves_canonical_and_normalization_checks(trusted_tfds, tmp_path, monkeypatch, bad_hash):
    root, directory, _ = trusted_tfds
    manifest = ROOT / "experiments/manifests/libero_plus_rlds/canonical_d_seed42.jsonl.gz"
    normalization = ROOT / "experiments/manifests/libero_plus_rlds/normalization_train_d_seed42.json"
    if not normalization.is_file():
        pytest.skip("External canonical normalization is not installed")
    output = tmp_path / "prepared"
    called = {}
    monkeypatch.setattr(module, "tfds_episodes", lambda path: called.setdefault("directory", path))
    def capture(episodes, destination, **kwargs):
        called.update(destination=destination, options=kwargs)
    monkeypatch.setattr(module, "prepare", capture)
    monkeypatch.setattr(sys, "argv", ["prepare_episode_data", "--tfds-root", str(root),
        "--output", str(output), "--canonical-manifest", str(manifest),
        "--canonical-sha256", "0" * 64 if bad_hash == "canonical" else module.CANONICAL_SHA256,
        "--canonical-normalization", str(normalization), "--canonical-normalization-sha256",
        "0" * 64 if bad_hash == "normalization" else module.sha256(normalization)])
    before = snapshot(root)
    if bad_hash:
        with pytest.raises(ValueError, match="SHA-256"):
            module.main()
        assert not called
    else:
        module.main()
        assert called["directory"] == directory.resolve()
        assert called["destination"] == output
        options = called["options"]
        assert options["seed"] == 42
        assert options["canonical"].sha256 == module.CANONICAL_SHA256
        assert len(options["canonical"].episode_ids) == 14347
        assert options["canonical_normalization"] == json.loads(normalization.read_text())
        assert options["canonical_normalization_sha256"] == module.sha256(normalization)
        assert options["source"]["extraction_manifest"] is None
    assert not output.exists()
    assert snapshot(root) == before


@pytest.mark.parametrize("entrypoint", ["source", "episode_cli", "normalization"])
def test_external_entrypoints_reject_unverified_local_tfds(local_tfds, tmp_path, monkeypatch, entrypoint):
    from vlade.data.canonical_normalization import compute
    root, _, _ = local_tfds
    output = tmp_path / "output"
    before = snapshot(root)
    monkeypatch.setattr(module, "tfds_episodes", lambda *_: pytest.fail("Unverified data must not be read"))
    monkeypatch.setattr(module, "prepare", lambda *_a, **_kw: pytest.fail("Unverified data must not be prepared"))
    with pytest.raises(ValueError, match="hfasset:// keys or a verified HF cache"):
        if entrypoint == "source":
            module._preparation_source(root)
        elif entrypoint == "normalization":
            compute(root, tmp_path / "absent-manifest", output)
        else:
            monkeypatch.setattr(sys, "argv", ["prepare_episode_data", "--tfds-root", str(root),
                "--output", str(output), "--canonical-manifest", str(tmp_path / "absent-manifest"),
                "--canonical-normalization", str(tmp_path / "absent-stats"),
                "--canonical-normalization-sha256", "a" * 64])
            module.main()
    assert not output.exists()
    assert not output.with_name(output.name + ".incomplete").exists()
    assert snapshot(root) == before


@pytest.mark.parametrize("reference", ["key", "cache"])
def test_verified_hf_tfds_is_reused_read_only(hf_tfds, reference):
    before = snapshot(hf_tfds)
    value = Path("hfasset://raw_rlds") if reference == "key" else hf_tfds
    directory, source = module._preparation_source(value)
    assert directory == hf_tfds / "nested/libero_mix/1.0.0"
    assert source["dataset_info"]["name"] == "libero_mix"
    assert snapshot(hf_tfds) == before


@pytest.mark.parametrize("entrypoint", ["source", "normalization"])
def test_tampered_hf_tfds_is_rejected_before_processing(hf_tfds, tmp_path, entrypoint):
    from vlade.data.canonical_normalization import compute
    (hf_tfds / "nested/libero_mix/1.0.0/features.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        if entrypoint == "source":
            module._preparation_source(hf_tfds)
        else:
            compute(hf_tfds, tmp_path / "absent-manifest", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("reference", ["key", "cache"])
def test_normalization_accepts_verified_hf_root_before_shard_check(hf_tfds, tmp_path, reference):
    from vlade.data.canonical_normalization import compute
    value = Path("hfasset://raw_rlds") if reference == "key" else hf_tfds
    manifest = ROOT / "experiments/manifests/libero_plus_rlds/canonical_d_seed42.jsonl.gz"
    with pytest.raises(FileNotFoundError, match="all 1,024 TFRecord shards"):
        compute(value, manifest, tmp_path / "output")
    assert not (tmp_path / "output").exists()
