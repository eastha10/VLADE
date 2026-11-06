"""Local TFDS input checks without TensorFlow or dataset preprocessing."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import vlade.data.episode_data as module


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


@pytest.mark.parametrize("direct", [False, True])
def test_existing_local_tfds_is_read_only(local_tfds, direct):
    root, directory, info = local_tfds
    before = snapshot(root)
    actual, source = module._preparation_source(directory if direct else root)
    assert actual == directory.resolve()
    assert source == {"dataset_info": info,
                      "dataset_info_sha256": module.sha256(directory / "dataset_info.json"),
                      "extraction_manifest": None}
    assert snapshot(root) == before
    assert not (root / "extraction_manifest.json").exists()


def test_extraction_manifest_keeps_legacy_source(local_tfds):
    root, directory, info = local_tfds
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
def test_invalid_local_tfds_is_rejected_before_output(local_tfds, case, message):
    root, directory, info = local_tfds
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
def test_cli_preserves_canonical_and_normalization_checks(local_tfds, tmp_path, monkeypatch, bad_hash):
    root, directory, _ = local_tfds
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
