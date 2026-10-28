"""Plain-file Colab source generation, staging and revision regression checks."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.utils import colab_notebooks


MANIFEST = Path("experiments/manifests/colab_source.json")


def write_file(root, relative, contents):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def write_manifest(root, files, revisions=None):
    value = {"schema_version": 1, "files_sha256": files,
             "source_revisions": revisions or {".": "parent-sha", "third_party/ava_vla": "ava-sha",
                                                 "third_party/openvla_oft": "oft-sha"}}
    manifest = write_file(root, MANIFEST, (json.dumps(value, sort_keys=True) + "\n").encode())
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_canonical_normalization_source_hash_uses_git_lf_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_notebooks, "ROOT", tmp_path)
    path = write_file(tmp_path, colab_notebooks.CANONICAL_NORMALIZATION.as_posix(), b'{\r\n  "value": 1\r\n}\r\n')
    expected = hashlib.sha256(b'{\n  "value": 1\n}\n').hexdigest()
    assert colab_notebooks.source_sha256(path) == expected


def test_generation_keeps_source_structure_without_archive_or_self_reference(tmp_path, monkeypatch):
    root = tmp_path / "project"
    included = {
        "src/vlade/core.py": b"VALUE = 42\n",
        "scripts/train/run.py": b"print('run')\n",
        "configs/model.json": b"{}\n",
        "docs/environment/setup.md": b"setup\n",
        "third_party/ava_vla/module.py": b"pass\n",
        "third_party/ava_vla/LICENSE": b"upstream license\n",
        "third_party/openvla_oft/other.py": b"pass\n",
        "experiments/baselines/ava_vla/libero.json": b"{}\n",
        "experiments/manifests/ava_vla_source.json": b"{}\n",
        "README.md": b"readme\n", "STRUCTURE.md": b"structure\n", "AGENTS.md": b"instructions\n",
    }
    excluded = {
        "data/secret.json": b"not source",
        "checkpoints/model.json": b"not source",
        "artifacts/result.json": b"not source",
        "src/.git/private.py": b"not source",
        "src/__pycache__/cached.py": b"not source",
        "src/credentials.env": b"not source",
        "experiments/colab_01_download_data_weights.ipynb": b"old notebook",
    }
    for relative, data in {**included, **excluded}.items():
        write_file(root, relative, data)
    write_manifest(root, {"old.py": "old"})
    monkeypatch.setattr(colab_notebooks, "ROOT", root)
    monkeypatch.setattr(colab_notebooks, "revision", lambda path: "parent-sha" if path == root else
                        "ava-sha" if path.name == "ava_vla" else "oft-sha")

    generated = colab_notebooks.generate()
    first_manifest = (root / MANIFEST).read_bytes()
    manifest = json.loads(first_manifest)
    assert manifest["files_sha256"] == {
        name: hashlib.sha256(contents).hexdigest() for name, contents in included.items()
    }
    assert len(generated) == 5
    assert {p.parent for p in generated[:4]} == {root / "experiments"}
    assert generated[-1] == root / MANIFEST
    digest = hashlib.sha256(first_manifest).hexdigest()
    for path in generated[:4]:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        assert digest in cells["drive-and-source"]
        assert "files.upload" not in cells["drive-and-source"]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), path.name, "exec")
    colab_notebooks.generate()
    assert (root / MANIFEST).read_bytes() == first_manifest
    assert not list(root.rglob("*.zip"))


def test_staging_copies_only_listed_files_and_restores_exact_source(tmp_path):
    root = tmp_path / "drive-project"
    source = write_file(root, "src/vlade/core.py", b"VALUE = 42\n")
    write_file(root, "data/large-marker.bin", b"must stay on Drive")
    digest = write_manifest(root, {"src/vlade/core.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    code = colab_notebooks.stage_source(root, tmp_path / "runtime", digest)
    assert (code / "src/vlade/core.py").read_bytes() == source.read_bytes()
    assert (code / MANIFEST).read_bytes() == (root / MANIFEST).read_bytes()
    assert not (code / "data").exists()
    (code / "src/vlade/core.py").write_bytes(b"damaged staged copy")
    assert colab_notebooks.stage_source(root, tmp_path / "runtime", digest) == code
    assert (code / "src/vlade/core.py").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("change", ["missing", "modified", "wrong-manifest", "missing-manifest",
                                    "parent-path", "absolute-path", "backslash-path", "drive-path"])
def test_staging_rejects_incomplete_or_mismatched_source_before_manifest_publication(tmp_path, change):
    root = tmp_path / "drive-project"
    source = write_file(root, "src/vlade/core.py", b"VALUE = 42\n")
    relative = "src/vlade/core.py"
    if change == "parent-path":
        relative = "../outside.py"
    elif change == "absolute-path":
        relative = "/outside.py"
    elif change == "backslash-path":
        relative = "src\\vlade\\core.py"
    elif change == "drive-path":
        relative = "C:/outside.py"
    digest = write_manifest(root, {relative: hashlib.sha256(source.read_bytes()).hexdigest()})
    if change == "missing":
        source.unlink()
    elif change == "modified":
        source.write_bytes(b"changed after manifest")
    elif change == "wrong-manifest":
        digest = "0" * 64
    elif change == "missing-manifest":
        (root / MANIFEST).unlink()
    scratch = tmp_path / "runtime"
    with pytest.raises((ValueError, FileNotFoundError)):
        colab_notebooks.stage_source(root, scratch, digest)
    assert not list(scratch.rglob("colab_source.json"))


def test_staging_rejects_overlapping_project_and_runtime(tmp_path):
    root = tmp_path / "drive-project"
    digest = write_manifest(root, {})
    with pytest.raises(ValueError):
        colab_notebooks.stage_source(root, root / "runtime", digest)


def test_git_revision_falls_back_to_plain_source_manifest(tmp_path, monkeypatch):
    from vlade.training import fft

    root = tmp_path / "staged-project"
    write_manifest(root, {}, {".": "parent-sha", "third_party/ava_vla": "ava-sha",
                              "third_party/openvla_oft": "oft-sha"})
    monkeypatch.setattr(fft, "REPO_ROOT", root)
    monkeypatch.setattr(fft.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=128, stdout=""))
    assert fft._git_revision(root) == "parent-sha"
    assert fft._git_revision(root / "third_party/ava_vla") == "ava-sha"
    assert fft._git_revision(root / "third_party/openvla_oft") == "oft-sha"
