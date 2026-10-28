"""Pinned Hub downloads and safe extraction of the LIBERO+ spanned ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
import uuid
import os
import socket
from pathlib import Path, PurePosixPath
from vlade.utils.filesystem import publish_directory

DATA_REPO = "Sylvest/libero_plus_rlds"
DATA_REV = "fb0c7029b076030d5d57227229e4f7460def1f7c"
AVA_REPO = "LiAuto-DSR/avavla-libero-4in1"
AVA_REV = "090bf68d234ea13bc78476236f82458b25b797b8"
OFT_REPO = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
OFT_REV = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
PARTS = ("libero_plus_mixdata.z01", "libero_plus_mixdata.z02", "libero_plus_mixdata.zip")
PART_METADATA = {
    "libero_plus_mixdata.z01": (32_212_254_720, "61928db465d2ae68994796174ddaaccd951c3728550e8454f3331b32ab555741"),
    "libero_plus_mixdata.z02": (32_212_254_720, "57abe4f928e80b5c40f5f3ffaabafe175ec2b301adf9e85c1b7779a9a6bdb2e0"),
    "libero_plus_mixdata.zip": (11_119_747_860, "56bf87a8decce0a0a331b09f3fe5011af3fb061a1867a761c755283342b0836e"),
}
OWNER = ".vlade-extraction-owner.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(repo: str, revision: str, destination: Path, *, dataset: bool = False) -> dict:
    """Download pinned files, resume via Hub cache, then verify every payload."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = (api.dataset_info if dataset else api.model_info)(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Hub did not resolve the requested immutable revision")
    files = [item for item in info.siblings if (item.rfilename in PARTS if dataset else
             not item.rfilename.startswith(("logs/", ".")))]
    if dataset and {item.rfilename for item in files} != set(PARTS):
        raise ValueError("Expected all three spanned ZIP parts")
    if dataset:
        for item in files:
            expected_size, expected_sha256 = PART_METADATA[item.rfilename]
            official_sha256 = getattr(item.lfs, "sha256", None) if item.lfs else None
            if item.size != expected_size or (official_sha256 and official_sha256 != expected_sha256):
                raise ValueError(f"Pinned dataset metadata changed: {item.rfilename}")
    destination.mkdir(parents=True, exist_ok=True)
    required = sum(item.size or 0 for item in files if not (destination / item.rfilename).exists())
    if shutil.disk_usage(destination).free < required + 2 * 1024**3:
        raise OSError(f"Insufficient space: at least {required / 1024**3:.1f} GiB plus margin required")
    rows = []
    for item in files:
        path = Path(hf_hub_download(repo, item.rfilename, repo_type="dataset" if dataset else "model",
                                   revision=revision, local_dir=destination))
        actual = sha256(path)
        if item.size is not None and path.stat().st_size != item.size:
            raise ValueError(f"Size mismatch: {item.rfilename}")
        lfs = item.lfs
        expected = PART_METADATA[item.rfilename][1] if dataset else (getattr(lfs, "sha256", None) if lfs else None)
        if expected and actual != expected:
            raise ValueError(f"LFS SHA256 mismatch: {item.rfilename}; remove the damaged file and retry")
        if not expected and item.blob_id:
            git_hash = hashlib.sha1()
            git_hash.update(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    git_hash.update(block)
            if git_hash.hexdigest() != item.blob_id:
                raise ValueError(f"Git blob mismatch: {item.rfilename}")
        rows.append({"path": item.rfilename, "bytes": path.stat().st_size, "sha256": actual,
                     "official_lfs_sha256": expected})
        print(f"Verified {item.rfilename}", flush=True)
    result = {"repository": repo, "revision": revision, "files": rows}
    temporary = destination / "download_manifest.json.tmp"
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(destination / "download_manifest.json")
    return result


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    names: set[str] = set()
    for item in members:
        path = PurePosixPath(item.filename)
        if (item.filename == OWNER or path.is_absolute() or ".." in path.parts or "\\" in item.filename or
                ":" in item.filename or (item.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError(f"Unsafe archive member: {item.filename}")
        if item.filename in names:
            raise ValueError(f"Duplicate archive member: {item.filename}")
        names.add(item.filename)
    return members


def _remove_owned_stage(path: Path, parent: Path, identity: dict, *, current_attempt: bool = False) -> None:
    """Remove only our marked partial directory, after checking its absolute boundary."""
    if not path.exists():
        return
    if path.is_symlink() or path.resolve().parent != parent.resolve():
        raise ValueError("Partial extraction is outside its intended parent")
    marker = path / OWNER
    if not marker.is_file():
        return  # Preserve older/unrecognized user content.
    owner = json.loads(marker.read_text(encoding="utf-8"))
    if any(owner.get(key) != value for key, value in identity.items()):
        return
    if not current_attempt:
        # Never reap another live extraction or an attempt from an unknown host.
        # Colab is POSIX; do not use os.kill(pid, 0) on Windows.
        if os.name != "posix" or owner.get("owner_host") != socket.gethostname() or type(owner.get("owner_pid")) is not int:
            return
        try:
            os.kill(owner["owner_pid"], 0)
            return
        except ProcessLookupError:
            pass
        except PermissionError:
            return
    shutil.rmtree(path)


def extract(parts: Path, output: Path, scratch: Path) -> None:
    """Join with Info-ZIP, validate CRC during extraction, publish completion last."""
    if (output / "COMPLETE").is_file():
        saved = json.loads((output / "extraction_manifest.json").read_text(encoding="utf-8"))
        if saved["download_manifest_sha256"] != sha256(parts / "download_manifest.json"):
            raise ValueError("Extracted data came from a different download manifest")
        print("Using already completed extraction", flush=True)
        return
    if output.exists():
        raise FileExistsError(f"Unfinished output exists; use a new path: {output}")
    manifest = json.loads((parts / "download_manifest.json").read_text(encoding="utf-8"))
    if (manifest["repository"], manifest["revision"]) != (DATA_REPO, DATA_REV):
        raise ValueError("Unexpected data source")
    for item in manifest["files"]:
        if sha256(parts / item["path"]) != item["sha256"]:
            raise ValueError(f"Archive changed after verification: {item['path']}")
    identity = {"purpose": "vlade_libero_extraction", "output": str(output.resolve()),
                "download_manifest_sha256": sha256(parts / "download_manifest.json")}
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    for partial in output.parent.glob(f".{output.name}-*.incomplete"):
        _remove_owned_stage(partial, output.parent, identity)
    for partial in scratch.glob("vlade-libero-*.incomplete"):
        _remove_owned_stage(partial, scratch, identity)
    required = sum((parts / name).stat().st_size for name in PARTS)
    if shutil.disk_usage(scratch).free < required + 2 * 1024**3:
        raise OSError("Scratch needs enough space for the combined ~75.5 GB ZIP")
    attempt = uuid.uuid4().hex
    scratch_stage = scratch / f"vlade-libero-{attempt}.incomplete"
    scratch_stage.mkdir()
    owner = {**identity, "owner_host": socket.gethostname(), "owner_pid": os.getpid()}
    (scratch_stage / OWNER).write_text(json.dumps(owner), encoding="utf-8")
    combined = scratch_stage / "combined.zip"
    stage = output.parent / f".{output.name}-{attempt}.incomplete"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / OWNER).write_text(json.dumps(owner), encoding="utf-8")
    try:
        subprocess.run(["zip", "-s", "0", str(parts / PARTS[-1]), "--out", str(combined)], check=True)
        with zipfile.ZipFile(combined) as archive:
            members = safe_members(archive)
            expanded = sum(item.file_size for item in members)
            if shutil.disk_usage(stage).free < expanded + 2 * 1024**3:
                raise OSError(f"Extraction requires {expanded / 1024**3:.1f} GiB plus margin")
            archive.extractall(stage, members)
        infos = list(stage.rglob("dataset_info.json"))
        if len(infos) != 1 or not list(infos[0].parent.glob("*tfrecord*")):
            raise ValueError("Expected exactly one extracted TFDS dataset with TFRecord shards")
        result = {"download_manifest_sha256": identity["download_manifest_sha256"],
                  "tfds_directory": infos[0].parent.relative_to(stage).as_posix(),
                  "expanded_bytes": expanded, "zip_crc_checked": True}
        (stage / "extraction_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (stage / "COMPLETE").write_text("ok\n", encoding="utf-8")
        publish_directory(stage, output)
    finally:
        _remove_owned_stage(stage, output.parent, identity, current_attempt=True)
        _remove_owned_stage(scratch_stage, scratch, identity, current_attempt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("data", "ava", "oft", "extract"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("/content/libero_zip_scratch"))
    args = parser.parse_args()
    if args.action == "data":
        download(DATA_REPO, DATA_REV, args.root / "data/raw/libero_plus_rlds", dataset=True)
    elif args.action == "ava":
        download(AVA_REPO, AVA_REV, args.root / "checkpoints/teacher/avavla-libero-4in1")
    elif args.action == "oft":
        download(OFT_REPO, OFT_REV,
                 args.root / "checkpoints/baselines/openvla_oft/openvla-7b-oft-finetuned-libero-spatial-object-goal-10")
    else:
        extract(args.root / "data/raw/libero_plus_rlds", args.root / "data/processed/libero_plus_tfds", args.scratch)


if __name__ == "__main__":
    main()
