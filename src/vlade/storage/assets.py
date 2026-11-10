"""Resolve portable asset references through an immutable HF registry.

Only completed, hash-verified payloads are published into the local cache.
External training inputs use resolve_input; resolve_path also accepts new local
run outputs and internal preprocessing paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_REPO = "VLADE-konkuk/VLADE-checkpoints"
REGISTRY_FILE = "assets/registry.json"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_INPUTS: dict[tuple[str, str], tuple] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if not value or posix.is_absolute() or any(p in {"..", "."} for p in posix.parts) or "\\" in value or ":" in value:
        raise ValueError("Unsafe asset-relative path")
    return Path(*posix.parts)


def configure_http() -> None:
    """Optional verified TLS compatibility for Windows; never disables certificates."""
    if os.environ.get("VLADE_HF_TLS12") != "1":
        return
    import ssl
    import httpx
    from huggingface_hub import set_client_factory
    context = ssl.create_default_context()
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_2
    set_client_factory(lambda: httpx.Client(transport=httpx.HTTPTransport(verify=context, retries=5),
        follow_redirects=True, timeout=httpx.Timeout(120, connect=20)))


def read_registry(path: Path | None = None) -> dict:
    path = path or Path(os.environ.get("VLADE_ASSET_REGISTRY", ROOT / "configs/storage/hf_assets.lock.json"))
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported HF asset registry")
    return registry


def refresh_registry(destination: Path, *, revision: str = "main") -> dict:
    """Download the completed migration registry at a concrete remote commit."""
    configure_http()
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(REGISTRY_REPO, revision=revision)
    if info.private is not True:
        raise ValueError("VLADE storage registry must be private")
    path = hf_hub_download(REGISTRY_REPO, REGISTRY_FILE, revision=info.sha)
    value = read_registry(Path(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_bytes(Path(path).read_bytes())
    temporary.replace(destination)
    return value


def asset_record(key: str, registry: Path | None = None) -> dict:
    if not NAME.fullmatch(key) or ".." in key:
        raise ValueError("Invalid HF asset key")
    result = read_registry(registry)["assets"].get(key)
    if result is None:
        raise FileNotFoundError(f"HF asset {key} has not been published; finish migration and refresh the registry")
    if not re.fullmatch(r"[0-9a-f]{40}", result.get("revision", "")):
        raise ValueError("Asset must pin a complete immutable HF commit SHA")
    if result.get("repo_type") not in {"model", "dataset"}:
        raise ValueError("Invalid HF asset repository type")
    return result


def cache_path(key: str, record: dict, cache: Path | None = None) -> Path:
    if cache is None:
        env = os.environ.get("VLADE_ASSET_CACHE")
        cache = Path(env) if env else ROOT / ("data/hf_assets" if record["repo_type"] == "dataset" else "checkpoints/hf_assets")
    local_name = record.get("local_name", "content")
    if not NAME.fullmatch(local_name) or ".." in local_name:
        raise ValueError("Invalid asset cache directory name")
    return cache / key / record["revision"] / local_name


def _verify_files(root: Path, files: dict) -> None:
    for relative, expected in files.items():
        path = root / safe_relative(relative)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Asset payload is missing or outside its directory")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"HF asset integrity mismatch: {relative}")


def _extract_archive(archive: Path, stage: Path, expected: dict) -> None:
    with tarfile.open(archive, "r:*") as stream:
        seen = set()
        members = stream.getmembers()
        required = 0
        for member in members:
            target = safe_relative(member.name.rstrip("/"))
            if target.as_posix() in seen or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe or duplicate HF dataset TAR member")
            seen.add(target.as_posix())
            required += member.size
        if shutil.disk_usage(stage.parent).free < required + 1024**3:
            raise OSError("Insufficient local space for HF dataset extraction")
        stage.mkdir()
        stream.extractall(stage, members=members, filter="data")
    root = stage / safe_relative(expected["archive_root"]) if expected.get("archive_root") else stage
    _verify_files(root, expected["identity_files"])
    if not (root / "COMPLETE").is_file():
        raise ValueError("Prepared HF dataset is incomplete")


def _verify_frame_metadata(root: Path, record: dict) -> None:
    """Validate a small streaming descriptor without downloading all image shards."""
    _verify_files(root, record['metadata_files'])
    manifest_path = root / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    complete = json.loads((root / 'COMPLETE').read_text('utf-8'))
    if (sha256(manifest_path) != complete.get('manifest_sha256') or
            manifest.get('identity') != complete.get('identity') or
            manifest.get('stored_frame_interval_steps') != 1 or not manifest.get('files')):
        raise ValueError('Full-frame dataset is incomplete or has a mismatched seal')


def fetch_asset(key: str, *, registry: Path | None = None, cache: Path | None = None) -> Path:
    record = asset_record(key, registry)
    destination = cache_path(key, record, cache)
    receipt = destination.parent / "verified.json"
    if receipt.is_file():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous != record:
            raise ValueError("Cached asset does not match the pinned registry")
        if record["format"] == "archive":
            _verify_files(destination, record["identity_files"])
            if not (destination / 'COMPLETE').is_file():
                raise ValueError('Prepared HF dataset is incomplete')
        elif record["format"] in {"directory", "upstream_files"}:
            _verify_files(destination, record["files"])
        elif record["format"] == "parquet_frames":
            _verify_frame_metadata(destination, record)
        elif record["format"] == "upstream_model":
            manifest = json.loads((destination / "download_manifest.json").read_text(encoding="utf-8"))
            _verify_files(destination, {row["path"]:{"bytes":row["bytes"],"sha256":row["sha256"]}
                                        for row in manifest["files"]})
        elif record["format"] == "rlds_spanned_zip":
            if not (destination / "COMPLETE").is_file():
                raise ValueError("Extracted HF RLDS is incomplete")
            manifest = json.loads((destination / "extraction_manifest.json").read_text("utf-8"))
            folder = destination / safe_relative(manifest["tfds_directory"])
            if not (folder / "dataset_info.json").is_file() or not list(folder.glob("*tfrecord*")):
                raise ValueError("Extracted HF RLDS payload is missing")
        return destination
    if destination.exists():
        raise FileExistsError("Unverified asset cache exists; choose a fresh cache path")
    configure_http()
    from huggingface_hub import hf_hub_download
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / (".download-" + uuid.uuid4().hex)
    if record["format"] == "upstream_model":
        from vlade.data.colab_assets import download
        stage.mkdir()
        download(record["repo_id"], record["revision"], stage)
    elif record["format"] == "rlds_spanned_zip":
        from vlade.data.colab_assets import download, extract
        stage.mkdir()
        download(record["repo_id"], record["revision"], stage / "archives", dataset=True)
        extract(stage / "archives", stage / "content", stage / "zip-scratch")
        from vlade.utils.filesystem import publish_directory
        publish_directory(stage / "content", destination)
        if stage.resolve().parent != destination.parent.resolve() or not stage.name.startswith(".download-"):
            raise ValueError("Unexpected temporary extraction path")
        shutil.rmtree(stage)
    elif record["format"] in {"directory", "upstream_files"}:
        stage.mkdir()
        for relative, expected in record["files"].items():
            target = stage / safe_relative(relative)
            remote = (record["prefix"] + "/files/" + relative
                      if record["format"] == "directory" else relative)
            source = Path(hf_hub_download(record["repo_id"], remote,
                repo_type=record["repo_type"], revision=record["revision"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        _verify_files(stage, record["files"])
    elif record["format"] == "parquet_frames":
        stage.mkdir()
        for relative in record['metadata_files']:
            target = stage / safe_relative(relative)
            source = Path(hf_hub_download(record['repo_id'], relative,
                repo_type=record['repo_type'], revision=record['revision']))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        _verify_frame_metadata(stage, record)
    elif record["format"] == "archive":
        archive = Path(hf_hub_download(record["repo_id"], record["archive_path"],
            repo_type=record["repo_type"], revision=record["revision"]))
        if archive.stat().st_size != record["archive_bytes"] or sha256(archive) != record["archive_sha256"]:
            raise ValueError("HF dataset archive integrity mismatch")
        _extract_archive(archive, stage, record)
        nested = stage / safe_relative(record["archive_root"]) if record.get("archive_root") else stage
        if nested != stage:
            nested.rename(destination)
            # Remove only our freshly created empty extraction parents.
            for parent in sorted(stage.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if parent.is_dir():
                    try: parent.rmdir()
                    except OSError: pass
            try: stage.rmdir()
            except OSError: pass
    else:
        raise ValueError("Unknown HF asset format")
    if not destination.exists():
        from vlade.utils.filesystem import publish_directory
        publish_directory(stage, destination)
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    return destination


def resolve_path(value: str | Path, *, registry: Path | None = None) -> Path:
    text = str(value)
    if text.startswith("hfasset:"):
        key = text[len("hfasset:"):].strip("/\\")
        return fetch_asset(key, registry=registry)
    return Path(value)


def resolve_input(value: str | Path, *, registry: Path | None = None) -> Path:
    """Require HF provenance for external inputs, including already cached paths.

    Local paths are accepted only when they point to the exact verified cache
    directory of a pinned registry entry. New outputs/resume paths use
    resolve_path instead; they are never mistaken for downloaded initial inputs.
    """
    if str(value).startswith("hfasset:"):
        key = str(value)[len("hfasset:"):].strip("/\\")
        record = asset_record(key, registry)
        path = cache_path(key, record).resolve()
        cache = path.parents[2]
    else:
        path = Path(value).expanduser().resolve()
        if len(path.parents) < 3 or not (path.parent / "verified.json").is_file():
            raise ValueError("External weights/data must use hfasset:// keys or a verified HF cache path")
        key = path.parent.parent.name
        record = asset_record(key, registry)
        cache = path.parents[2]
        if cache_path(key, record, cache).resolve() != path:
            raise ValueError("Local input does not match its pinned HF cache path")

    def stamp():
        # Share verification only inside this process. Explicit fetch still does a
        # full integrity check; changed size/timestamps invalidate reuse here.
        names = set(record.get("files", record.get("identity_files", record.get("metadata_files", {}))))
        if record["format"] in {"archive", "rlds_spanned_zip", "parquet_frames"}:
            names.add("COMPLETE")
        if record["format"] == "upstream_model":
            names.add("download_manifest.json")
            names.update(row["path"] for row in json.loads((path / "download_manifest.json").read_text("utf-8"))["files"])
        if record["format"] == "rlds_spanned_zip":
            names.add("extraction_manifest.json")
        paths = [path / safe_relative(name) for name in sorted(names)] + [path.parent / "verified.json"]
        return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths)

    identity = (str(path), json.dumps(record, sort_keys=True))
    try:
        if identity in _INPUTS and _INPUTS[identity] == stamp():
            return path
    except (OSError, ValueError, KeyError):
        _INPUTS.pop(identity, None)
    result = fetch_asset(key, registry=registry, cache=cache)
    _INPUTS[identity] = stamp()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("key")
    fetch.add_argument("--registry", type=Path)
    fetch.add_argument("--cache", type=Path)
    refresh = sub.add_parser("refresh-registry")
    refresh.add_argument("--output", type=Path, required=True)
    refresh.add_argument("--revision", default="main")
    publish = sub.add_parser("publish")
    publish.add_argument("key")
    publish.add_argument("--directory", type=Path, required=True)
    publish.add_argument("--dataset", action="store_true")
    args = parser.parse_args()
    if args.command == "fetch":
        print(fetch_asset(args.key, registry=args.registry, cache=args.cache))
    elif args.command == "refresh-registry":
        value = refresh_registry(args.output, revision=args.revision)
        print(json.dumps({"assets": sorted(value["assets"])}))
    else:
        configure_http()
        from huggingface_hub import HfApi
        from vlade.storage.migrate import publish_files
        directory = args.directory.resolve()
        sources = {p.relative_to(directory).as_posix():p for p in directory.rglob("*")
                   if p.is_file() and not any(part.startswith(".") for part in p.relative_to(directory).parts)}
        if not (directory / "COMPLETE").is_file():
            raise ValueError("Only completed assets can be published")
        if args.dataset and args.key.startswith('teacher_cache_t1'):
            manifest = json.loads((directory / 'cache_manifest.json').read_text(encoding='utf-8'))
            if manifest.get('scope') != 'full_train_split' or manifest.get('status') != 'completed':
                raise ValueError('Only a completed full train split can be published as a T1 cache')
        repo = "VLADE-konkuk/vlade-teacher-cache" if args.dataset else REGISTRY_REPO
        value = publish_files(HfApi(), args.key, sources, repo=repo,
                              kind="dataset" if args.dataset else "model", local_name=directory.name)
        print(json.dumps({"assets": sorted(value["assets"])}))


if __name__ == "__main__":
    main()
