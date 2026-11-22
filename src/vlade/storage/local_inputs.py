"""Bind existing, fully verified HF payloads without copying or modifying them."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
from typing import Callable
import uuid


def _remote_identity(record: dict) -> dict:
    """Check the immutable remote fingerprints; download small metadata only."""
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session
    from .assets import configure_http

    configure_http()
    info = HfApi().repo_info(record["repo_id"], repo_type=record["repo_type"],
                              revision=record["revision"], files_metadata=True)
    if info.sha != record["revision"]:
        raise ValueError("Remote HF revision differs from the pinned record")
    remote = {item.rfilename: item for item in info.siblings}
    if record["format"] in {"directory", "hub_checkpoint"}:
        separator = "/files/" if record["format"] == "directory" else "/"
        expected = {record["prefix"] + separator + name: value for name, value in record["files"].items()}
    elif record["format"] == "upstream_files":
        expected = record["files"]
    else:
        expected = {record["archive_path"]: {"bytes": record["archive_bytes"], "sha256": record["archive_sha256"]}}
    for name, fingerprint in expected.items():
        item = remote.get(name)
        if item is None or item.size != fingerprint["bytes"]:
            raise ValueError("Pinned remote HF file is missing or has a different size")
        if getattr(item, "lfs", None):
            if item.lfs.sha256 != fingerprint["sha256"]:
                raise ValueError("Pinned remote HF LFS hash differs")
        else:
            if fingerprint["bytes"] > 2 * 1024**2:
                raise ValueError("Local binding refuses a large non-LFS metadata download")
            response = get_session().get(hf_hub_url(record["repo_id"], name, revision=record["revision"],
                                                   repo_type=record["repo_type"]), headers=build_hf_headers(), timeout=30)
            response.raise_for_status()
            if (len(response.content) != fingerprint["bytes"]
                    or hashlib.sha256(response.content).hexdigest() != fingerprint["sha256"]):
                raise ValueError("Pinned remote metadata hash differs")
    return {"repo_id": record["repo_id"], "repo_type": record["repo_type"], "revision": info.sha}


def _metadata(root: Path, record: dict) -> None:
    from .assets import _verify_files
    if not root.is_dir():
        raise ValueError("Registered local input directory is missing")
    if record.get("format") in {"directory", "hub_checkpoint", "upstream_files"}:
        if not record.get("files"):
            raise ValueError("Directory binding requires explicit pinned file fingerprints")
        _verify_files(root, record["files"])
    elif record.get("format") == "archive" and record.get("repo_type") == "dataset":
        if not {"manifest.json", "normalization.json", "splits.json"} <= record.get("identity_files", {}).keys():
            raise ValueError("Prepared binding requires pinned manifest, normalization and split fingerprints")
        _verify_files(root, record["identity_files"])
        complete = root / "COMPLETE"
        if not complete.is_file() or complete.is_symlink():
            raise ValueError("Registered prepared dataset is incomplete")
    else:
        raise ValueError("Local bindings require explicit HF files or a prepared dataset archive")


def _prepared_payload(root: Path, progress: Callable[[int], None] | None = None) -> int:
    """The HF-pinned manifest authenticates each existing prepared payload file."""
    from .assets import safe_relative, sha256
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or not isinstance(manifest.get("episodes"), list) or not manifest["episodes"]:
        raise ValueError("Local prepared binding requires a version-2 episode manifest")
    def jobs():
        seen = set()
        for entry in manifest["episodes"]:
            relative = safe_relative(entry["episode_id"])
            if len(relative.parts) != 1 or entry["episode_id"] in seen:
                raise ValueError("Invalid/duplicate prepared episode identity")
            seen.add(entry["episode_id"])
            if entry.get("is_calibration") and entry["split"] != "train":
                raise ValueError("Calibration must remain inside the original train split")
            folder = root / "episodes" / relative
            if not folder.resolve().is_relative_to(root):
                raise ValueError("Prepared episode escapes its registered input")
            files = entry.get("files", {})
            if not {"trajectory.npz", "queries.json"} <= files.keys():
                raise ValueError("Prepared episode omits trajectory/query fingerprints")
            for name, expected in files.items():
                relative = safe_relative(name)
                if len(relative.parts) != 1 or not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise ValueError("Invalid prepared payload fingerprint")
                yield folder / relative, expected
    def check(job):
        path, expected = job
        if not path.is_file() or path.is_symlink():
            raise ValueError("Prepared payload is missing or linked outside the input")
        before = path.stat()
        if sha256(path) != expected:
            raise ValueError("Prepared payload integrity mismatch")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Prepared payload changed during verification")
    count = 0
    iterator = iter(jobs())
    with ThreadPoolExecutor(max_workers=4) as pool:
        while batch := list(islice(iterator, 1024)):
            list(pool.map(check, batch))
            count += len(batch)
            if progress:
                progress(count)
    return count


def _read_bindings(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("bindings"), dict):
        raise ValueError("Invalid local HF input binding schema")
    return data


def registered_roots() -> list[Path]:
    """Expose read-only roots solely for output-path protection."""
    location = os.environ.get("VLADE_LOCAL_INPUTS")
    if not location:
        return []
    value = _read_bindings(Path(location).expanduser().resolve(strict=True))
    roots = [Path(entry["root"]) for entry in value["bindings"].values()]
    if any(not root.is_absolute() for root in roots):
        raise ValueError("Registered local roots must be absolute")
    return [root.resolve() for root in roots]


def register_local_input(key: str, root: Path, bindings: Path, *, registry: Path | None = None,
                         progress: Callable[[int], None] | None = None) -> dict:
    """Verify remote and all local bytes, then write a separate runtime receipt."""
    from .assets import asset_record
    root = root.expanduser().resolve(strict=True)
    bindings = bindings.expanduser().resolve()
    if bindings.is_relative_to(root):
        raise ValueError("Binding metadata must stay outside the existing input directory")
    record = asset_record(key, registry)
    if record.get("format") not in {"directory", "hub_checkpoint", "upstream_files", "archive"}:
        raise ValueError("Unsupported local HF binding format")
    remote = _remote_identity(record)
    _metadata(root, record)
    files = _prepared_payload(root, progress) if record["format"] == "archive" else len(record["files"])
    # Recheck authenticated metadata after scanning the dataset to detect races.
    if record["format"] == "archive":
        _metadata(root, record)
    entry = {"root": str(root), "record": record, "remote": remote,
             "verification": {"all_payload_files_verified": files, "read_only": True,
                              "verified_at_utc": datetime.now(timezone.utc).isoformat()}}
    if bindings.exists():
        previous = _read_bindings(bindings)
        if any(bindings.is_relative_to(Path(e["root"]).resolve()) for e in previous["bindings"].values()):
            raise ValueError("Binding metadata cannot be written inside an existing registered input")
    bindings.parent.mkdir(parents=True, exist_ok=True)
    lock = bindings.with_name(bindings.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = bindings.with_name(bindings.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        value = _read_bindings(bindings) if bindings.exists() else {"schema_version": 1, "bindings": {}}
        # Refuse a receipt location inside any input already registered here.
        if any(bindings.is_relative_to(Path(e["root"]).resolve()) for e in value["bindings"].values()):
            raise ValueError("Binding metadata cannot be written inside an existing registered input")
        value["bindings"][key] = entry
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(bindings)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        lock.unlink()
    return entry


def resolve_registered_input(value: str | Path, *, registry: Path | None = None) -> Path | None:
    """Accept only explicitly bound roots matching the current immutable record.

    Each process rehashes model files or the dataset's pinned metadata. Prepared
    trajectory/query/image readers retain their existing per-file SHA256 checks.
    A runtime receipt never replaces validation of the actual bytes being used.
    """
    from .assets import asset_record
    location = os.environ.get("VLADE_LOCAL_INPUTS")
    if not location:
        return None
    bindings = Path(location).expanduser().resolve(strict=True)
    registered = _read_bindings(bindings)["bindings"]
    text = str(value)
    if text.startswith("hfasset:"):
        key = text[len("hfasset:"):].strip("/\\")
        if key not in registered:
            return None
    else:
        requested = Path(value).expanduser().resolve()
        matches = [k for k, entry in registered.items() if requested == Path(entry["root"])]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("Ambiguous local HF binding")
        key = matches[0]
    entry = registered[key]
    record = asset_record(key, registry)
    if entry.get("record") != record:
        raise ValueError("Local HF binding does not match the pinned registry record")
    expected_remote = {k: record[k] for k in ("repo_id", "repo_type", "revision")}
    if entry.get("remote") != expected_remote:
        raise ValueError("Local HF binding provenance differs")
    stored = Path(entry["root"])
    root = stored.resolve(strict=True)
    if not stored.is_absolute() or root != stored or bindings.is_relative_to(root):
        raise ValueError("Registered local input path changed or contains its receipt")
    _metadata(root, record)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    def progress(count: int) -> None:
        if count % 20480 == 0:
            print(f"Verified {count:,} existing prepared files", flush=True)
    entry = register_local_input(args.key, args.path, args.bindings, registry=args.registry, progress=progress)
    print(json.dumps({"key": args.key, "root": entry["root"], "verification": entry["verification"]}, indent=2))
