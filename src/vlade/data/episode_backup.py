"""Synchronous, integrity-sealed archives for resumable episode preparation."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


BACKUP_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _contract_sha256(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}-",
                                     suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _safe_name(value: str) -> bool:
    return (isinstance(value, str) and value not in {"", ".", ".."} and
            not any(character in value for character in ("/", "\\", ":")))


def _validate_entry(stage: Path, entry: dict) -> str:
    episode_id = entry.get("episode_id")
    if not _safe_name(episode_id):
        raise ValueError("Invalid episode identity in backup checkpoint")
    checkpoint = stage / ".checkpoints" / f"{episode_id}.json"
    if json.loads(checkpoint.read_text(encoding="utf-8")) != entry:
        raise ValueError(f"Episode checkpoint mismatch: {episode_id}")
    folder = stage / "episodes" / episode_id
    files = entry.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Episode checkpoint has no files: {episode_id}")
    for filename, expected in files.items():
        if not _safe_name(filename) or file_sha256(folder / filename) != expected:
            raise ValueError(f"Episode integrity mismatch: {episode_id}/{filename}")
    return file_sha256(checkpoint)


def _read_metadata(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "archive", "archive_sha256", "archive_size",
                "first_index", "last_index", "episode_ids", "checkpoint_sha256",
                "resume_contract_sha256"}
    if (not isinstance(value, dict) or set(value) != required or
            value["schema_version"] != BACKUP_SCHEMA_VERSION):
        raise ValueError(f"Invalid episode backup metadata: {path}")
    archive = value["archive"]
    if not _safe_name(archive) or not archive.endswith(".tar"):
        raise ValueError(f"Invalid episode backup archive name: {archive}")
    ids = value["episode_ids"]
    hashes = value["checkpoint_sha256"]
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or
            any(not _safe_name(item) for item in ids) or
            not isinstance(hashes, dict) or set(hashes) != set(ids)):
        raise ValueError("Invalid episode list in backup metadata")
    return value


def _validate_archive(metadata_path: Path, expected_contract: dict | None = None) -> tuple[dict, Path]:
    metadata = _read_metadata(metadata_path)
    if expected_contract is not None and metadata["resume_contract_sha256"] != _contract_sha256(expected_contract):
        raise ValueError("Episode backup belongs to a different preparation contract")
    archive = metadata_path.parent / metadata["archive"]
    if (not archive.is_file() or archive.stat().st_size != metadata["archive_size"] or
            file_sha256(archive) != metadata["archive_sha256"]):
        raise ValueError(f"Episode backup archive integrity mismatch: {archive}")
    with tarfile.open(archive, "r") as bundle:
        members = bundle.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in member.name or
                    not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe episode backup member: {member.name}")
            allowed = (relative == PurePosixPath("backup.json") or
                       (len(relative.parts) >= 2 and relative.parts[0] in {"episodes", ".checkpoints"}))
            if not allowed:
                raise ValueError(f"Unexpected episode backup member: {member.name}")
        stream = bundle.extractfile("backup.json")
        if stream is None:
            raise ValueError("Episode backup has no internal manifest")
        internal = json.loads(stream.read())
    for key in ("schema_version", "first_index", "last_index", "episode_ids",
                "checkpoint_sha256", "resume_contract_sha256"):
        if internal.get(key) != metadata[key]:
            raise ValueError(f"Episode backup metadata mismatch: {key}")
    return metadata, archive


class EpisodeArchiveBackup:
    """Pause preparation every N episodes and publish one immutable Drive archive."""

    def __init__(self, backup_dir: Path, *, every: int = 100, min_free_gib: float = 10.0):
        if every <= 0 or min_free_gib < 0:
            raise ValueError("Backup interval must be positive and free-space threshold nonnegative")
        self.backup_dir = Path(backup_dir)
        self.every = every
        self.min_free_bytes = int(min_free_gib * 1024**3)
        self.pending: list[tuple[int, dict]] = []

    def check_space(self, stage: Path) -> None:
        free = shutil.disk_usage(stage.parent).free
        if free < self.min_free_bytes:
            raise OSError(f"Local SSD free space below configured threshold: {free} < {self.min_free_bytes}")

    def record(self, stage: Path, index: int, entry: dict) -> None:
        self.pending.append((index, entry))
        if len(self.pending) >= self.every:
            self._publish(stage)

    def flush(self, stage: Path) -> None:
        if self.pending:
            self._publish(stage)

    def _publish(self, stage: Path) -> None:
        self.check_space(stage)
        contract = json.loads((stage / "resume.json").read_text(encoding="utf-8"))
        checkpoint_hashes = {entry["episode_id"]: _validate_entry(stage, entry)
                             for _, entry in self.pending}
        first, last = self.pending[0][0], self.pending[-1][0]
        ids = [entry["episode_id"] for _, entry in self.pending]
        identity = hashlib.sha256(_json_bytes(checkpoint_hashes)).hexdigest()[:12]
        stem = f"episodes-{first:07d}-{last:07d}-{identity}"
        internal = {"schema_version": BACKUP_SCHEMA_VERSION, "first_index": first, "last_index": last,
                    "episode_ids": ids, "checkpoint_sha256": checkpoint_hashes,
                    "resume_contract_sha256": _contract_sha256(contract)}
        metadata_path = self.backup_dir / f"{stem}.json"
        if metadata_path.is_file():
            try:
                existing, _ = _validate_archive(metadata_path, contract)
                if all(existing[key] == internal[key] for key in internal):
                    print(f"Verified existing backup {metadata_path.name} ({len(ids)} episodes)", flush=True)
                    self.pending.clear()
                    return
            except (OSError, ValueError, KeyError, json.JSONDecodeError, tarfile.TarError):
                pass
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=stage.parent, prefix=".episode-backup-") as temporary_dir:
            local_archive = Path(temporary_dir) / f"{stem}.tar"
            with tarfile.open(local_archive, "w") as bundle:
                payload = _json_bytes(internal)
                info = tarfile.TarInfo("backup.json")
                info.size = len(payload)
                info.mtime = 0
                bundle.addfile(info, io.BytesIO(payload))
                for _, entry in self.pending:
                    episode_id = entry["episode_id"]
                    folder = stage / "episodes" / episode_id
                    for filename in sorted(entry["files"]):
                        bundle.add(folder / filename, arcname=f"episodes/{episode_id}/{filename}", recursive=False)
                    checkpoint = stage / ".checkpoints" / f"{episode_id}.json"
                    bundle.add(checkpoint, arcname=f".checkpoints/{episode_id}.json", recursive=False)
            archive_sha256 = file_sha256(local_archive)
            archive_name = local_archive.name
            destination = self.backup_dir / archive_name
            partial = self.backup_dir / f".{archive_name}.{uuid.uuid4().hex}.partial"
            shutil.copyfile(local_archive, partial)
            with partial.open("rb+") as stream:
                os.fsync(stream.fileno())
            if file_sha256(partial) != archive_sha256:
                partial.unlink(missing_ok=True)
                raise OSError("Drive backup copy failed SHA-256 verification")
            os.replace(partial, destination)
            metadata = {**internal, "archive": archive_name, "archive_sha256": archive_sha256,
                        "archive_size": destination.stat().st_size}
            _write_json_atomic(metadata_path, metadata)
            print(f"Published backup {archive_name} ({len(ids)} episodes)", flush=True)
        self.pending.clear()


def _extract_verified_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r") as bundle:
        for member in bundle.getmembers():
            if member.isdir():
                continue
            relative = PurePosixPath(member.name)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read episode backup member: {member.name}")
            with target.open("wb") as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)


def restore_episode_archives(backup_dir: Path, stage: Path, expected_contract: dict) -> dict:
    """Restore valid archives locally; corrupt/incomplete archives are skipped for source replay."""
    backup_dir, stage = Path(backup_dir), Path(stage)
    report = {"restored_episodes": 0, "accepted_archives": [], "skipped_archives": []}
    if not backup_dir.is_dir():
        return report
    stage.mkdir(parents=True, exist_ok=True)
    if (stage / "COMPLETE").exists():
        raise ValueError("Refusing to restore into a completed staging directory")
    seen: dict[str, str] = {}
    for metadata_path in sorted(backup_dir.glob("episodes-*.json")):
        try:
            metadata, archive = _validate_archive(metadata_path, expected_contract)
            for episode_id, digest in metadata["checkpoint_sha256"].items():
                if episode_id in seen and seen[episode_id] != digest:
                    raise ValueError(f"Conflicting backed-up episode: {episode_id}")
                existing_checkpoint = stage / ".checkpoints" / f"{episode_id}.json"
                if existing_checkpoint.is_file():
                    try:
                        existing_entry = json.loads(existing_checkpoint.read_text(encoding="utf-8"))
                        existing_digest = _validate_entry(stage, existing_entry)
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        existing_digest = None
                    if existing_digest is not None and existing_digest != digest:
                        raise ValueError(f"Refusing to overwrite a different valid local episode: {episode_id}")
            with tempfile.TemporaryDirectory(dir=stage.parent, prefix=".episode-restore-") as temporary_dir:
                extracted = Path(temporary_dir)
                _extract_verified_archive(archive, extracted)
                entries = {}
                for episode_id in metadata["episode_ids"]:
                    checkpoint = extracted / ".checkpoints" / f"{episode_id}.json"
                    if file_sha256(checkpoint) != metadata["checkpoint_sha256"][episode_id]:
                        raise ValueError(f"Restored checkpoint hash mismatch: {episode_id}")
                    entry = json.loads(checkpoint.read_text(encoding="utf-8"))
                    _validate_entry(extracted, entry)
                    entries[episode_id] = entry
                for episode_id, entry in entries.items():
                    target_folder = stage / "episodes" / episode_id
                    target_folder.mkdir(parents=True, exist_ok=True)
                    for filename in entry["files"]:
                        os.replace(extracted / "episodes" / episode_id / filename, target_folder / filename)
                    target_checkpoint = stage / ".checkpoints" / f"{episode_id}.json"
                    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(extracted / ".checkpoints" / f"{episode_id}.json", target_checkpoint)
                    seen[episode_id] = metadata["checkpoint_sha256"][episode_id]
            report["accepted_archives"].append(metadata_path.name)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, tarfile.TarError) as error:
            report["skipped_archives"].append({"metadata": metadata_path.name, "error": str(error)})
    report["restored_episodes"] = len(seen)
    return report


def import_legacy_incomplete(source_stage: Path, local_stage: Path) -> int:
    """Copy a legacy Drive stage locally without modifying it; prepare --resume validates it."""
    source_stage, local_stage = Path(source_stage), Path(local_stage)
    if source_stage.resolve() == local_stage.resolve():
        raise ValueError("Legacy and local staging paths must differ")
    copied = 0
    for name in ("episodes", ".checkpoints"):
        source = source_stage / name
        if not source.is_dir():
            continue
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Legacy staging data may not contain symlinks")
        destination = local_stage / name
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            target = destination / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        if name == "episodes":
            copied = sum(path.is_dir() for path in source.iterdir())
    return copied
