"""Restore the split S0 checkpoint backup into a verified local directory."""
from __future__ import annotations

import hashlib
import re
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterator

from vlade.utils.filesystem import publish_directory


def _copy_parts(parts_dir: Path, parts: list[dict], archive: BinaryIO) -> str:
    digest = hashlib.sha256()
    for part in parts:
        source = parts_dir / part["name"]
        copied = 0
        with source.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                copied += len(block)
                if copied > part["bytes"]:
                    raise ValueError(f"Unexpected file size: {source.name}")
                digest.update(block)
                archive.write(block)
        if copied != part["bytes"]:
            raise ValueError(f"Unexpected file size: {source.name}")
    return digest.hexdigest()


def _checked_members(archive: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    for member in archive:
        if not (member.isdir() or member.isreg()):
            raise ValueError(f"Unsupported tar entry: {member.name}")
        if member.name.startswith("/") or ".." in member.name.split("/"):
            raise ValueError(f"Invalid tar path: {member.name}")
        if "\\" in member.name or ":" in member.name:
            raise ValueError(f"Invalid tar path: {member.name}")
        yield member


def restore_checkpoint(
    parts_dir: Path,
    output_dir: Path,
    metadata: dict,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    """Join the listed parts, verify the checkpoint seal, and publish its directory."""
    from vlade.training.run_control import verify_checkpoint

    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("Expected a 64-character tar SHA-256 digest")

    parts = metadata["parts"]
    for part in parts:
        path = parts_dir / part["name"]
        if not path.is_file() or path.stat().st_size != part["bytes"]:
            raise ValueError(f"Missing part or unexpected file size: {path}")

    name = metadata["checkpoint_directory"]
    destination = output_dir / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    if output_dir.resolve().is_relative_to(parts_dir.resolve()):
        raise ValueError("Choose an output directory outside the source backup")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s0-restore-", dir=output_dir) as temp:
        stage = Path(temp)
        with tempfile.TemporaryFile(dir=output_dir) as archive:
            digest = _copy_parts(parts_dir, parts, archive)
            if expected_sha256 is not None and digest != expected_sha256.lower():
                raise ValueError(f"Reassembled tar SHA-256 mismatch: {digest}")
            archive.seek(0)
            with tarfile.open(fileobj=archive, mode="r:") as tar:
                tar.extractall(stage, members=_checked_members(tar), filter="data")

        # A tar can store either the checkpoint directory or its contents.
        nested = stage / name
        if nested.is_dir() and list(stage.iterdir()) == [nested]:
            # Move child contents up inside the private staging directory.
            for entry in nested.iterdir():
                entry.rename(stage / entry.name)
            nested.rmdir()
        checkpoint = stage
        seal = verify_checkpoint(checkpoint)
        if seal["step"] != metadata["optimizer_step"]:
            raise ValueError(f"Unexpected optimizer step: {seal['step']}")
        if not (checkpoint / "trainer_state.pt").is_file():
            raise ValueError("Checkpoint is missing trainer_state.pt")
        publish_directory(checkpoint, destination)

    return destination, digest
