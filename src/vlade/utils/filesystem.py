"""Publish a completed sibling directory without overwriting existing output."""
from __future__ import annotations
import time
from pathlib import Path


def publish_directory(stage: Path, destination: Path, *, attempts: int = 5) -> None:
    if stage.is_symlink() or stage.resolve().parent != destination.resolve().parent:
        raise ValueError("Publishing requires a regular staging directory in the destination parent")
    for attempt in range(attempts):
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite completed output: {destination}")
        try:
            stage.rename(destination)
            return
        except PermissionError:
            # Windows virus/index scanners can briefly hold a just-written file.
            if attempt + 1 == attempts:
                raise
            time.sleep(.1)
