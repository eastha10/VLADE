"""Merge a selected Hub registry snapshot with the committed input pins."""

from __future__ import annotations

import json
from pathlib import Path


def prepare_registry(code: Path, work: Path, revision: str) -> Path:
    """Keep committed assets fixed while enabling newly published run/cache keys."""
    from vlade.storage.assets import refresh_registry, read_registry
    from vlade.data.episode_data import write_json

    destination = work / "registries" / (revision + ".json")
    if not revision or any(c not in "0123456789abcdef" for c in revision) or len(revision) != 40:
        raise ValueError("Research registry_revision must be an immutable 40-character commit")
    published = refresh_registry(destination, revision=revision)
    committed = read_registry(code / "configs/storage/hf_assets.lock.json")
    published["assets"].update(committed["assets"])
    write_json(destination, published)
    return destination
