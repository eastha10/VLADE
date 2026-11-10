"""Prepare verified LIBERO+ assets for the installed benchmark package."""

from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import zipfile


def prepare(work: Path, asset: str = "hfasset://libero_plus_assets") -> None:
    from vlade.storage.assets import resolve_input
    from vlade.data.colab_assets import sha256
    from vlade.data.episode_data import write_json
    from vlade.storage.assets import safe_relative

    source = resolve_input(asset)
    archive = source / "assets.zip"
    # resolve_input verifies the archive against its pinned HF record before extraction.
    identity = sha256(archive)
    destination = work / "benchmark-assets" / identity[:16]
    receipt = destination / "extraction.json"
    if receipt.is_file():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved.get("archive_sha256") != identity:
            raise ValueError("Benchmark assets belong to another source")
        for name, expected in saved["files"].items():
            path = destination / safe_relative(name)
            if not path.is_file() or path.is_symlink() or sha256(path) != expected:
                raise ValueError("Extracted benchmark assets failed integrity verification")
    else:
        if destination.exists():
            raise FileExistsError(
                "Incomplete benchmark asset extraction exists; use a fresh workspace"
            )
        with zipfile.ZipFile(archive) as stream:
            seen = set()
            for member in stream.infolist():
                name = safe_relative(member.filename.rstrip("/")).as_posix()
                if name.casefold() in seen or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Unsafe or duplicate benchmark ZIP member")
                seen.add(name.casefold())
            destination.mkdir(parents=True)
            stream.extractall(destination)
        files = {
            p.relative_to(destination).as_posix(): sha256(p)
            for p in destination.rglob("*")
            if p.is_file()
        }
        write_json(receipt, {"archive_sha256": identity, "files": files})
    if not (destination / "assets/textures").is_dir():
        raise ValueError("The pinned benchmark asset payload is incomplete")
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("Install the configured evaluation runtime first")
    package = Path(next(iter(spec.submodule_search_locations))) / "libero"
    config_dir = work / "benchmark-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    value = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(work / "benchmark-datasets"),
        "assets": str(destination / "assets"),
    }
    (config_dir / "config.yaml").write_text(yaml.safe_dump(value), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
