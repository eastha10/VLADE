"""Compatibility names for regenerating the maintained HF/Git notebook clients."""

from pathlib import Path
from vlade.utils.colab_notebooks import PROFILES, write_thin_notebook


def apply_hf_storage(path: Path, root: Path) -> None:
    profile, title = PROFILES[path.name]
    write_thin_notebook(path, profile, title)


def refresh_suite_sources(path: Path, root: Path, digest=None) -> None:
    apply_hf_storage(path, root)


def refresh_inference_sources(path: Path) -> None:
    apply_hf_storage(path, path.parents[1])
