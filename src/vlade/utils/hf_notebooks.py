"""Compatibility entry points that regenerate the single maintained notebook."""
from pathlib import Path
from vlade.utils.colab_notebooks import NOTEBOOK_NAME, write_unified_notebook


def apply_hf_storage(path: Path, root: Path) -> None:
    """Refresh the unified notebook without recreating retired notebook paths."""
    write_unified_notebook(root / "experiments" / NOTEBOOK_NAME)


def refresh_suite_sources(path: Path, root: Path, digest=None) -> None:
    apply_hf_storage(path, root)


def refresh_inference_sources(path: Path) -> None:
    apply_hf_storage(path, path.parents[1])
