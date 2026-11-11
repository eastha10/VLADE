"""Compatibility generator for the consolidated Colab research client."""
from pathlib import Path
from vlade.utils.colab_notebooks import NOTEBOOK_NAME, write_unified_notebook


def generate_paper_notebook(root: Path, digest: str) -> Path:
    """Regenerate the unified client, which includes the ava_paper profile."""
    return write_unified_notebook(root / "experiments" / NOTEBOOK_NAME)
