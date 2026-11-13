"""Git commit identity for research code; source files are not hash manifests."""

from __future__ import annotations
from pathlib import Path
import subprocess


def source_identity(root: Path, *, require_clean: bool = False) -> dict[str, str]:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    commit = git("rev-parse", "HEAD")
    if require_clean:
        # User records do not change executable source identity.
        status = git(
            "status",
            "--porcelain",
            "--ignore-submodules=all",
            "--untracked-files=all",
            "--",
            "src",
            "scripts",
            "benchmarks",
            "configs",
            "experiments",
            "README.md",
            "STRUCTURE.md",
            "AGENTS.md",
        )
        if status:
            raise ValueError(
                "Commit the research source/config changes before running a pinned experiment"
            )
    return {"git_commit": commit}
