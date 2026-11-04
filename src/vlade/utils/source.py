"""Stage committed source with Git; data and weights belong in the HF cache."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile


def stage_source(repository: str, scratch_root: Path, source_ref: str = "main") -> Path:
    """Resolve a ref once and return a clean checkout of that immutable commit.

    Existing checkouts are never updated: another process may be using them.
    Git checks object integrity, so no per-file SHA256 inventory is needed.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", source_ref) or ".." in source_ref:
        raise ValueError("Invalid Git source reference")
    scratch_root = scratch_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    def git(directory, *args):
        return subprocess.run(["git", "-C", str(directory), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="vlade-fetch-", dir=scratch_root) as temporary:
        staging = Path(temporary) / "source"
        git(scratch_root, "clone", "--no-checkout", "--filter=blob:none", "--", repository, str(staging))
        git(staging, "fetch", "--depth", "1", "origin", source_ref)
        commit = git(staging, "rev-parse", "FETCH_HEAD^{commit}")
        # Keep submodule metadata paths short on Windows; check the full SHA below.
        destination = scratch_root / ("vlade_colab_" + commit[:16])
        if destination.exists():
            if (git(destination, "rev-parse", "HEAD") != commit
                    or git(destination, "remote", "get-url", "origin") != repository
                    or git(destination, "status", "--porcelain", "--untracked-files=normal")
                    or any(line[:1] in {"-", "+", "U"} for line in git(destination, "submodule", "status").splitlines()
                           if "third_party/ava_vla" in line)):
                raise ValueError("Existing source checkout changed; use a fresh source cache")
            return destination
        git(staging, "checkout", "--detach", commit)
        # The parent commit pins the AVA gitlink. Other baselines use their own environments.
        if (staging / ".gitmodules").is_file():
            git(staging, "submodule", "update", "--init", "--depth", "1", "--", "third_party/ava_vla")
        staging.rename(destination)
    return destination
