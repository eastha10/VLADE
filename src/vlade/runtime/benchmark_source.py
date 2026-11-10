"""Install the pinned benchmark namespace and its non-package resource files."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


def install(spec: dict, work: Path, python: Path, env: dict) -> None:
    """Keep the benchmark checkout in the runtime, including BDDL/init resources."""
    from vlade.utils.source import stage_source

    pin = spec["benchmark_source"]
    source = stage_source(pin["repository"], work / "benchmark-code", pin["commit"])
    site = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
            env=env,
        ).strip()
    )
    if not site.resolve().is_relative_to(python.parent.parent.resolve()):
        raise ValueError("Benchmark namespace target is outside the isolated runtime")
    if not (source / "libero/libero/__init__.py").is_file():
        raise ValueError("Pinned benchmark namespace is missing")
    (site / "vlade-benchmark.pth").write_text(str(source) + "\n", encoding="utf-8")
    record = {"pin": pin, "source": str(source), "path_file": str(site / "vlade-benchmark.pth")}
    (python.parent.parent / "benchmark-source.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def verify(spec: dict, python: Path) -> None:
    """Refuse a replaced benchmark implementation or modified runtime checkout."""
    record = json.loads((python.parent.parent / "benchmark-source.json").read_text())
    if record["pin"] != spec["benchmark_source"]:
        raise ValueError("Benchmark source pin differs from the runtime specification")
    source = Path(record["source"])

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    if (
        git("rev-parse", "HEAD") != record["pin"]["commit"]
        or git("remote", "get-url", "origin") != record["pin"]["repository"]
        or git("status", "--porcelain", "--untracked-files=all", "--ignore-submodules=all")
        or Path(record["path_file"]).read_text().strip() != str(source)
    ):
        raise ValueError("Benchmark runtime source changed")
