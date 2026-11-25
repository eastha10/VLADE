"""Run Plan checks against HEAD dependencies, excluding unrelated working-tree edits."""
from __future__ import annotations

import importlib.abc
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))


class HeadDependencyLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        self.sources = {}
        names = subprocess.check_output(["git", "diff", "--name-only", "--", "src"], cwd=ROOT, text=True).splitlines()
        for path in names:
            if path.startswith("src/vlade/") and path.endswith(".py"):
                source = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT)
                self.sources[path[4:-3].replace("/", ".")] = (path, source)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname, self, origin=str(ROOT / self.sources[fullname][0]))

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, source = self.sources[module.__name__]
        module.__file__ = str(ROOT / path)
        exec(compile(source, module.__file__, "exec"), module.__dict__)


if __name__ == "__main__":
    loader = HeadDependencyLoader()
    sys.meta_path.insert(0, loader)
    print("Testing against HEAD for", len(loader.sources), "unrelated modified source modules", flush=True)
    import pytest

    raise SystemExit(pytest.main(["-q", "-x", "-p", "no:cacheprovider",
        "--basetemp=.harness/v1/2026-11-25-teacher-plan-training/scratch/pytest-publication",
        ".harness/v1/2026-11-25-teacher-plan-training/checks/test_teacher_plan_training.py",
        ".harness/v1/2026-11-24-plan-generation-fix/checks/test_plan_generation_fix.py"]))
