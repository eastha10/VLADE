"""Pytest plugin loading changed vlade modules from the exact Git index bytes."""
import importlib.abc
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / "AGENTS.md").is_file()


class StagedSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        names = set()
        for args in (("diff", "--name-only", "HEAD", "--", "src"),
                     ("diff", "--name-only", "--", "src")):
            names.update(subprocess.check_output(["git", *args], cwd=ROOT, text=True).splitlines())
        self.sources = {}
        for name in sorted(names):
            if name.startswith("src/vlade/") and name.endswith(".py"):
                content = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
                module = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
                assert not module.endswith(".__init__"), "Package edits require explicit package handling"
                self.sources[module] = (str(ROOT / name), content)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname, self, origin=self.sources[fullname][0])

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        filename, content = self.sources[module.__name__]
        module.__file__ = filename
        exec(compile(content, filename, "exec"), module.__dict__)


sys.meta_path.insert(0, StagedSources())
