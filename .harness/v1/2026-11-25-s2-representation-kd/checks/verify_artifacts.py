"""Check the delivered configuration, notebook and documentation without bytecode."""
import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))
import nbformat
from vlade.utils.representation_kd_notebook import CONFIG_CELL, RUN_CELL, write_notebook

paths = ["src/vlade/distillation/representation.py", "src/vlade/distillation/representation_config.py",
         "src/vlade/research/distillation.py", "src/vlade/training/qwen_vla_task.py",
         "src/vlade/training/teacher_inference.py", "src/vlade/utils/representation_kd_notebook.py",
         "scripts/prepare/generate_representation_kd_notebook.py"]
for name in paths:
    compile((ROOT / name).read_text(encoding="utf-8"), name, "exec")
json.loads((ROOT / "configs/experiments/qwen_vla_representation_kd.json").read_text())
path = ROOT / "experiments/colab_s2_representation_kd_existing_session.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        assert cell.outputs == [] and cell.execution_count is None
        compile(cell.source, cell.id, "exec")
assert notebook.cells[1].source.strip() == CONFIG_CELL.strip()
assert notebook.cells[2].source.strip() == RUN_CELL.strip()
scratch = ROOT / ".harness/v1/2026-11-25-s2-representation-kd/scratch/notebook-regenerated.ipynb"
write_notebook(scratch)
assert scratch.read_bytes() == path.read_bytes()
scratch.unlink()
document = ROOT / "docs/environment/s2_representation_distillation.md"
links = re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8"))
for link in links:
    if not link.startswith("http"):
        assert (document.parent / link).resolve().exists(), link
print(json.dumps({"compiled_sources":len(paths),"notebook_schema":"valid",
    "generated_notebook":"reproduced_byte_exactly","local_links":len(links),"config_json":"valid"}))
