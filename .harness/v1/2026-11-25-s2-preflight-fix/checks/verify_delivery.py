"""Compile changed source and check formal docs, configuration and read-only plan."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / "AGENTS.md").is_file()
TASK = Path(__file__).resolve().parents[1]
sources = ["src/vlade/distillation/representation_config.py", "src/vlade/training/qwen_vla_task.py",
           "src/vlade/research/representation_recovery.py", "src/vlade/utils/representation_kd_notebook.py"]
for name in sources:
    compile((ROOT / name).read_text(encoding="utf-8"), name, "exec")
document = ROOT / "docs/environment/s2_representation_distillation.md"
links = re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8"))
for link in links:
    if not link.startswith("http"):
        assert (document.parent / link).resolve().exists(), link
s1 = json.loads((ROOT / "configs/experiments/qwen_vla_action_kd.json").read_text())
s2 = json.loads((ROOT / "configs/experiments/qwen_vla_representation_kd.json").read_text())
representation = s2["distillation"].pop("representation")
assert s1 == s2
work = TASK / "scratch/plan-must-not-be-created"
assert not work.exists()
process = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/train/run_distillation.py"),
    "--config", "configs/experiments/qwen_vla_representation_kd.json", "--workspace", str(work),
    "--run-group", "s2-cpu-plan", "--action", "plan"],
    env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=subprocess.PIPE, text=True, check=True)
plan = json.loads(process.stdout)
assert plan["hub_runs"] == {"S2": "runs/s2-cpu-plan/S2"} and not work.exists()
assert plan["variants"]["gt_kd"]["distillation"]["representation"] == representation
print(json.dumps({"compiled_sources": len(sources), "local_links": len(links),
                  "config_conditions": "S1 unchanged except representation", "plan": "read-only S2 route"}))
