"""Build an S2-only index patch without modifying pending Plan worktree edits."""
import difflib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / "AGENTS.md").is_file()
scratch = ROOT / ".harness/v1/2026-11-25-s2-representation-kd/scratch"
scratch.mkdir(parents=True, exist_ok=True)

def head(name):
    return subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT).decode("utf-8")

def current(name):
    return (ROOT / name).read_bytes().decode("utf-8")

snapshots = {}
name = "README.md"
now = current(name)
block = now[now.index("S2의 **T1 중간 표현"):now.index("QwenVLA S0의 [Action KD]")]
anchor = "QwenVLA S0의 [Action KD]"
snapshots[name] = head(name).replace(anchor, block + anchor, 1)

name = "STRUCTURE.md"
now = current(name)
start = now.index("S2 표현 증류는")
end = now.index("실제 가중치로 검증된 학습 경로는", start)
original = head(name)
start_old = original.index("공통 루프의 loss callback에 향후")
end_old = original.index("실제 가중치로 검증된 학습 경로는", start_old)
snapshots[name] = original[:start_old] + now[start:end] + original[end_old:]

name = "docs/environment/action_level_distillation.md"
now = current(name)
block = now[now.index("현재 S2의 **중간 표현"):now.index("L4 24GB에서")]
snapshots[name] = head(name).replace("L4 24GB에서", block + "L4 24GB에서", 1)

name = "src/vlade/training/qwen_vla_task.py"
now = current(name)
assert now.count("plan_generation_provenance") == 2
snapshots[name] = now.replace("plan_generation_provenance", "require_plan_supervision")

name = "src/vlade/training/teacher_inference.py"
now = current(name)
lines = now.splitlines(keepends=True)
index = next(i for i, line in enumerate(lines) if line.lstrip().startswith('"plan_generation": plan_generation_provenance('))
assert lines[index + 1].lstrip().startswith('"plan_supervision": teacher.signature.get(')
now = "".join(lines[:index] + lines[index+2:])
assert now.count("plan_generation_provenance") == 2
snapshots[name] = now.replace("plan_generation_provenance", "require_plan_supervision")

patch = []
for name, content in snapshots.items():
    original = head(name)
    assert original != content
    patch.extend(difflib.unified_diff(original.splitlines(keepends=True), content.splitlines(keepends=True),
                                    fromfile="a/"+name, tofile="b/"+name))
    if name.endswith(".py"):
        compile(content, str(ROOT / name), "exec")
path = scratch / "s2-index.patch"
path.write_bytes("".join(patch).encode("utf-8"))
print("Prepared S2-only patch:", path.relative_to(ROOT), "files:", len(snapshots))
