"""Generate the QwenVLA trainable-scope and Teacher-plan notebook clients."""
from pathlib import Path

from vlade.utils.action_kd_notebook import write_notebook as write_distillation_notebook


def write_notebook(path: Path, *, teacher_plan: bool = False) -> Path:
    return write_distillation_notebook(path, teacher_plan=teacher_plan,
                                      variant="gt_kd" if teacher_plan else "gt_only")
