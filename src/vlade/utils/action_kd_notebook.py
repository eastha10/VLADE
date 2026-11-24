"""Generate config-only QwenVLA training notebooks."""
from __future__ import annotations

import json
from pathlib import Path

from vlade.utils.colab_notebooks import ROOT, bootstrap, cell


def write_notebook(path: Path, *, teacher_plan: bool = False, variant: str = "gt_kd") -> Path:
    profile = "qwen_vla_plan_kd" if teacher_plan else "qwen_vla_action_kd"
    title = ("S2 · Action + CoT Distillation 훈련" if teacher_plan else
             "QwenVLA 백본 학습 범위 비교" if variant == "gt_only" else "S1 · Action-level Distillation 훈련")
    resume_key = "gt_only" if variant == "gt_only" else "S2" if teacher_plan else "S1"
    group = "qwen-vla-scope" if variant == "gt_only" else profile.replace("_", "-")
    overrides = {"workflow.variant": variant} if variant == "gt_only" else {}
    workflow = "plan → setup → preflight → probe → pilot → train" if teacher_plan else "plan → setup → preflight → pilot → train"
    actions = ["plan", "setup", "preflight"] + (["probe"] if teacher_plan else []) + ["pilot", "train", "retry"]
    cells = [
        cell("markdown", f"""
            # VLADE · {title}

            Colab Secrets에 `HF_TOKEN`을 설정한 다음 `{workflow}` 순서로 실행합니다.
            설정을 바꾸면 새 `RUN_GROUP`을 사용합니다. 재개할 때는 원래 `SOURCE_COMMIT`과 `RUN_GROUP`을 지정합니다.
            자세한 설정과 로그 위치는 [실행 안내](../docs/environment/action_level_distillation.md)를 참고하십시오.
        """, "purpose"),
        cell("code", bootstrap(source_ref="refs/heads/main"), "source"),
        cell("code", f"""
            from vlade.research.distillation import new_run_group

            CONFIG = 'configs/experiments/{profile}.json'
            WORKSPACE = '/content/vlade_distillation'
            RUN_GROUP = globals().get('RUN_GROUP') or new_run_group({group!r})
            ACTION = "plan"  #@param {json.dumps(actions)}
            OVERRIDES = {overrides!r}
            RESUME_REVISIONS = {{}}  # 예: {{'{resume_key}': '40자리 HF commit SHA'}}
        """, "config"),
        cell("code", """
            from vlade.research.distillation import job

            job(CODE / CONFIG, workspace=WORKSPACE, run_group=RUN_GROUP, action=ACTION,
                overrides=OVERRIDES, resume_revisions=RESUME_REVISIONS, colab=True)
        """, "run"),
    ]
    notebook = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    print(write_notebook(ROOT / "experiments/colab_action_distillation.ipynb").relative_to(ROOT))
