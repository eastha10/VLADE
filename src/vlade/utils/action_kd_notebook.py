"""Generate config-only QwenVLA training notebooks."""
from __future__ import annotations

import json
from pathlib import Path

from vlade.utils.colab_notebooks import ROOT, bootstrap, cell


def write_notebook(path: Path, *, teacher_plan: bool = False, variant: str = "gt_kd") -> Path:
    profile = "qwen_vla_plan_kd" if teacher_plan else "qwen_vla_action_kd"
    config = (f"src/vlade/legacy/plan_cot/configs/experiments/{profile}.json" if teacher_plan
              else f"configs/experiments/{profile}.json")
    title = ("Legacy · Plan CoT + Action Distillation" if teacher_plan else
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

            CONFIG = {config!r}
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
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                    newline="\r\n" if teacher_plan else None)
    return path


def main() -> None:
    print(write_notebook(ROOT / "experiments/colab_action_distillation.ipynb").relative_to(ROOT))


def write_l4_notebook(path: Path) -> Path:
    """A settings-only client for native S0/S1/S2 with an explicit cache choice."""
    cells = [
        cell("markdown", """
            # VLADE · Colab L4 24GB 학습

            S0, S1 또는 legacy Plan 실험 중 한 작업만 선택합니다. GPU 적합성·메모리·속도는 파일럿에서 확인합니다.
            HF_TOKEN Secret을 허용하고 plan → setup → preflight → pilot → train 순서로 실행합니다.
            s2_legacy_plan은 과거 자연어 Plan 경로이며 현재 표현 증류 S2와 다릅니다.
            legacy Plan은 preflight 다음 probe에서 실제 플랜을 확인합니다.
            검증된 전체 액션 cache가 있으면 s1_cache를 선택합니다. 없으면 s1_online을 사용하거나
            별도 cache_teacher 작업으로 생성·검증·HF 게시합니다. cache를 자동 생성하거나 대체하지 않습니다.
            온라인 Teacher는 Student와 번갈아 GPU에 적재하므로 충분한 CPU RAM과 전송 시간이 필요합니다.
            기존 A100 실행을 재개할 때는 원래 소스·설정을 유지하고 이 프로필로 바꾸지 마십시오.
            세부 제한은 [L4 안내](../docs/environment/colab_l4_training.md)를 따릅니다.
        """, "purpose"),
        cell("code", bootstrap(source_ref="refs/heads/main"), "source"),
        cell("code", """
            from vlade.research.distillation import new_run_group

            PROFILE = 's0'  #@param ['s0', 's1_cache', 's1_online', 's2_legacy_plan']
            CONFIGS = {
                's0': 'configs/experiments/qwen_vla_s0_l4.json',
                's1_cache': 'configs/experiments/qwen_vla_action_kd_l4_cache.json',
                's1_online': 'configs/experiments/qwen_vla_action_kd_l4.json',
                's2_legacy_plan': 'src/vlade/legacy/plan_cot/configs/experiments/qwen_vla_plan_kd_l4.json',
            }
            CONFIG = CONFIGS[PROFILE]
            WORKSPACE = '/content/vlade_l4'
            RUN_GROUP = globals().get('RUN_GROUP') or new_run_group('l4-' + PROFILE)
            ACTION = 'plan'  #@param ['plan', 'setup', 'preflight', 'probe', 'pilot', 'train', 'retry']
            OVERRIDES = {}
            RESUME_REVISIONS = {}  # 예: {'S1': '40자리 HF commit SHA'}
        """, "config"),
        cell("code", """
            from vlade.research.distillation import job

            job(CODE / CONFIG, workspace=WORKSPACE, run_group=RUN_GROUP, action=ACTION,
                overrides=OVERRIDES, resume_revisions=RESUME_REVISIONS, colab=True)
        """, "run"),
    ]
    notebook = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
