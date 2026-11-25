"""Configuration-only Colab client for supervised Teacher Plan training."""
from __future__ import annotations

import json
from pathlib import Path

from vlade.utils.colab_notebooks import bootstrap, cell


def write_notebook(path: Path) -> Path:
    """Generate an output-free notebook; training and annotation logic stay in modules."""
    cells = [
        cell("markdown", """
            # VLADE · T1 Plan 학습 → S2 CoT+Action KD

            기존 Action-only T1을 **초기 가중치**로 사용하고 새 run에서
            **Action L1 + λ × Plan CE**로 전체 Teacher를 학습합니다.
            기존 T1의 optimizer를 재개하거나 기존 가중치를 덮어쓰지 않습니다.
            이미지·proprio·프롬프트는 CE label에서 제외하고 Plan 본문과 EOS를 학습합니다.
            Student S2 학습은 별도 CoT+Action KD 노트북에서 실행합니다.

            **먼저 이 구현을 포함한 Git commit을 원격에 반영하고 SOURCE_REF에 지정하십시오.**
            실행 중인 CoT KD 소스를 덮어쓰지 말고 새 Colab 런타임을 사용합니다.
            A100급 BF16 GPU와 큰 CPU RAM/디스크가 필요하며 실제 적합성은 파일럿에서 확인합니다.
            모든 실행 스위치는 기본 False입니다. 이 파일 생성으로 GPU 학습이나 HF 업로드가 실행되지는 않습니다.

            Plan 정답이 없으면 먼저 template으로 Train 관측 참조·빈 주석을 내보냅니다.
            관측의 두 카메라·proprio·지시를 확인해 작성하고 검토한 주석만 게시하십시오.
            미래 프레임·Validation/Test 정답·지시문 복제·고정 문장을 정답으로 쓰지 않습니다.
            형식 검사와 주석 이력만으로 Plan의 의미 품질을 보장하지 않습니다.
        """, "purpose"),
        cell("code", bootstrap(), "source"),
        cell("code", """
            import json, uuid
            from datetime import datetime, timezone

            if not (CODE / 'src/vlade/legacy/plan_cot/research/teacher_plan_training.py').is_file():
                raise RuntimeError('Plan 학습 구현이 없는 commit입니다. 구현을 반영한 SOURCE_REF로 새 런타임을 시작하세요.')
            from vlade.legacy.plan_cot.research.teacher_plan_training import job, load_settings

            WORKSPACE = '/content/vlade_distillation'
            CONFIG = CODE / 'src/vlade/legacy/plan_cot/configs/experiments/teacher_plan_training.json'
            RUN_GROUP = globals().get('RUN_GROUP') or (
                'teacher-plan-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
            # 주석 게시 이후 출력된 ANNOTATIONS_REGISTRY_REVISION으로 변경하고 새 RUN_GROUP을 쓰십시오.
            REGISTRY_REVISION = '195d32e54c7f48960411c32963d91746183de488'
            ANNOTATIONS_ASSET = 'hfasset://teacher_plans_d_seed42'
            RESUME_REVISION = None  # main의 last가 있는 불변 HF commit SHA; 같은 group/설정/소스 유지
            REVIEW_SOURCE = ''  # 실제 검토 뒤 'human_reviewed' 또는 'externally_reviewed'
            OVERRIDES = {
                'registry_revision': REGISTRY_REVISION,
                'annotations': ANNOTATIONS_ASSET,
                'plan_loss_weight': 1.0,
                'train.max_steps': 1000,
                'train.save_steps': 100,
                'train.eval_steps': 100,
                'train.warmup_steps': 100,
            }
            RUN_SETUP = False
            EXPORT_TEMPLATE = False
            PUBLISH_REVIEWED = False
            RUN_PREFLIGHT = False
            RUN_PILOT = False
            RUN_TRAIN = False
            RETRY_UPLOAD = False
            EXPORT_TEACHER_PIN = False

            _selection = (SOURCE_COMMIT, WORKSPACE, RUN_GROUP, json.dumps(OVERRIDES, sort_keys=True))
            def execute(action):
                current = (SOURCE_COMMIT, WORKSPACE, RUN_GROUP, json.dumps(OVERRIDES, sort_keys=True))
                if current != _selection:
                    raise RuntimeError('설정이 바뀌었습니다. 설정과 계획 셀부터 다시 확인하세요.')
                job(CONFIG, workspace=WORKSPACE, run_group=RUN_GROUP, action=action,
                    overrides=OVERRIDES, resume_revision=RESUME_REVISION,
                    review_source=REVIEW_SOURCE or None, colab=True)

            load_settings(CONFIG, OVERRIDES)
            execute('plan')
            print('SOURCE_COMMIT:', SOURCE_COMMIT, '| RUN_GROUP:', RUN_GROUP)
        """, "config-plan"),
        cell("markdown", """
            ## 1. AVA 환경 준비

            RUN_SETUP=True로 설정 셀을 적용한 뒤 실행합니다. HF_TOKEN Secret 접근을 허용하십시오.
            setup은 패키지를 설치하며 모델 학습은 실행하지 않습니다.
        """, "setup-help"),
        cell("code", "if RUN_SETUP:\n    execute('setup')", "setup"),
        cell("markdown", """
            ## 2. Train Plan 주석 준비와 게시

            Plan 정답이 이미 HF에 게시됐다면 해당 registry revision·asset 키를 설정하고 3번으로 갑니다.
            없으면 EXPORT_TEMPLATE=True로 실행합니다. 검증된 prepared 입력을 가져올 수 있으므로
            처음에는 다운로드가 발생합니다. 출력은 WORKSPACE/annotations/RUN_GROUP입니다.

            plans.jsonl의 빈 plan을 실제 관측에 맞게 작성합니다. observations.jsonl은 같은 시점의
            primary/wrist 이미지 상대 경로와 정규화된 proprio만 제공하며 미래 액션은 포함하지 않습니다.
            이미지는 WORKSPACE의 검증된 prepared 캐시에서 확인합니다. 모든 Train query의 주석이 필요합니다.
            검토 후 REVIEW_SOURCE를 지정하고 PUBLISH_REVIEWED=True로 게시합니다.
            비공개 dataset VLADE-konkuk/libero-plus-prepared와 모델 registry를 갱신합니다.

            게시가 끝나면 출력된 ANNOTATIONS_REGISTRY_REVISION을 사용하고 **새 RUN_GROUP**으로
            설정·계획부터 다시 시작합니다. 주석 작성용 run과 학습 run의 입력 계약을 분리합니다.
        """, "annotations-help"),
        cell("code", "if EXPORT_TEMPLATE:\n    execute('template')\nif PUBLISH_REVIEWED:\n    execute('publish-annotations')", "annotations"),
        cell("markdown", """
            ## 3. 입력·주석·재개 사전 검사

            RUN_PREFLIGHT=True로 실행합니다. HF 입력 무결성, 전체 Train 주석 coverage, 원래 T1의
            데이터 계약, canonical 분할·정규화, private Hub 연결을 확인합니다. GPU 학습은 시작하지 않습니다.
            주석이 없는 기본 registry에서는 설명과 함께 중단합니다.
            기존 Action-only T1을 초기화로 쓰므로 이 단계에서 T1의 Plan 이력을 요구하지 않습니다.
        """, "preflight-help"),
        cell("code", "if RUN_PREFLIGHT:\n    execute('preflight')", "preflight"),
        cell("markdown", """
            ## 4. 별도 1-step 파일럿

            RUN_PILOT=True로 실제 1 optimizer step(누적 8회)의 학습·검증·봉인·HF 게시를 확인합니다.
            pilot은 runs/RUN_GROUP-pilot/T1_plan_pilot에 게시됩니다. pilot의 목표는 1 step이며
            본 학습의 완료 모델이 아닙니다. 본 학습은 원래 T1에서 별도로 시작합니다.
            파일럿을 S2의 최종 Teacher로 등록하지 마십시오.
        """, "pilot-help"),
        cell("code", "if RUN_PILOT:\n    execute('pilot')", "pilot"),
        cell("markdown", """
            ## 5. 본 학습과 HF 재개

            RUN_TRAIN=True로 실행합니다. 본 학습은 pilot 저장·원격 검증을 확인한 뒤 원래 T1에서
            시작하며 runs/RUN_GROUP/T1_plan에 게시합니다. 기존 FFT의 full scope·K4/detach2·
            optimizer·Hub pending·best(val_l1) 계약을 유지합니다. λ·LR·step은 미검증 시작값입니다.

            새 런타임의 재개는 같은 SOURCE_COMMIT, RUN_GROUP, 설정, WORKSPACE와
            RESUME_REVISION=<main last가 저장된 불변 HF SHA>를 지정하고 setup → preflight → train을 실행합니다.
            복구된 main checkpoint가 있으면 pilot을 다시 학습하지 않습니다. 원본 T1을 --resume으로 넘기지 않습니다.
        """, "train-help"),
        cell("code", "if RUN_TRAIN:\n    execute('train')", "train"),
        cell("markdown", """
            ## 6. 실패 업로드 복구와 S2 연결

            RETRY_UPLOAD=True는 해당 group의 pilot/main pending을 재시도합니다. 원본을 보존하십시오.
            main 목표 step을 완료한 뒤 EXPORT_TEACHER_PIN=True로 실행하면 로컬 봉인·Plan 이력과
            HF final pointer·원격 seal/file identity를 대조하고 TEACHER_CHECKPOINT_PIN을 출력합니다.
            전체 가중치를 다시 다운로드하는 검사는 아닙니다. 결과는 runs/RUN_GROUP/T1_plan-handoff.json입니다.

            출력 pin을 CoT+Action KD 노트북의 TEACHER_CHECKPOINT_PIN에 넣고 새 S2 run에서
            preflight → probe → pilot → train을 실행합니다. probe의 실제 Plan 원문을 확인하고
            Plan–액션 일치도·held-out rollout을 별도로 평가해야 합니다.
        """, "handoff-help"),
        cell("code", "if RETRY_UPLOAD:\n    execute('retry')\nif EXPORT_TEACHER_PIN:\n    execute('export-pin')", "handoff"),
    ]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\r\n")
    return path
