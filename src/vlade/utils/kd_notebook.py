"""Derive the independent offline action-KD notebook from the shared BC recipe."""
from __future__ import annotations

import json
from pathlib import Path


def generate_kd_notebook(root: Path, digest: str) -> Path:
    from vlade.utils.colab_notebooks import bootstrap, cell, notebook

    reference = json.loads((root / "experiments/colab_03_qwen_task_only.ipynb").read_text(encoding="utf-8"))
    cells = reference["cells"]
    cells[0] = cell("markdown", """
        # 05 · Qwen3.5-0.8B 오프라인 액션 증류
        02에서 검증·선정한 T1의 고정 action cache를 사용합니다. Teacher는 GPU에 로드하지 않습니다.
        손실 = task_weight × 정답 액션 L1 + action_kd_weight × Teacher 액션 L1이며 padding은 제외합니다.
        기본 가중치는 각각 1.0인 시작값으로, 튜닝·성능 검증 결과가 아닙니다. hidden feature 증류는 포함하지 않습니다.
        입력/초기 모델/seed/sampler/학습 예산은 03 Task-only와 동일하고 결과 폴더는 분리합니다.
        **준비 전용:** 설치·다운로드·1-step·overfit·본 학습 플래그는 모두 False입니다.
        순서: 실행 스위치/캐시 경로 확인 → 설치 → 기존 Qwen 가중치 확인(없으면 다운로드 스위치)
        → 캐시/데이터 사전 검사 및 1-step·재로딩 → overfit → 결과 확인 후 본 학습.
        본 학습은 T1-full-action 또는 T1-full-hidden과 원본 prepared 이미지가 필요합니다.
        부분 pilot cache는 episode_limit을 지정한 점검에만 사용할 수 있습니다.
        04의 LoRA checkpoint는 현재 02의 cache 생성 도구와 직접 호환되지 않습니다.
        실제 GPU 학습과 rollout 성공률은 아직 검증하지 않았습니다.
        """, "purpose")
    controls = cell("code", """
        RUN_INSTALL = False
        RUN_DOWNLOAD = False
        RUN_SMOKE = False
        RUN_OVERFIT = False
        RUN_MAIN_KD = False
        kd_defaults = json.loads((CODE / 'configs/experiments/qwen_action_kd.json').read_text())
        TEACHER_CACHE_RELATIVE = kd_defaults['teacher_cache']  # 필요하면 T1-full-hidden으로 변경
        TASK_WEIGHT = kd_defaults['task_weight']
        ACTION_KD_WEIGHT = kd_defaults['action_kd_weight']
        """, "execution-controls")
    cells.insert(2, controls)
    for index, item in enumerate(cells):
        identifier = item["id"]
        source = "".join(item["source"])
        if identifier == "drive-and-source":
            source = bootstrap(digest, isolated_uv=True)
        elif identifier == "install-student":
            source = source.replace("vlade_qwen_task_py313", "vlade_qwen_kd_py313")
            source = "if RUN_INSTALL:\n" + "\n".join("    " + line for line in source.splitlines())
            source += "\nelse:\n    STUDENT_PY = Path('/content/vlade_qwen_kd_py313/bin/python')\n    print('환경 설치 대기: RUN_INSTALL=False')"
        elif identifier == "download-student":
            lines = source.splitlines()
            source = lines[0] + "\nif RUN_DOWNLOAD:\n" + "\n".join("    " + line for line in lines[1:])
            source += "\nelse:\n    print('기존 Qwen 가중치 사용:', QWEN, '/ 다운로드 미실행')"
        elif identifier == "student-config":
            source = source.replace("RUN_MAIN_STUDENT = False", "# RUN_MAIN_KD는 실행 스위치 셀에서 지정합니다.")
            source = source.replace("qwen-task-epoch-seed42", "qwen-action-kd-epoch-seed42")
            source = source.replace("method='task_only'", "method='action_kd'")
            source = source.replace("runs/student_task", "runs/student_action_kd")
            source = source.replace("run_student_task.py", "run_student_kd.py")
            source = source.replace("def student_config(suffix, **updates):", """base.update(teacher_cache=str(ROOT / TEACHER_CACHE_RELATIVE),
            task_weight=TASK_WEIGHT, action_kd_weight=ACTION_KD_WEIGHT)
print('증류 설정:', base['teacher_cache'], 'task=', TASK_WEIGHT, 'action KD=', ACTION_KD_WEIGHT)
def student_config(suffix, **updates):""")
        elif identifier == "student-smoke":
            prefix, body = source.split("train_student(smoke_config, smoke_output)", 1)
            body = "train_student(smoke_config, smoke_output)" + body
            source = prefix + "if RUN_SMOKE:\n" + "\n".join("    " + line for line in body.splitlines())
            source += "\nelse:\n    print('1-step/재로딩 대기: RUN_SMOKE=False')"
        elif identifier == "student-overfit":
            prefix, body = source.split("train_student(overfit_config, overfit_output)", 1)
            body = "train_student(overfit_config, overfit_output)" + body
            # Plotting is optional; the report is the gate for main training.
            body = body.split("import matplotlib.pyplot as plt", 1)[0]
            source = prefix + "if RUN_OVERFIT:\n" + "\n".join("    " + line for line in body.splitlines())
            source += "\nelse:\n    print('작은 episode 점검 대기: RUN_OVERFIT=False')"
        elif identifier == "student-main":
            source = source.replace("RUN_MAIN_STUDENT", "RUN_MAIN_KD")
        if item["cell_type"] == "code":
            cells[index] = cell("code", source, identifier)
    path = root / "experiments/colab_05_qwen_action_distillation.ipynb"
    notebook(path, cells)
    return path
