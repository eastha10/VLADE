"""Settings-only Kaggle/Colab client for two-GPU Teacher execution."""
from __future__ import annotations

import json
from pathlib import Path

from vlade.utils.colab_notebooks import ROOT, bootstrap, cell


def write_notebook(path: Path) -> Path:
    source = bootstrap().replace("Path('/content/vlade_sources')", "BASE / 'sources'")
    source = "BASE = Path('/kaggle/working/vlade_teacher' if PLATFORM == 'kaggle' else '/content/vlade_teacher')\n" + source
    # Path must be available before the common immutable-source bootstrap.
    source = "from pathlib import Path\n" + source
    cells = [
        cell("markdown", """
            # VLADE · Teacher 두 GPU 실행

            AVA/OFT Teacher만 층별로 두 GPU에 배치합니다. Student는 기존 단일 GPU 실행을 유지합니다.
            T4 두 장은 FP16을 사용합니다. BF16 기존 실행의 수치와 동일한 것으로 보고하지 마십시오.
            FFT는 전체 파라미터·기존 optimizer·batch/누적·K4/TBPTT를 유지하고 gradient를 CPU에 보관합니다.
            기본 ACTION=plan은 설치·가중치 다운로드·학습·cache 게시를 실행하지 않습니다.
            source → 설정 → plan → setup → 별도 짧은 파일럿 → 본 실행 순서로 진행합니다.
            T4 실기기 검증 전입니다. GPU 두 장의 여유·CPU RAM·저장 공간은 파일럿에서 확인합니다.
            실행 규약은 [Teacher 분할 안내](../docs/environment/teacher_two_gpu.md)를 따릅니다.
        """, "purpose"),
        cell("code", "PLATFORM = 'kaggle'  # 'kaggle' 또는 'colab'", "platform"),
        cell("code", source, "source"),
        cell("code", """
            import uuid

            PROFILE = 'ava_fft_tp2'  # ava_fft_tp2 / oft_fft_tp2 / cache_teacher_tp2
            ACTION = 'plan'  # plan / setup / run
            CONFIG = CODE / 'configs/research' / (PROFILE + '.json')
            WORKSPACE = BASE / 'work'
            RUN_GROUP_OVERRIDE = ''  # 재개 시 최초 run group 지정
            RUN_GROUPS = globals().get('RUN_GROUPS', {})
            RUN_GROUPS.setdefault(PROFILE, PROFILE.replace('_', '-') + '-' + uuid.uuid4().hex[:8])
            RUN_GROUP = RUN_GROUP_OVERRIDE or RUN_GROUPS[PROFILE]
            OVERRIDES = {'run_group': RUN_GROUP}
            # 별도 FFT 파일럿: 새 group에서 train.max_steps=1, train.warmup_steps=0,
            # train.save_steps=1. 본 실행은 원래 목표와 새 group을 사용합니다.
        """, "config"),
        cell("code", """
            import os
            from vlade.research.client import job

            if PROFILE not in {'ava_fft_tp2', 'oft_fft_tp2', 'cache_teacher_tp2'}:
                raise ValueError('Teacher 프로필을 선택하십시오.')
            if ACTION != 'plan':
                if PLATFORM == 'kaggle':
                    from kaggle_secrets import UserSecretsClient
                    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
                else:
                    from google.colab import userdata
                    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
            job(CONFIG, action=ACTION, workspace=WORKSPACE, overrides=OVERRIDES)
        """, "run"),
    ]
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    print(write_notebook(ROOT / "experiments/teacher_two_gpu.ipynb").relative_to(ROOT))
