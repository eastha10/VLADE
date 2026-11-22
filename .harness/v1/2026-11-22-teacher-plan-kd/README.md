# T1 접근 플랜을 조건으로 사용하는 Action KD

- `src/vlade/training/{fixed_cache,teacher_inference,student_task,qwen_vla_task}.py`에서 쿼리별 T1 플랜 생성과 전송, 두 Student 입력 경로, 플랜 조건부 검증을 연결했습니다. Student action head의 기존 masked GT L1과 Teacher L1을 사용합니다.
- `configs/experiments/qwen_vla_plan_kd.json`과 `experiments/colab_qwen_vla_unfrozen.ipynb`에서 실행 조건을 지정합니다. `scripts/train/select_plan_kd_weight.py`는 동일 조건의 목표 step checkpoint를 검증하고 최저 validation GT L1의 λ를 반환합니다.
- `python -B -m pytest -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-22-teacher-plan-kd/scratch/pytest-final`로 추가한 플랜 검사, 온라인 KD·기존 Action KD·QwenVLA 검사를 실행해 55개 통과했습니다. 과거 노트북 경로 두 개가 저장소에 없는 검사 1개는 최종 실행에서 제외했습니다. 같은 옵션으로 `--basetemp=.harness/v1/2026-11-22-teacher-plan-kd/scratch/q2`를 지정한 S0 양자화 검사 23개가 통과했습니다. 첫 실행에서는 Windows `PermissionError`가 1건 발생했습니다.
- CPU에서 축소 Llama가 embedding prefix 뒤에 3개 토큰을 생성하는 경로를 확인했습니다. 연구 CLI의 `--action plan`, 변경 Python·JSON·노트북 문법, 문서 링크 및 `git diff --check --ignore-submodules=all`을 검증했습니다.
- 실제 AVA T1 가중치 기반 플랜 생성·품질 평가, GPU 학습, λ 실측 및 롤아웃은 미실행입니다. 로컬 CPU 검증 범위이며 기존 저장소 배포·가중치 업로드는 실행하지 않았습니다.

## Colab 실행본과 플랜 로그

- `experiments/colab_teacher_plan_kd.ipynb`에 HF 입력 확인, 1-step 파일럿, 플랜 로그 확인, 본 학습, ZIP 다운로드, 완료 λ 후보 비교를 연결했습니다. 생성 진입점은 `scripts/prepare/generate_teacher_plan_kd_notebook.py`입니다.
- Native trainer는 `teacher-plans-*.jsonl`에 에피소드, 타임스텝, 작업 지시, 플랜 원문, GT L1, 학습 쿼리의 Teacher L1과 생성 요청 오류를 기록합니다. 실행별 JSONL을 분리하고 각 행을 즉시 flush합니다.
- 동일한 pytest 옵션으로 `--basetemp=.harness/v1/2026-11-22-teacher-plan-kd/scratch/pytest-colab`을 지정해 56개가 통과했고, 기존 노트북 경로를 참조하는 검사 1개는 제외했습니다. 생성 노트북 바이트 일치, nbformat 검증, 셀 컴파일, 합성 플랜 조회와 오류 요약을 확인했습니다. 실제 Colab GPU 실행은 미실행입니다.
