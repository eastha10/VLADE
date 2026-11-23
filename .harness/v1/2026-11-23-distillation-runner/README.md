# Config 기반 Distillation 실행

- `scripts/train/run_distillation.py`와 `src/vlade/research/distillation.py`를 추가했습니다. Action-only·CoT+Action KD는 QwenVLA schema 2 S0에서 시작하며 환경 준비, HF 입력 검증, 1-step 파일럿, 본 학습, 고정 HF commit 재개, pending 재전송을 같은 진입점으로 실행합니다.
- `configs/experiments/qwen_vla_action_kd.json`과 `qwen_vla_plan_kd.json`으로 실험을 선택합니다. 세 Colab 노트북에는 소스·설정·실행 호출만 둡니다. 기존 schema 1 연구 CLI는 유지했습니다.
- `qwen_vla_task.py`의 GPU 없는 입력 preflight를 추가했습니다. Validation에서 건너뛴 같은 에피소드 쿼리를 순서대로 재생해 상태를 복원합니다.
- `python -B -m pytest -q -p no:cacheprovider --basetemp=<scratch>/pytest-complete`로 관련 8개 모듈을 실행했습니다. 81개 통과, 이전 노트북 셀과 이름을 직접 참조하는 3개 검사는 제외했습니다. 최신 노트북 호출과 단계별 CLI 전달은 새 회귀 검사로 확인했습니다.
- 두 설정의 실제 CLI `--action plan` 실행에서 CoT 토큰 값 0/48과 동일 S0/T1 참조를 확인했습니다. Python 컴파일, 노트북 구조와 셀 문법, JSON, 문서 링크 및 `git diff --check --ignore-submodules=all` 검사를 통과했습니다.
- CUDA 환경을 사용할 수 없어 실제 T1/S0 파일럿 학습과 비공개 Hub 게시·재개는 미실행입니다.
