# T1 Plan 학습 실행 준비

2026-11-25 KST. 프로젝트 루트 `C:/Users/kimdo/Desktop/vlade`, HEAD
`4b24ece9dbdc5cb690a1bab189c121af85d95967`에서 작업했습니다.
시작 시 CoT KD 노트북·L4·두 GPU Teacher의 미커밋 변경이 있었으며 보존했습니다.
사용자 요청은 기존 Action-only VLA T1을 위한 Plan 학습 실행본 제작입니다.

## 변경

- `src/vlade/data/teacher_plan_annotations.py`, `scripts/data/prepare_teacher_plans.py`:
  HF 검증된 prepared 입력에서 Train 현재 관측 참조·빈 Plan 주석을 내보냅니다.
  전체 coverage·정확한 instruction·중복/누락/held-out/빈 플랜을 검증하고 실제 검토 선언 뒤
  기존 private asset 업로더로 게시합니다. Plan 정답을 합성하거나 기존 T1 출력을 복제하지 않습니다.
  관측 참조에는 현재 두 카메라 상대 경로와 정규화 proprio만 포함합니다.
- `src/vlade/research/teacher_plan_training.py`, `scripts/train/run_teacher_plan.py`:
  기존 FFT/Plan CE/Hub를 호출하는 설정 기반 클라이언트입니다. 초기화는 고정 HF T1,
  새 목적은 Action L1 + 양의 가중치 Plan CE이며 원본 T1의 optimizer 재개는 하지 않습니다.
  주석은 학습 전에 고정 HF 자산으로 해결·검증합니다. 기존 입력/봉인/Plan 이력 검사를 유지합니다.
- `configs/experiments/teacher_plan_training.json`: 1,000 step, LR 2e-5, 누적 8,
  K4/detach2, λ=1 시작값입니다. 검증된 최적 설정이라는 의미가 아닙니다.
- `src/vlade/utils/teacher_plan_training_notebook.py`,
  `scripts/prepare/generate_teacher_plan_training_notebook.py`,
  `experiments/colab_teacher_plan_training.ipynb`: 설정 전용 Colab 실행본.
  기본 스위치는 False이며 실행 시 고정된 Git commit을 사용합니다.
- 별도 pilot은 1 step/warmup0이며 `<group>-pilot/T1_plan_pilot`에 게시합니다.
  main은 원래 T1에서 별도로 시작해 `<group>/T1_plan`에 게시하고, 같은 main만 재개합니다.
  main의 원격 복구 시 pilot 재학습을 요구하지 않습니다.
- 완료 handoff는 로컬 seal·Plan 이력과 HF final pointer·원격 seal/파일 identity를 대조합니다.
  실제 검증한 불변 revision의 S2용 `TEACHER_CHECKPOINT_PIN`과 pending 여부를 출력합니다.
  원격 가중치 전체 재다운로드와 Plan 의미·rollout 검증은 별도입니다.
- `docs/environment/teacher_plan_training.md`, README/STRUCTURE와 기존 Plan/KD 안내에
  주석 작성·게시·새 run·학습/재개·S2 연결을 설명했습니다.

## 실행한 검사

기존 Python을 사용했으며 패키지 설치는 하지 않았습니다. 검사는 Torch CPU 환경에서
수행했고 모든 fixture·checkpoint·임시 경로는 이 작업 scratch에 지정했습니다.
`PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONPATH=<root>/src`,
`VLADE_PROJECT_ROOT=<root>`, `TEMP/TMP/TMPDIR=<task>/scratch`, HF/Transformers offline,
TorchInductor/Triton/CUDA cache를 scratch에 지정했습니다.

노트북 생성:

```powershell
python -B -X utf8 scripts/prepare/generate_teacher_plan_training_notebook.py
```

새 CLI의 실제 plan 액션을 scratch workspace 인수로 실행했으며 설정·별도 Hub run을
출력하고 workspace를 만들거나 HF 입력을 해결하지 않는 것을 확인했습니다.

초기 새 검사 `scratch/pytest-first`는 **30 passed, 2.48초**였습니다.
기존 Plan CE 회귀와 함께 다음 명령을 실행했습니다.

```powershell
python -B -X utf8 -m pytest -q -x -p no:cacheprovider --basetemp=.harness/v1/2026-11-25-teacher-plan-training/scratch/pytest-supervision .harness/v1/2026-11-25-teacher-plan-training/checks/test_teacher_plan_training.py .harness/v1/2026-11-24-plan-generation-fix/checks/test_plan_generation_fix.py
```

**59 passed, 28.09초**, SWIG deprecation 경고 2개였습니다.
새 주석 양식·검토 후 패키지의 기존 PlanAnnotations 연결, 비등록 로컬 데이터 거절,
누락·중복·held-out·instruction·identity 거절, 입력 다운로드 전 미등록 주석 거절,
T1 초기화와 pilot/main 분리·복구, 설정 불변성·HF handoff/pending·seal 변조 거절을 확인했습니다.
새 workflow 검사는 HF/모델/GPU 실행을 대체했으며 실제 학습 성공으로 해석하지 않습니다.
기존 회귀에서는 tiny Llama의 실제 Plan CE·vision/proprio/LM head gradient,
짧은 supervised fit/생성, 실제 checkpoint seal·저장/재개와 고정 HF 캐시 변조 거절을 실행했습니다.

## 미실행

실제 주석 작성·의미 검토·HF 게시, 실제 T1/데이터 다운로드·GPU 학습·메모리/속도/수렴,
실제 원격 checkpoint 게시·복구·handoff, held-out Plan 의미·LIBERO+ rollout은 미실행입니다.
기본 annotation asset의 현재 원격 가용성은 조회하지 않았으며 존재를 가정하지 않습니다.
Git commit/push·브랜치 변경·Drive 배포는 실행하지 않았습니다. Colab에서 새 실행기를
사용하려면 이 구현을 포함한 Git commit을 먼저 원격에 반영해야 합니다.

## 최종 확인과 Git 게시 준비

- 마지막 보완(재개 상태 파일·목표 미완료 거절, 생성 노트북 일치 검사) 후 같은 2개 검사 파일을
  `scratch/pytest-final`에서 실행해 **59 passed, 26.28초**였습니다. 경고 2개는 앞선 SWIG 경고입니다.
- Python 7개 메모리 compile, 설정 JSON, 노트북 15개 셀의 nbformat 스키마·고유 ID·무출력,
  관련 문서의 로컬 링크, 비밀 토큰 패턴 부재·보존 용량 한도, diff 공백 검사를 통과했습니다.
- 사용자가 Git 게시를 추가 요청했습니다. GitHub 연결에서 원격 main이 로컬 HEAD
  `4b24ece9dbdc5cb690a1bab189c121af85d95967`과 일치하는 것을 확인했습니다.
  셸의 원격 조회는 DNS 실패였으며 GitHub 연결로 이번 Plan 학습·CoT 준비 파일만 게시합니다.
  기존 L4/두 GPU의 미커밋 수정은 이번 게시 범위에서 제외하고 로컬에 보존합니다.
- `checks/check_publication_dependencies.py`는 기존 수정 소스 11개를 로컬 HEAD의 코드로
  메모리에서 읽어, 이번 파일들이 이전 미커밋 작업에 의존하지 않는지 검사했습니다.
  `scratch/pytest-publication`에서 **59 passed, 27.23초**였으며 SWIG 경고 2개가 있었습니다.
  저장소 전체 복사·branch 변경 없이 공개할 새 모듈과 기존 HEAD 의존성의 조합을 실행했습니다.
