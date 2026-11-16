# HF T1 추론 → S0 Action KD · 전용 실행 노트북

2026-11-16 한국 시간. 사용자 요청에 따라 HF T1 가중치의 직접 추론을 S0 학습에 연결하고,
HF 연결·가중치 선택·학습 실행 절차를 별도 ipynb로 작성했습니다.
실제 가중치 다운로드·GPU 학습·HF 원격 변경·Git commit/push·Drive 반영은 실행하지 않았습니다.

## 정본 변경

- `src/vlade/training/teacher_inference.py`: 별도 AVA 환경의 고정 T1 추론, JSON 액션 전달,
  train Episode별 reset, 중간 query 재개 시 인과적 prefix 재추론, 응답 제한 및 worker 종료.
- `src/vlade/training/student_initialization.py`: HF·checkpoint seal·데이터·모델 설정 검사 후
  schema-1 S0의 backbone/action head/processor 전체 복원. KD optimizer는 새로 시작하고 재개와 구분.
- `src/vlade/training/student_task.py`: 기존 masked L1, 공통 loop·optimizer·저장·Hub·재개에
  온라인 T1을 연결. validation은 Teacher 없이 GT만 사용. 기존 cache/Task-only 경로 보존.
- `src/vlade/training/fixed_cache.py`: 기존 SequentialTeacher에 device/layer 인수 추가.
  cache 기본 layer -2는 보존하며 온라인 경로는 T1에 저장된 temporal layer를 전달.
- `src/vlade/training/run_control.py`, `scripts/train/check_student_checkpoint.py`:
  온라인 Teacher pilot identity와 검증된 generated checkpoint processor 재로딩.
- `src/vlade/storage/checkpoint_inputs.py`, `src/vlade/storage/assets.py`: HF runs의 불변 revision,
  seal·COMPLETE·원격 LFS hash를 검사하여 런타임 registry에 pin. hub_checkpoint 입력의 전체
  파일 SHA256 검사 및 로컬 캐시 변조 거부. 모델 재업로드·원격 registry 수정 없음.
- `src/vlade/research/{cli,configuration,execution}.py`: T1/S0/λ 설정, AVA·Student 별도 setup,
  pilot/main 설정 파일 분리 및 resume 경로만 바뀌는 기존 실행 허용. 학습 identity 검사는 유지.
- `configs/research/qwen_kd.json`: S0 초기화 + T1 추론 기본값. `student_s0`, `teacher_t1`은
  committed lock에 없는 선택 키이며 전용 노트북에서 실제 HF run 경로/commit으로 연결 가능.
- `src/vlade/utils/action_kd_notebook.py`, `scripts/prepare/generate_action_kd_notebook.py`,
  `experiments/colab_action_distillation.ipynb`: HF 인증·선택·연결·계획·환경·입력 검사·파일럿·본 학습·재개.
  모든 실행 스위치는 False이며 모든 code cell의 출력은 비어 있음.
- `src/vlade/utils/colab_notebooks.py`, `experiments/colab_vlade.ipynb`: qwen_kd 설명 갱신.
- `scripts/train/run_student_kd.py`: 온라인/캐시 지원 설명 갱신.
- `README.md`, `STRUCTURE.md`, `docs/environment/{action_level_distillation,qwen_action_distillation_colab,research_runs}.md`:
  현재 실행 경로, 노션 수식, HF pin·schema·미검증 범위 안내.

근거: 노션 `11-06 Tasks`의 Action-Level Distillation 수식
https://www.notion.so/40fb5dd0687080008d39d86c89c7af45
`L = L_BC(S, GT) + λ L_KD(S, T1)`. 기존 masked normalized L1을 사용하며 기본 task_weight=1,
action_kd_weight=λ=1은 시작값입니다. 별도 intermediate feature 증류는 구현하지 않았습니다.

## 실제 검증

기존 Anaconda의 Python / torch 2.12.0+cpu / transformers 5.9.0을 사용했습니다.
패키지를 설치하지 않았습니다. 새 검사는 이 작업의 `checks/`에만 작성했습니다.

최종 명령 (프로젝트 루트에서 실행):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:VLADE_PROJECT_ROOT=(Get-Location).Path
$env:PYTHONPATH=((Get-Location).Path+'\src')
$env:TEMP=((Get-Location).Path+'\.harness\v1\2026-11-16-online-action-kd\scratch')
$env:TMP=$env:TEMP
python -B -m pytest .harness/v1/2026-11-16-online-action-kd/checks .harness/v1/2026-11-11-ava-teacher-code/checks/test_action_distillation.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_training_loop.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hf_assets.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_research_runtime.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_final_cache_review.py -k 'not notebook_is_opt_in_and_keeps_task_only_separate' -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-16-online-action-kd/scratch/pytest-05
```

실제 interpreter는 기존 Anaconda python.exe를 사용했습니다. 결과: **88 passed, 1 deselected**,
54.76초. Swig 관련 DeprecationWarning 2개. 실제 JSON pipe의 액션 전달/종료, CPU autograd,
Teacher freeze·state reset/replay, tiny native Qwen 전체 S0 저장/복원, 미등록 로컬 입력 거부,
원격 seal/hash 오류·변조 캐시 거부, 온라인 preflight, pilot→main→resume 설정 경로,
기존 cache/optimizer/RNG 회귀를 확인했습니다. HF API와 AVA model은 해당 검사에서 모의 객체를 사용했습니다.

처음 실행한 검사에서는 예전 개별 03/05 노트북을 요구하는 검사 1개가 FileNotFoundError로 실패했습니다.
이 과거 검사 파일은 변경하지 않고 최종 실행에서 제외했습니다. 현재 통합/전용 노트북의
실행 opt-in·소스 구문·설정 분리는 새 검사로 확인했습니다.
중간 실행에서 합성 T1 fixture의 data identity에 불필요한 counts가 들어간 오류를 고쳤으며
최종 상태에서 위 전체 관련 검사를 다시 실행했습니다.

추가로 실제 실행:

- `python -B scripts/prepare/generate_action_kd_notebook.py`,
  `python -B scripts/prepare/generate_colab_notebooks.py`: 각 정본 notebook 생성.
- `python -B scripts/research/run.py --config configs/research/qwen_kd.json --workspace .harness/v1/2026-11-16-online-action-kd/scratch/plan-only --action plan`:
  execution=not_started, T1/S0 설정 확인. 해당 plan 경로는 생성되지 않음.
- Python 소스 15개를 메모리 compile, research JSON 18개 validate,
  ipynb 2개의 code cell 구문·빈 출력 검사 및 주요 신규 모듈 import 통과.
- `git diff --ignore-submodules=all --check` 통과.
  기본 submodule 상태 조회는 기존 metadata 접근 제한으로 실패하므로 ignore-submodules로 확인.
- `git check-ignore .harness/v1/2026-11-16-online-action-kd/scratch/probe`: scratch 제외 확인.

## 남은 범위

- 실제 T1/S0 run 경로·불변 HF commit으로 연결한 GPU forward/backward·VRAM·지연·성공률은 미실행.
- schema-2 QwenVLA 학습 전환은 이 경로에 포함하지 않음. 현재 S0 계약과 구조가 다르면 거부.
- Colab에서 사용할 소스 commit 배포는 사용자 요청 전까지 미실행.
- 기존 사용자 변경 `.harness/v1/2026-11-11-unified-colab/README.md`는 보존.

이번 작업의 합성 fixture/checkpoint/캐시는 scratch에서 생성했으며 검증 완료 후 정리했습니다.
과거 기록과 공식 데이터·가중치·외부 submodule은 변경하지 않았습니다.
