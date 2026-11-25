# Teacher 두 GPU 실행

2026-11-25 KST. 루트 `C:/Users/kimdo/Desktop/vlade`, branch `main`,
HEAD `4b24ece9dbdc5cb690a1bab189c121af85d95967`을 유지했습니다.
시작 시 앞선 L4 작업의 미커밋 변경이 있었고 이를 보존했습니다. 사용자의 최신 범위는
**Teacher만 분할, Student는 단일 GPU 유지**입니다. AVA/OFT full FFT까지 포함했습니다.
Student 분할이 반드시 빨라지는 것은 아니며 메모리 분할과 데이터 병렬 속도 향상을 구분해 답했습니다.

## 실제 변경

- `src/vlade/training/teacher_parallel.py`: 한 프로세스에서 vision/AVA/decoder prefix와
  decoder suffix/norm/head를 두 CUDA device에 나누고 decoder 입력을 differentiable하게 전송합니다.
  실제 parameter bytes로 경계를 선택하거나 명시적 split을 사용합니다. module key는 보존합니다.
  정상 forward와 checkpoint 재계산에 같은 전송 hook을 사용하고 generation logits를 prefix device로 돌려줍니다.
- `gradient_offload.py`, `loop.py`: 원래 batch·누적·손실·K4/TBPTT를 유지하며 누적 gradient를 CPU에
  보관하고 전체 clipping 후 원래 optimizer의 parameter별 업데이트를 순차 실행합니다.
  group/index/moment/step을 보존합니다. overflow는 동일 window/RNG로 bounded retry하고 실패한
  시도에서 cursor/schedule을 진행하지 않습니다. scaler state를 trainer checkpoint에 넣습니다.
- `fft.py`: 선택적 FP16/BF16 Teacher 배치, 원래 paged AdamW8bit, GPU/RAM 사전 확인,
  두 GPU의 메모리·dtype·실제 split 기록, full checkpoint·scaler의 seal/재개를 연결했습니다.
  BNB 0.43.3의 dense moment loader 대신 원래 paged buffer를 tensor별 재구성하고,
  memory-mapped trainer state를 복원 후 해제합니다. 단일 GPU 경로는 기존 loader를 사용합니다.
- `fixed_cache.py`, `teacher_inference.py`, `residency.py`, `teacher_plan_supervision.py`:
  선택적 두 GPU T1 cache/온라인 추론·Plan dtype·generation·CPU 복귀와 장치별 메모리를 연결했습니다.
  새 cache에는 execution 조건을 기록하고 `teacher_cache_t1_fp16_tp2`로 BF16 cache와 구분합니다.
  HF 입력·T1 선정·Plan 감독·전체 coverage/replay·기존 private Hub와 pending 계약을 유지합니다.
- `models/teachers/components/attn_weight_generator.py`: binary sink mask를 활성 tensor dtype으로
  생성합니다. 고정 BF16 mask가 FP16 weight와 결합될 때 indexed assignment dtype 충돌이 생기는
  문제를 해결했습니다. 수식·sink 좌표·기존 dtype의 출력은 바꾸지 않았습니다. 외부 submodule은 수정하지 않았습니다.
- `research/configuration.py`, `execution.py`: Teacher FFT/cache에만 top-level 설정을 전달합니다.
  `configs/research/{ava_fft,oft_fft,cache_teacher}_tp2.json`은 FP16·두 GPU·CPU gradient를 명시하며
  기존 seed/LR/step/validation/optimizer/AVA K4·detach 2를 유지합니다.
- `utils/teacher_parallel_notebook.py`, `scripts/prepare/generate_teacher_parallel_notebook.py`,
  `experiments/teacher_two_gpu.ipynb`: Kaggle/Colab의 소스·설정·Secrets·연구 CLI 호출 전용 클라이언트.
  기본 plan이며 Student를 자동 분할하지 않습니다.
- `docs/environment/teacher_two_gpu.md`, README/STRUCTURE 및 기존 L4 안내에 사용법·제약을 연결했습니다.

## 실행한 검사

기존 Anaconda Python, Torch `2.12.0+cpu`, Transformers `5.9.0`, pytest `8.3.4`를 사용했습니다.
CUDA는 사용할 수 없었습니다. 패키지 설치는 하지 않았습니다.
pytest는 `-B`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`PYTHONPATH=<root>/src`, `VLADE_PROJECT_ROOT=<root>`로 실행했습니다.
TEMP/TMP·TorchInductor·Triton·CUDA cache와 pytest의 각 고유 basetemp는 이번 scratch로 지정했습니다.
회귀 검사에서는 HF/Transformers offline을 켰습니다.

초기 신규 검사 실행:

```powershell
python -B -m pytest -q -x -p no:cacheprovider --basetemp=.harness/v1/2026-11-25-teacher-two-gpu/scratch/pytest-new-first .harness/v1/2026-11-25-teacher-two-gpu/checks/test_teacher_parallel.py
```

**21 passed, 1 skipped, 36.66초.** 실제 두 CUDA GPU가 필요한 한 검사는 skip했습니다.
원래 AdamW와 3회 optimizer update·누적 8·global clipping/moment의 일치, FP16 overflow의
동일 window/RNG 재시도 및 무갱신 실패, K4/detach 2 gradient, scaler/optimizer 재개,
native Llama의 전송 hook·출력/gradient·state key, seal/설정 변경 거절, plan 무실행을 확인했습니다.
여기에 paged moment 복원·native checkpoint 재계산 검사를 추가한 관련 6개 파일 실행은
`scratch/pytest-related-first`에서 **102 passed, 1 skipped, 50.55초**였습니다.

실제 `compute_loss` → `fft_objective` → `sequence_objective` → 공통 loop 경로에서도
tiny native Llama와 원래 AVA/proprio/action 구성요소로 **K4·detach 2·누적 8·optimizer 1 update**를
CPU에서 수행했습니다. vision은 합성 Linear이며 CUDA device route만 CPU로 대체했습니다.
실제 AVA fork의 attention을 vanilla Llama로 검증한 것으로 주장하지 않습니다.
`scratch/pytest-native-fft-bf16`의 해당 검사 **1 passed, 36.45초**였습니다.
초기 fixture는 FP32 embedding과 BF16 action slot을 혼합해 실패했고, 실제 loader처럼 전체
구성요소를 BF16으로 맞춘 뒤 vision/language/projector/action/proprio/AVA gradient가 통과했습니다.

FP16 sink dtype 거절 회귀와 기존 AVA generator의 고정 upstream 출력 일치도 별도 실행했습니다.
`scratch/pytest-fp16-sink`에서 **2 passed, 37 deselected, 43.39초**였습니다.
이는 다음 종합 검사의 sink 1개와 중복되며 upstream 출력 일치 1개가 추가 확인입니다.

최종 소스의 종합 검사 명령:

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-25-teacher-two-gpu/scratch/pytest-complete .harness/v1/2026-11-25-teacher-two-gpu/checks/test_teacher_parallel.py .harness/v1/2026-11-25-l4-training/checks/test_l4_training.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_training_loop.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_fft.py .harness/v1/2026-11-11-sequence-training/checks/test_temporal_training.py .harness/v1/2026-11-24-plan-generation-fix/checks/test_plan_generation_fix.py .harness/v1/2026-11-16-online-action-kd/checks/test_online_action_kd.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_action_distillation.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hf_assets.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hub_checkpoints.py .harness/v1/2026-11-23-distillation-runner/checks/test_distillation_runner.py -k 'not test_native_preflight_verifies_teacher_and_student_without_cuda and not test_notebook_is_opt_in_and_keeps_task_only_separate'
```

종합 직전 상태의 같은 11개 파일은 `scratch/pytest-final`에서 **183 passed, 2 skipped,
2 deselected, 143.65초**였습니다. 두 제외 검사는 앞선 L4 기록과 같은 과거 fixture로,
Plan 감독 이력 없는 S2 허용 및 삭제된 03/05 노트북 이름 참조입니다. 현 규약에 맞는
수락·거절·현재 노트북 검사는 이번 검사와 기존 L4 검사에서 실행했습니다.

최종 `scratch/pytest-complete`는 **184 passed, 2 skipped, 2 deselected, 122.60초**였습니다.
skip은 새 두-GPU CUDA 검사와 기존 L4 CUDA residency 검사로 장비가 없어 실행하지 못했습니다.
SWIG deprecation 경고 2개가 있었습니다. 최종 상태에서 Student·기존 FFT·HF 거절/수락·Hub
원자적 pointer/pending 및 새 Teacher 분할 계약의 CPU 회귀가 통과했습니다.

변경/신규 Python 14개의 메모리 compile, 새 JSON 3개, 노트북 code 셀 4개의 문법·무출력,
관련 로컬 문서 링크 및 `git diff --check --ignore-submodules=all`을 확인했습니다.
노트북은 생성 진입점으로 실제 생성했습니다. 파일별 소스 SHA 목록은 만들지 않았습니다.

## 미검증과 실행 범위

실제 두 T4/CUDA·legacy Torch 2.2/AVA fork·bitsandbytes GPU paging·두 GPU 사이의 autograd
복사·7B full FFT 적합성/속도/수렴·실제 T1 cache 및 Hub 게시/복구는 미검증입니다.
CPU 테스트와 메모리 분산 구현을 실기기 성공이나 속도 향상으로 보고하지 않습니다.
FP16은 BF16과 수치 범위가 다르며 별도 run/profile/cache로 다룹니다. 큰 CPU RAM과 수십 GB의
checkpoint 보관 공간도 필요합니다. GPU 수와 여유를 확인하지 못하면 Teacher 입력을 해결하기 전에 거절합니다.
첨부의 Pro+ 연동 60시간 혜택은 공식 자료에서 확인하지 못했으며 실행 quota로 가정하지 않았습니다.

GPU 할당·유료 서비스·공식 학습·대용량 다운로드·Git commit/push·브랜치 변경·Drive 배포·
HF 업로드를 수행하지 않았습니다. 기존 L4 정본과 과거 하네스 기록은 보존했습니다.
완료 시 이번 scratch의 합성 checkpoint·fixture·cache를 경계 확인 후 정리했습니다.
