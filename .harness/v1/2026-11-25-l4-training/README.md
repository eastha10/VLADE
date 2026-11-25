# Colab L4 학습 설정·메모리 제어

2026-11-25 KST. 프로젝트 루트 `C:/Users/kimdo/Desktop/vlade`, branch `main`,
HEAD `4b24ece9dbdc5cb690a1bab189c121af85d95967`에서 작업했습니다.
시작 시 `--ignore-submodules=all` Git 상태에는 변경이 없었습니다. 일반 status는 고정
AVA submodule의 Git metadata 접근 제한으로 실패하여 이후 submodule 무시 조회를 사용했습니다.
commit·push·브랜치 변경·submodule 갱신·Drive 배포·HF 업로드는 하지 않았습니다.

## 실제 변경

- `src/vlade/training/memory.py`: 선택적 L4 profile 검증, 원래 effective batch 확인,
  FP32 AdamW 정적 추정, 공개 HF SDPA 전환·non-reentrant checkpointing, CUDA 측정 로그.
- `src/vlade/training/residency.py`, `teacher_inference.py`: 온라인 T1은 CPU에서 시작하고
  요청마다 Student weight·누적 gradient·optimizer를 CPU로 옮긴 뒤 T1을 GPU에 적재합니다.
  T1·episode context가 CPU로 돌아온 뒤 응답하고 Student를 복원합니다. Parameter 객체·moment를
  보존하며 timeout은 worker 종료 후 복원합니다. Colab available RAM 사전 검사를 추가했습니다.
- `qwen_vla_task.py`: Teacher target을 Student graph 생성 전에 받습니다. autocast를 각 forward로
  제한하여 no-grad replay cast가 gradient forward/Teacher 교대로 넘어가지 않게 했습니다.
  loss·scope·query/episode/detach·누적·optimizer·HF 검증·seal·재개·Hub pointer 계약은 유지했습니다.
- `src/vlade/research/distillation.py`: 기존 KD workflow를 유지하면서 Task-only pretrained 초기화의
  S0도 동일 pilot/resume/Hub 경로로 실행합니다. S0/S1/S2 출력명을 분리합니다.
- `configs/experiments/qwen_vla_{s0,action_kd,plan_kd}_l4.json` 및
  `qwen_vla_action_kd_l4_cache.json`: FP32 파라미터·BF16 autocast, microbatch 1×누적 8,
  SDPA·checkpointing. 원래 full scope·seed·loss·LR·step·검증량을 유지합니다.
- `src/vlade/utils/action_kd_notebook.py`, `scripts/prepare/generate_l4_notebook.py`,
  `experiments/colab_l4_training.ipynb`: 설정/실행 위주의 별도 L4 클라이언트. 기본 plan만 실행합니다.
- `docs/environment/colab_l4_training.md`, 기존 S0/KD 안내, README/STRUCTURE에 경로·사용법·한계를 연결했습니다.
  기존 query-head trainer·FFT·Suite·논문 LoRA와 새 native 경로를 구분했습니다.

## 조사 결과와 실측 범위

로컬 S0의 config를 shape 참고로 meta 모델을 구성해 전체 946,437,191개, backbone 852,985,920개
파라미터를 셌습니다. 모델 가중치와 데이터는 로드하지 않았습니다. FP32 weight/gradient/AdamW
두 moment의 정적 합계는 15,142,995,056 bytes, 약 14.10 GiB이며 activation은 포함하지 않습니다.
고정 HF lock의 AVA/OFT backbone shard 합계는 각각 15,082,600,832 bytes입니다.
BF16 weight+gradient만 대략 28.09 GiB여서 기존 resident full FFT는 L4 24GB에 부적합합니다.
FFT를 LoRA/QLoRA로 변경하지 않았습니다.

로컬 `data/teacher_cache/`에는 `.gitkeep`만 있고 committed lock에는 `teacher_cache_t1`이 없습니다.
`hf_hub_download`로 설정의 불변 registry `195d32e54c7f48960411c32963d91746183de488`의
`assets/registry.json`만 읽으려 했으나 DNS `getaddrinfo` 실패로 완료하지 못했습니다.
원격 cache 존재 여부는 미확인입니다. 실제 가중치·데이터·Teacher cache 다운로드/생성은 하지 않았습니다.
기존 full cache 검증/재사용 및 별도 생성·게시 경로를 보존했습니다.

## 실행한 검증

기존 `C:/Users/kimdo/anaconda3/python.exe`를 사용했습니다. Torch `2.12.0+cpu`,
Transformers `5.9.0`, pytest `8.3.4`, CUDA 사용 불가입니다. 패키지를 설치하지 않았습니다.
각 pytest 실행에서 `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`PYTHONPATH=<root>/src`, `VLADE_PROJECT_ROOT=<root>`, `TEMP/TMP=<이번 scratch>`를 지정했습니다.
회귀 실행은 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, TorchInductor/Trition/CUDA cache도
이번 scratch로 지정했습니다. 모든 basetemp는 새 고유 경로입니다.

```powershell
python -B -m pytest -q -x -p no:cacheprovider --basetemp=.harness/v1/2026-11-25-l4-training/scratch/pytest-l4-complete .harness/v1/2026-11-25-l4-training/checks/test_l4_training.py
```

**29 passed, 1 skipped, 67.14초.** CUDA residency 실기기 1개는 CUDA가 없어 skip했습니다.
tiny native Qwen vision·merger·hybrid language + 원래 AVA/proprio/action에서 eager/SDPA 출력·gradient,
실제 decoder checkpoint 재계산, BF16 CPU replay 이후 backbone gradient, Teacher 요청의 autocast
분리, 누적 8·AdamW·seal·저장/재개를 확인했습니다. 실제 native trainer를 CPU 배치로 주입해
pilot 1 step → 저장 → 2 step 재개, microstep 8/16, final 여부, 변경 profile의 identity 거절을 확인했습니다.
이 trainer smoke의 외부 입력·GPU 배치·Hub만 합성으로 대체했고 모델/손실/backward/optimizer/저장은 실제입니다.
CPU offload 객체·gradient/moment 보존, RAM 거절, timeout 종료 순서, worker JSON pipe와
감독된 S2 preflight 수락·미감독 거절, 새 노트북의 기본 plan·무출력을 확인했습니다.

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-25-l4-training/scratch/pytest-regression-final .harness/v1/2026-11-23-distillation-runner/checks/test_distillation_runner.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_qwen_vla.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_training_loop.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_action_distillation.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hf_assets.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hub_checkpoints.py .harness/v1/2026-11-16-online-action-kd/checks/test_online_action_kd.py .harness/v1/2026-11-24-plan-generation-fix/checks/test_plan_generation_fix.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_fft.py -k 'not test_native_preflight_verifies_teacher_and_student_without_cuda and not test_notebook_is_opt_in_and_keeps_task_only_separate'
```

**138 passed, 2 deselected, 147.67초.** 처음에는 138 passed/2 failed였습니다. 제외한 두 과거 검사는
Plan 감독 이력 없는 S2 Teacher를 허용하는 fixture와 삭제된 03/05 노트북 파일명 참조입니다.
과거 기록을 수정하거나 운영 경계를 완화하지 않았습니다. 이번 검사에서 감독 계약 수락/거절과
현재 설정 전용 노트북을 확인했으며 기존 runner의 현재 노트북 검사도 통과했습니다.
최종 합계 **167 passed, 1 skipped, 과거 fixture 2 deselected**입니다. SWIG deprecation 경고 2개가 있었습니다.

초기 새 검사 두 실패는 비교 시 dropout RNG를 맞추지 않은 것과 재로딩 모델의 train mode/RNG
fixture 문제였습니다. 같은 RNG와 mode를 적용한 최종 상태에서 출력·gradient 및 재개 동등성이 통과했습니다.

변경/신규 Python 8개의 메모리 compile, L4 JSON 4개, 노트북 code 셀 3개의 문법·무출력,
관련 문서의 로컬 링크, `git diff --check --ignore-submodules=all`을 확인했습니다.
노트북은 `python -B -m scripts.prepare.generate_l4_notebook`으로 실제 생성했습니다.
소스 파일별 SHA 목록은 생성하지 않았습니다.

## 미실행

실제 L4 24GB의 FP32/BF16 학습·CUDA SDPA·GPU eviction·메모리 peak/속도, 실제 pretrained 0.8B와
7B T1의 동시 프로세스 실행, Linux high-RAM에서의 적합성, 수렴·LIBERO+ 성공률은 미검증입니다.
비공개 HF 실제 checkpoint 전송/복구, GPU cache 생성, 공식 학습·대용량 다운로드도 하지 않았습니다.
새 Colab에서 사용하려면 검토 후 이 구현을 포함한 Git commit을 별도로 게시해야 합니다.
이번 작업의 테스트 checkpoint·fixture·임시 cache는 scratch에만 생성했으며 완료 시 정리했습니다.
