# VLADE

현재 Colab 실행은 단일 [colab_vlade.ipynb](experiments/colab_vlade.ipynb)에서 작업을 선택하며, [설정 기반 연구 명령](docs/environment/research_runs.md)을 사용합니다. 훈련 step·평가 rollout 수는 `configs/research/`에서 제어합니다.

새 AVA FFT·Suite-balanced는 [연속 K4 시퀀스 학습](docs/environment/ava_sequence_training.md)을 사용합니다.
훈련은 시퀀스 시작에, 추론은 Episode 시작에 상태를 초기화합니다. 기존 전처리본과 Episode 분할은 보존합니다.

현재 로컬/Git 버전은 **Git commit 기반 소스 배포 + Hugging Face 가중치·데이터 입력**을 사용합니다. 실행 중인 Colab과 Drive는 자동 갱신하지 않으며, Drive 업로드는 별도 명령이 있을 때 수행합니다. [새 실행 방식](docs/environment/hf_inputs_and_git_source.md)을 먼저 확인하십시오.

학습 중간·검증 best·목표 step의 final을 비공개 Hugging Face Hub에 자동 저장하려면
[Hub 인증·체크포인트·재개 안내](docs/environment/hugging_face_checkpoints.md)를 참고하십시오.
기존 FFT/Qwen/논문 AVA 루프를 유지하며 환경 변수로 켭니다. 토큰은 Colab Secrets로 전달합니다.

원본의 모든 프레임을 별도 공개 HF 데이터셋으로 보존하는 CPU 재처리는
[전체 프레임 데이터 안내](docs/environment/libero_plus_fullframes.md)를 참고하십시오.
통합 노트북의 `PROFILE='fullframes'`를 선택하며 기존 비공개 자료·학습 기본값은 유지합니다.

**Vision-Language-Action Distillation for Edge Deployment**

VLADE는 대규모 Vision-Language-Action(VLA) 로봇 제어 정책의 시각·언어 이해와 조작 능력을 경량 Student 모델에 전달하여, 제한된 연산 자원을 가진 엣지 디바이스에서도 높은 작업 성공률과 낮은 제어 지연시간을 함께 달성하는 것을 목표로 하는 연구 프로젝트입니다.

핵심 연구 범위는 action distillation, representation distillation, task supervision이며,
주 비교 실험은 LIBERO+에서 Full Fine-Tuning(FFT)으로 수행합니다. 원본 LIBERO는
baseline 재현용으로 분리하고 CALVIN은 후속 일반화 벤치마크로 사용합니다. 최종
후보는 Jetson Orin Nano급 환경에서 모델 크기, VRAM, 추론 지연시간과 제어 주기를
평가할 계획입니다.

기반 Teacher로 AVA-VLA를 연결하고, AVA-VLA와 OpenVLA-OFT를 서로 독립된
baseline으로 등록했습니다. 사용자 포크 `eastha10/AVA-VLA`의 고정 submodule은 출처 확인용으로 보존합니다.
현재 연구 실행은 `src/vlade/`의 모델·데이터·학습·평가 모듈과 `configs/runtime/`의 환경을 사용하며 `third_party/` 없이 구성됩니다.

사용 중인 AVA Teacher의 LIBERO 연속 L1 모델 코드는 `src/vlade/models/teachers/`에
비전 백본·projector·action head·시간 attention·processor별로 정리했습니다.
기존 AVA FFT·논문 학습·순차 Teacher cache가 이 코드를 사용하며, 데이터 파이프라인은
프로젝트의 `src/vlade/data/prismatic/`를 사용합니다. 세부 범위는 [Teacher 안내](docs/environment/ava_vla.md)에 있습니다.

현재 단계는 baseline 등록, FFT 공통 학습 루프와 Teacher 출력 저장·Student 계약의 CPU 검증까지입니다.
실제 Colab GPU 학습은 아직 검증하지 않았습니다. Teacher 어댑터,
Qwen Student의 연속 action head와 별도 VLM 연결 계약은 구현했으며 실제 0.8B 가중치·증류 학습은 아직 검증하지 않았습니다. 두 공개 체크포인트의
A100 단일 추론은 통과했으며, 실제 benchmark 평가 결과는 아직 검증하지 않았습니다. baseline 사용법은
[`experiments/baselines/README.md`](experiments/baselines/README.md), AVA-VLA 연결
방법과 검토할 가중치는 [`docs/environment/ava_vla.md`](docs/environment/ava_vla.md)에
정리했습니다. 로컬 가중치 보관 상태와 Colab A100 이관 조건은
[`docs/environment/local_stage_to_colab.md`](docs/environment/local_stage_to_colab.md)에
기록했습니다.

FFT 실행 방법과 AVA temporal 학습의 조건은
[`docs/environment/libero_plus_fft_colab.md`](docs/environment/libero_plus_fft_colab.md)에
정리했습니다.
통합 노트북의 `ava_fft`는 설정으로 max_steps와 총 8시간 예산을 제어합니다. K4 파일럿에서 실제
속도·메모리를 먼저 확인해야 합니다. 기존 1 epoch 예산 추정은 과거 단일 query 정책입니다.
FFT의 LR·전체 파라미터 학습은 논문 LoRA 설정과 다릅니다.
정책과 차이는 [예산 결정](docs/decisions/0004-ava-a100-budget.md)에 있습니다.

통합 노트북의 `ava_paper` 프로필은 논문 v4
Appendix D.1의 최고 보고 설정(사전학습 OpenVLA 초기화·100,000 step·batch 256)을 준비합니다.
rank 32 LoRA·K=4/2-frame TBPTT·attention 정규화 경로를 별도로 구현했습니다.
다운로드/설치/파일럿/학습은 기본 미실행이며 실제 GPU·LIBERO(+) 성능은 미검증입니다.
LIBERO+ 데이터와 세부 구현 차이는 [04 실행 안내](docs/environment/ava_paper_best_colab.md)에 기록합니다.

단일 액션 추론은 [통합 노트북](experiments/colab_vlade.ipynb)의 `inference_ava`에서 실행합니다.
2026-10-04에는 당시 A100 단일 추론 검사에서 AVA/OFT 두 모델 모두 유한한 8×7 액션 생성을
통과했습니다. 실행 조건과 결과 JSON은 `experiments/runs/inference_smoke/`에
보관했으며, Colab 런타임 종료와 활성 세션 없음까지 확인했습니다. 이 검사는
LIBERO(+) 성공률이나 FFT 학습 검증을 포함하지 않습니다.

통합 노트북의 `qwen_kd` 프로필은 `cache_teacher`로 생성한 전체 T1 action cache를
사용하는 Qwen 오프라인 액션 증류를 준비합니다. GT L1과 Teacher L1의 가중합, cache/원본 데이터
대조, 저장·재개를 연결하며 실행 스위치는 모두 기본 False입니다.
사용 순서는 [액션 증류 안내](docs/environment/qwen_action_distillation_colab.md)에 정리했습니다.
실제 GPU 증류와 성공률은 아직 검증하지 않았습니다.

## 소스 받기

```bash
git clone https://github.com/eastha10/VLADE.git
cd VLADE
```

일반 clone만으로 연구 소스를 받습니다. 실행 방법과 설정 항목은 [연구 실행 안내](docs/environment/research_runs.md)를 따릅니다.

Teacher 연결 설정은 `configs/models/teacher/ava_vla.json`, baseline 설정은
`experiments/baselines/`, 원본·의존성 커밋과 검증 상태는
`experiments/manifests/`에서 관리합니다.

## 이양 원칙

- 채택한 원본 저장소는 `third_party/<name>/`에 두고 직접 수정하지 않습니다.
- 원본 모델과 VLADE 사이의 차이는 `src/vlade/` 아래 어댑터와 증류 로직으로 분리합니다.
- 모델·데이터·실험 조건은 `configs/`에서 관리합니다.
- baseline 재현 결과와 VLADE 결과는 `experiments/`에서 구분합니다.
- 대용량 데이터, 체크포인트, 로그와 영상은 Git에 포함하지 않습니다.
- 모든 실행은 commit, checkpoint, dataset revision, seed, hardware, 환경, latency 측정 조건을 manifest에 남깁니다.

## 기본 실험 흐름

1. Git commit과 HF 입력 revision을 고정하고 연구 설정을 선택합니다.
2. `src/vlade/models/teachers/`와 `students/`에 공통 인터페이스 어댑터를 둡니다.
3. 공식 LIBERO baseline을 재현해 `experiments/baselines/`에 원본 조건을 고정합니다.
4. 두 baseline의 LIBERO+ FFT 조건을 맞춰 주 비교점을 생성합니다.
5. action KD와 representation KD를 `src/vlade/distillation/`에서 한 요소씩 추가합니다.
6. LIBERO+와 CALVIN 평가는 `benchmarks/`와 `src/vlade/evaluation/`을 통해 수행합니다.
7. 최종 후보를 Jetson/TensorRT 경로로 옮겨 지연시간과 메모리를 측정합니다.

등록된 LIBERO baseline 명령은 실제 모델을 로드하지 않고 먼저 확인할 수 있습니다.

```bash
python scripts/evaluate/run_baseline.py ava_vla --task-suite libero_10 --dry-run
python scripts/evaluate/run_baseline.py openvla_oft --task-suite libero_10 --dry-run
```

LIBERO+ FFT를 실행하기 전 파일 준비 상태는 아래처럼 확인합니다.

```bash
python scripts/train/run_fft.py --model ava_vla --preflight
python scripts/train/run_fft.py --model openvla_oft --preflight
```

## 폴더 안내

- `.harness/`: 새 에이전트 작업은 `.harness/v1/YYYY-MM-DD-작업명/`만 사용합니다. `v0/`는 과거 기록 보관본입니다. 보존 기록은 README와 필요한 작은 증거만 남기며, 테스트·검증·임시 의존성은 작업의 `scratch/`에서 사용 후 정리합니다. 자세한 규약은 `AGENTS.md`를 따릅니다.
- `.obsidian/`: Obsidian 작업 공간 설정을 보관합니다.
- `artifacts/`: 실험 로그, 지표, 그림, 영상과 프로파일 결과를 보관합니다.
- `benchmarks/`: LIBERO, LIBERO+, CALVIN 벤치마크 연결 코드를 관리합니다.
- `checkpoints/`: Teacher와 Student 모델 가중치를 보관합니다.
- `configs/`: 데이터, 모델, 증류, 실험과 배포 설정을 관리합니다.
- `data/`: 원본·전처리 데이터, 분할 정보와 Teacher cache를 보관합니다.
- `docs/`: 설계 결정, 환경, 평가 프로토콜과 논문 메모를 기록합니다.
- `experiments/`: 모델별 baseline 설정, ablation, 실행 기록과 manifest를 관리합니다.
- `notebooks/`: 탐색과 분석용 노트북을 보관합니다.
- `scripts/`: 학습, 평가, 배포와 데이터 처리 실행 스크립트를 둡니다.
- `src/`: VLADE의 핵심 구현 코드를 관리합니다.
- `tests/`: 사용자 지정 연구 검사를 위한 경계입니다. 에이전트 검증 전용 코드는 작업별 `.harness/v1/.../checks/`에 둡니다.
- `third_party/`: 원본 출처 확인용 고정 submodule이며 현재 연구 실행에는 필요하지 않습니다.

자세한 디렉터리 역할은 `STRUCTURE.md`를 참고하십시오.

## Training loop 보완 및 backbone 검토 (2026-10-24)

FFT 어댑터는 공통 `src/vlade/training/loop.py`를 사용합니다. 누적 학습,
clipping·비정상 값 검사, 선택적 LR 스케줄·val loss, JSONL 실행 기록,
완료 checkpoint의 optimizer/scheduler/RNG 재개를 지원합니다.
RLDS stream 위치는 복원하지 않으며 GPU FFT는 아직 검증하지 않았습니다.
에이전트 CPU 검사와 합성 smoke는 작업별 하네스 `checks/`에서 실행하며 출력은 해당 `scratch/`로 제한합니다. 실행·재개 제한은 [학습 안내](docs/environment/libero_plus_fft_colab.md)에 있습니다.

기존 Teacher/baseline과 Llama-2-7B 가중치는 유지합니다. Student 후보인 Qwen3,
Qwen3.5, SmolVLM2/SmolVLA, Llama 3.2, Gemma 4의 호환성과 계산량 검토는
[backbone 검토](docs/decisions/0003-backbone-review.md)에 정리했습니다.
이는 후보 우선순위이며 Student 구현·모델 교체나 VLA 성능 검증을 의미하지 않습니다.

## Qwen VLA Student (2026-11-10)

Student 구현은 [qwen_vla.py](src/vlade/models/students/qwen_vla.py)의 `QwenVLA`입니다.
Qwen3.5-0.8B의 자체 비전·내장 merger·언어 모듈에 AVA 시간 상태·proprio projector·
L1 action head를 연결합니다. DINOv2/SigLIP과 외부 3-layer projector는 사용하지 않습니다.

원본 AVA처럼 백본은 `models/students/backbones/qwen.py`, action head·proprio projector·AVA는
`models/students/components/action_heads.py`, `projectors.py`, `attn_weight_generator.py`로 분리했습니다.
설정은 `configs/models/student/qwen_vla_0_8b.json`이며 구조·교체 근거·실행 방법은
[Student 안내](docs/environment/qwen_vla_student.md)에 있습니다.
기존 Qwen query-head 초안과 Colab trainer는 보존하며, pretrained/GPU/rollout은 아직 미검증입니다.

## Teacher 저장·Student 계약·집필 준비 (2026-10-26)

Teacher action/GT token/hidden-state 저장·로드와 provenance·episode 순서·용량 검사를
`src/vlade/data/teacher_records.py`에 구현했습니다. Qwen3.5-0.8B를 별도 VLM으로
연결하는 계약과 연속 `[B,8,7]` action head는 `src/vlade/models/students/qwen.py`에 있습니다.
공식 tokenizer·processor와 축소 무작위 native Qwen 모듈에서 CPU 검증을 수행합니다.
모델 가중치 검토는 사용자 요청에 따라 추후로 미뤘으며 실제 pretrained 0.8B·GPU·KD
학습이나 성공률을 검증한 것은 아닙니다.

- [저장 형식·Student 계약·실행 안내](docs/environment/teacher_records_and_student.md)
- [논문 outline·Related Work·조건/결과표 양식](docs/paper/preparation.md)
- [실험 기록 JSON 템플릿](docs/paper/experiment-record.template.json)

합성 Student 검증 진입점은 작업별 하네스 `checks/student_smoke.py`에 있습니다.

실제 고정 T1 cache 생성·episode rollout 연결과 학습/평가 budget 확정은 별도 선행 작업입니다.

## Colab 데이터 준비·FFT·27일 업무 인계 (2026-10-28)

유지 관리되는 노트북은 Git 소스·HF 입력과 설정 기반 CLI를 사용합니다. [현재 실행 안내](docs/environment/research_runs.md)를 우선합니다.

[통합 Colab 노트북](experiments/colab_vlade.ipynb) 하나에서 `PROFILE`을 선택합니다.

| 작업 | PROFILE |
|---|---|
| HF 입력·단일 Episode·전체 프레임 준비 | `prepare_inputs`, `prepare_one_episode`, `fullframes` |
| AVA FFT·Suite-balanced·논문 LoRA | `ava_fft`, `ava_suite`, `ava_paper` |
| OFT FFT·AVA 한 step 재로딩 | `oft_fft`, `ava_one_step` |
| Teacher 선택·순차 cache | `select_teacher`, `cache_teacher` |
| Qwen Task-only·Action KD | `qwen_task`, `qwen_kd` |
| LIBERO+·원본 LIBERO 평가·단일 추론 | `evaluate_*`, `inference_ava` |

소스·인증 → 작업 선택 → 계획 확인 → 환경 설치 → 실행 순서입니다.
설치·실행은 기본 False이며, 작업별 run group과 모델별 환경은 분리됩니다.
Suite pilot4/main은 같은 실험 ID를 사용하고, 재개는 최초 run group을 명시합니다.

Colab에서 사용하는 코드와 노트북은 아래 위치의 일반 파일을 정본으로 관리합니다.

| 위치 | 용도 |
|---|---|
| `src/vlade/data/`, `src/vlade/training/` | 다운로드·episode 데이터·FFT·cache·Student 학습의 실제 구현 |
| `scripts/data/`, `scripts/train/` | 위 구현을 실행하는 명령과 checkpoint 재로딩 검사 |
| `experiments/colab_vlade.ipynb` | 데이터 준비·모델별 학습·재로딩 점검 노트북 |
| `configs/` | 모델·데이터·학습 조건과 고정 revision |
| `experiments/baselines/`, `experiments/manifests/` | 비교 기준과 원본 소스 기록, Colab Git 소스 배포 설정 |
| `docs/`, 작업별 `.harness/v1/.../checks/` | 실행 안내와 에이전트 검증 |

노트북과 `experiments/manifests/colab_source.json`은 저장소 루트에서 아래 명령으로 갱신합니다.

```bash
python scripts/prepare/generate_colab_notebooks.py
```

생성 구현은 `src/vlade/utils/colab_notebooks.py`에 있습니다. 노트북 시작 셀은 Git ref를
불변 commit으로 확정하고 해당 프로젝트 소스만 런타임에 준비합니다. 소스 파일별
SHA256 목록은 관리하지 않습니다. `experiments/manifests/colab_source.json`은 Git 저장소/ref
설정입니다. 로컬 변경은 먼저 commit·push하고, 재개 시 이전 `SOURCE_COMMIT`을 지정하십시오.

Git checkout에는 baseline·원본 소스 기록과 고정 AVA/OFT upstream을 포함합니다.
두 모델은 서로 다른 Transformers 환경에서 실행하지만 D/content-group 80/10/10 seed 42
canonical manifest와 확정 Train 전용 normalization을 공유합니다. Train/Validation/Test는
11,477/1,428/1,442 Episode이며 Calibration 728개는 Train 내부 subset입니다.
실행·분할 정본·AVA zero-context FFT 제한은 [Colab 인계 문서](docs/environment/colab_kanban_handoff.md)에
정리했습니다. 일반 파일 전환 후 전체 CPU 회귀 검사 **127개 통과(40.24초)**와 실제 GPU 미검증 범위를 구분하며,
CPU 검증을 실제 T1 선정/cache·Task-only 성능 결과로 해석하지 않습니다.

Colab 본 학습은 실제 train query 수×설정 epoch로 예산을 계산하며 기본 자동 실행을 끕니다.
공통 episode shuffle/query cursor 재개, 주기 validation, 첫 gradient 검사, 파일럿 실행 조건,
checkpoint 공간 검사·bounded debug와 T1 비교 보고서 검증을 연결했습니다.
기본 1 epoch·overfit 감소 기준 등 최종 실험 선택은 검토 대상이며 실제 GPU 성공을 뜻하지 않습니다.

추가 교차 검토 후 split manifest 정합성, checkpoint 파일·step 봉인, cache source/feature 대조,
파일럿 보고서 identity와 기존 T1 선택 충돌 검사, 소유한 추출 임시 파일의 재시작도 보완했습니다.

AVA 전용 1-step 노트북은 전체 Episode 전처리와 본 학습을 수행하지 않습니다. 원본 TFDS를
순서대로 읽다가 첫 canonical Train Episode를 확인한 즉시 중단하고 sealed full-Train
normalization을 그대로 사용합니다. A100 실측값은 Colab에서 실제 실행한 뒤 생성되는
`reload_report.json`만 근거로 삼으며, 이 저장소 변경 자체는 GPU 통과를 뜻하지 않습니다.

전처리 진입점과 첨부 AVA Suite-balanced FFT·paired rollout 노트북은
[전처리 및 노트북 안내](docs/environment/libero_plus_preprocessing_handoff.md)를 참고하십시오.


## Hugging Face 자료 참조

데이터·가중치 입력은 `hfasset://키`를 사용하고, `src/vlade/storage/`가 HF
레지스트리의 고정 commit과 SHA256을 확인하여 로컬 캐시에 준비합니다. 학습 진입점은 HF 키 또는 검증된 HF 캐시만 외부 입력으로 받습니다. 로컬·Colab·GitHub는
같은 참조 설정을 사용합니다. `configs/storage/hf_assets.lock.json`과
`scripts/data/hf_assets.py`가 입력 버전과 준비를 관리합니다. 기존 Drive 자료의 승인된 이관은
`migrate_hf_assets.py`로 수행합니다. 통합 노트북은 고정 HF 입력을 사용하며 현재 실행 출력은
기본 `/content/vlade_work/runs`에 저장하고 봉인된
중간·best·final checkpoint를 HF에 전송합니다. 원본 Drive 자료는 자동 삭제하지 않습니다.
자세한 인증·이관·재개 방법은 [HF 안내](docs/environment/hugging_face_checkpoints.md)를 참조하십시오.
