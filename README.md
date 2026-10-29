# VLADE

**Vision-Language-Action Distillation for Edge Deployment**

VLADE는 대규모 Vision-Language-Action(VLA) 로봇 제어 정책의 시각·언어 이해와 조작 능력을 경량 Student 모델에 전달하여, 제한된 연산 자원을 가진 엣지 디바이스에서도 높은 작업 성공률과 낮은 제어 지연시간을 함께 달성하는 것을 목표로 하는 연구 프로젝트입니다.

핵심 연구 범위는 action distillation, representation distillation, task supervision이며,
주 비교 실험은 LIBERO+에서 Full Fine-Tuning(FFT)으로 수행합니다. 원본 LIBERO는
baseline 재현용으로 분리하고 CALVIN은 후속 일반화 벤치마크로 사용합니다. 최종
후보는 Jetson Orin Nano급 환경에서 모델 크기, VRAM, 추론 지연시간과 제어 주기를
평가할 계획입니다.

기반 Teacher로 AVA-VLA를 연결하고, AVA-VLA와 OpenVLA-OFT를 서로 독립된
baseline으로 등록했습니다. 사용자 포크 `eastha10/AVA-VLA`의 원본 소스는
`third_party/ava_vla/`, OpenVLA-OFT 원본 소스는 `third_party/openvla_oft/` Git
submodule로 관리합니다. VLADE 자체 구현은 `src/vlade/`에 분리합니다.

현재 단계는 baseline 등록, FFT 공통 학습 루프와 Teacher 출력 저장·Student 계약의 CPU 검증까지입니다.
실제 Colab GPU 학습은 아직 검증하지 않았습니다. Teacher 어댑터,
Qwen Student의 연속 action head와 별도 VLM 연결 계약은 구현했으며 실제 0.8B 가중치·증류 학습은 아직 검증하지 않았습니다. 두 공개 체크포인트의
A100 단일 추론은 통과했으며, 실제 benchmark 평가 결과는 아직 검증하지 않았습니다. baseline 사용법은
[`experiments/baselines/README.md`](experiments/baselines/README.md), AVA-VLA 연결
방법과 검토할 가중치는 [`docs/environment/ava_vla.md`](docs/environment/ava_vla.md)에
정리했습니다. 로컬 가중치 보관 상태와 Colab A100 이관 조건은
[`docs/environment/local_stage_to_colab.md`](docs/environment/local_stage_to_colab.md)에
기록했습니다.

FFT 실행 방법과 AVA temporal 학습의 제한은
[`docs/environment/libero_plus_fft_colab.md`](docs/environment/libero_plus_fft_colab.md)에
정리했습니다.

Colab A100에서 데이터셋 없이 두 공개 체크포인트의 단일 액션 추론을 점검할
노트북은 [`experiments/colab_inference_smoke.ipynb`](experiments/colab_inference_smoke.ipynb)에
보관합니다. 2026-10-04에 A100에서 두 모델 모두 유한한 8×7 액션 생성을
통과했습니다. 실행 조건과 결과 JSON은 `experiments/runs/inference_smoke/`에
보관했으며, Colab 런타임 종료와 활성 세션 없음까지 확인했습니다. 이 검사는
LIBERO(+) 성공률이나 FFT 학습 검증을 포함하지 않습니다.

## 소스 받기

```bash
git clone --recurse-submodules https://github.com/eastha10/VLADE.git
cd VLADE
```

이미 clone한 저장소는 아래 명령으로 기록된 외부 소스 커밋을 받습니다.

```bash
git submodule update --init --recursive
```

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

1. `third_party/`에 선정한 VLA 코드를 연결합니다.
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
- `tests/`: 단위, 통합과 smoke test를 관리합니다.
- `third_party/`: AVA-VLA, OpenVLA-OFT 등 외부 원본 코드를 별도로 관리합니다.

자세한 디렉터리 역할은 `STRUCTURE.md`를 참고하십시오.

## Training loop 보완 및 backbone 검토 (2026-10-24)

FFT 어댑터는 공통 `src/vlade/training/loop.py`를 사용합니다. 누적 학습,
clipping·비정상 값 검사, 선택적 LR 스케줄·val loss, JSONL 실행 기록,
완료 checkpoint의 optimizer/scheduler/RNG 재개를 지원합니다.
RLDS stream 위치는 복원하지 않으며 GPU FFT는 아직 검증하지 않았습니다.
CPU 검사는 `python -m pytest tests/unit -q` 및
`python scripts/train/smoke_loop.py --output-dir .harness/v1/2026-10-29-training-smoke/scratch/outputs/cpu-loop-smoke`로
수행합니다. 실행·재개 제한은 [학습 안내](docs/environment/libero_plus_fft_colab.md)에 있습니다.

기존 Teacher/baseline과 Llama-2-7B 가중치는 유지합니다. Student 후보인 Qwen3,
Qwen3.5, SmolVLM2/SmolVLA, Llama 3.2, Gemma 4의 호환성과 계산량 검토는
[backbone 검토](docs/decisions/0003-backbone-review.md)에 정리했습니다.
이는 후보 우선순위이며 Student 구현·모델 교체나 VLA 성능 검증을 의미하지 않습니다.

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

```bash
python scripts/train/smoke_student.py --output-dir .harness/v1/2026-10-29-student-smoke/scratch/outputs/tiny-random-check
```

실제 고정 T1 cache 생성·episode rollout 연결과 학습/평가 budget 확정은 별도 선행 작업입니다.

## Colab 데이터 준비·FFT·27일 업무 인계 (2026-10-28)

Drive를 공통 저장소로 사용하는 순서별 노트북을 추가했습니다.

1. [LIBERO+·AVA 가중치 다운로드/검증·episode 전처리](experiments/colab_01_download_data_weights.ipynb)
2. [AVA-VLA FFT·저장/재개·T1 선택·순차 cache](experiments/colab_02_ava_fft_and_fixed_cache.ipynb)
3. [Qwen Task-only·실제 가중치 로딩·overfit·저장/재개](experiments/colab_03_qwen_task_only.ipynb)

Colab에서 사용하는 코드와 노트북은 아래 위치의 일반 파일을 정본으로 관리합니다.

| 위치 | 용도 |
|---|---|
| `src/vlade/data/`, `src/vlade/training/` | 다운로드·episode 데이터·FFT·cache·Student 학습의 실제 구현 |
| `scripts/data/`, `scripts/train/` | 위 구현을 실행하는 명령과 checkpoint 재로딩 검사 |
| `experiments/colab_01_*.ipynb`부터 `colab_03_*.ipynb` | Colab에서 여는 순서별 실행 노트북 |
| `configs/` | 모델·데이터·학습 조건과 고정 revision |
| `experiments/baselines/`, `experiments/manifests/` | 비교 기준과 원본 소스 기록, Colab 파일별 SHA256 manifest |
| `docs/`, `tests/` | 실행·연구 검토 안내와 회귀 검사 |

노트북과 `experiments/manifests/colab_source.json`은 저장소 루트에서 아래 명령으로 갱신합니다.

```bash
python scripts/prepare/generate_colab_notebooks.py
```

생성 구현은 `src/vlade/utils/colab_notebooks.py`에 있습니다. 현재 작업 파일 트리를
Drive의 `/content/drive/MyDrive/colab/VLADE`에 같은 구조로 배치한 뒤 `experiments/`의 노트북을
Colab에서 여십시오. 노트북의 `ROOT`와 `PROJECT_ROOT`는 기본적으로 이 같은 경로를 가리키며,
그 바로 아래에 `src/`, `scripts/`, `configs/`, `third_party/`가 있어야 합니다.
시작 셀은 manifest와 각 파일의 SHA256을 확인하고 필요한 일반 파일을 `/content`로 복사합니다.
미커밋 변경도 사용할 파일 트리와 manifest에 포함되어야 합니다. GitHub HEAD만 받은 사본에는
현재 작업 내용이 빠질 수 있습니다.

Colab manifest에는 baseline·원본 소스 기록을 포함하지만, 이번 실행에서 사용하는 upstream
코드는 `third_party/ava_vla/`입니다. OpenVLA-OFT는 기존 baseline의 별도 환경에서 다룹니다.
실행·분할 정본·AVA zero-context FFT 제한은 [Colab 인계 문서](docs/environment/colab_kanban_handoff.md)에
정리했습니다. 일반 파일 전환 후 전체 CPU 회귀 검사 **127개 통과(40.24초)**와 실제 GPU 미검증 범위를 구분하며,
CPU 검증을 실제 T1 선정/cache·Task-only 성능 결과로 해석하지 않습니다.

Colab 본 학습은 실제 train query 수×설정 epoch로 예산을 계산하며 기본 자동 실행을 끕니다.
공통 episode shuffle/query cursor 재개, 주기 validation, 첫 gradient 검사, 파일럿 실행 조건,
checkpoint 공간 검사·bounded debug와 T1 비교 보고서 검증을 연결했습니다.
기본 1 epoch·overfit 감소 기준 등 최종 실험 선택은 검토 대상이며 실제 GPU 성공을 뜻하지 않습니다.

추가 교차 검토 후 split manifest 정합성, checkpoint 파일·step 봉인, cache source/feature 대조,
파일럿 보고서 identity와 기존 T1 선택 충돌 검사, 소유한 추출 임시 파일의 재시작도 보완했습니다.
