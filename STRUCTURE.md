# Directory Structure

```text
vlade/
├─ configs/                 # 데이터·모델·증류·실험·배포 설정
├─ src/vlade/               # VLADE 자체 구현
│  ├─ data/                 # 입력 스키마, 전처리, 데이터 어댑터
│  ├─ models/               # Teacher/Student 래퍼와 공통 구성요소
│  ├─ distillation/         # action/representation KD와 정렬 로직
│  ├─ training/             # 학습 루프와 checkpoint 정책
│  ├─ evaluation/           # 성공률·일반화·효율 평가
│  ├─ deployment/           # Jetson 및 TensorRT 경로
│  └─ utils/                # 로깅, seed, 공통 유틸리티
├─ scripts/                 # 사용자가 실행하는 얇은 진입점
│  ├─ data/                # 자료 준비·검증·T1 선택·cache 생성
│  ├─ train/               # FFT/Student 학습·smoke·재로딩 검사
│  └─ prepare/             # Colab 노트북·source manifest 갱신
├─ benchmarks/              # LIBERO, LIBERO+, CALVIN 연결 계층
├─ experiments/             # baseline, ablation, run manifest
│  ├─ baselines/            # 외부 원본별 실행 설정(원본 코드는 복제하지 않음)
│  ├─ ablations/            # 구성요소별 제거·교체 실험
│  ├─ manifests/            # 원본 소스 기록·colab_source.json의 파일 SHA/revision
│  ├─ colab_01_*.ipynb      # 데이터·AVA 가중치 준비
│  ├─ colab_02_*.ipynb      # AVA FFT·T1 선택·cache
│  ├─ colab_03_*.ipynb      # Qwen Task-only
│  ├─ colab_04_*.ipynb      # OpenVLA-OFT 독립 환경 FFT·재로딩
│  └─ runs/                 # 실제 실행별 기록
├─ tests/                   # unit, integration, smoke test
├─ docs/                    # 의사결정·프로토콜·환경·논문 메모
├─ notebooks/               # 탐색 전용 분석
├─ third_party/             # 외부 원본 소스 연결
│  ├─ ava_vla/              # eastha10/AVA-VLA Git submodule (커밋 고정)
│  └─ openvla_oft/          # moojink/openvla-oft Git submodule (커밋 고정)
├─ data/                    # 로컬 데이터와 Teacher cache (Git 제외)
├─ checkpoints/             # Teacher/Student 가중치 (Git 제외)
└─ artifacts/               # 로그·수치·그림·영상·프로파일 (Git 제외)
```

## 경계 규칙

- `third_party/` 내부 로직을 VLADE 핵심 구현으로 간주하지 않습니다.
- 외부 baseline 코드는 `experiments/baselines/`에 복제하지 않고, 고정된
  `third_party/` 원본과 실행 설정을 연결합니다.
- benchmark별 observation/action 차이는 `benchmarks/` 또는 `data/adapters/`에서만 흡수합니다.
- Teacher와 Student가 달라도 학습 코드는 공통 인터페이스만 보도록 구성합니다.
- quantization, pruning 등의 압축 확장은 증류 baseline이 고정된 뒤 별도 설정으로 추가합니다.
- 논문 보고값과 자체 재현값을 같은 결과로 덮어쓰지 않습니다.

## 학습 구현

- `src/vlade/training/loop.py`: 모델과 독립적인 optimizer step 루프·검증·학습 상태
- `src/vlade/training/fft.py`: 기존 AVA/OFT의 모델·RLDS·L1·checkpoint 어댑터
- `src/vlade/training/smoke.py`: 합성 CPU 액션 정책의 학습·저장·재개 검사
- `scripts/train/smoke_loop.py`: 합성 검사의 얇은 CLI
- `tests/unit/test_training_loop.py`: 누적·clipping·NaN·validation·재개 검증
- `docs/decisions/0003-backbone-review.md`: Student 후보 비교와 도입 조건

공통 루프의 loss callback에 향후 action/representation KD를 연결할 수 있습니다.
실제 가중치로 검증된 학습 경로는 아직 없으며 기존 FFT는 GPU 검증 전입니다.
Student 계약 구현은 아래 범위이며 KD trainer는 미구현입니다.

## Teacher 저장·Student·집필 준비

- `src/vlade/data/teacher_records.py`: 제한된 debug 기록과 고정 Teacher cache 저장·검증·로드
- `src/vlade/models/contracts.py`: 정규화 액션 `[B,8,7]` 및 horizon feature 공통 출력·mask loss
- `src/vlade/models/students/qwen.py`: 별도 Qwen3.5 VLM 연결·processor·연속 query action head
- `src/vlade/models/students/smoke.py`, `scripts/train/smoke_student.py`: native Qwen 축소 CPU 검증
- `configs/models/student/qwen3_5_0_8b.json`: 고정 Hub revision·입력/출력·환경 검증 범위
- `docs/paper/`: 집필용 outline·원문 근거와 실험 기록 템플릿, 측정 결과 자체는 아님
- `docs/environment/teacher_records_and_student.md`: 저장 스키마·Student 설계·실행 제한

`data/teacher_cache/`는 실제 Teacher cache, `checkpoints/students/`는 로컬 Student
자료, `artifacts/student/`는 합성 검증 출력에 사용합니다. 대용량 생성물은 Git에 넣지 않습니다.
현재 공식 processor 자료만 준비했으며 사전학습 Student 가중치는 사용자 요청으로 보류했습니다.

## Colab 실행 연결 (2026-10-28)

- `experiments/colab_01_*.ipynb` → `colab_02_*.ipynb` / `colab_04_*.ipynb`: 공통 데이터 준비 후 AVA와 OFT를 독립 환경에서 실행합니다. `colab_03_*.ipynb`는 별도 Qwen Task-only 단계입니다.
- `src/vlade/data/colab_assets.py`, `episode_data.py`: Drive 자료 준비·공통 episode split/query/정규화.
- `src/vlade/data/canonical_split.py`, `canonical_normalization.py`: 확정 D/seed-42 manifest 검증·exact group 배정·Train 전용 통계.
- `src/vlade/training/prepared_fft.py`, `fixed_cache.py`, `student_task.py`: canonical 데이터 어댑터·순차 Teacher cache·Student BC.
- `scripts/data/`, `scripts/train/`: 해당 기능의 얇은 CLI와 checkpoint 재로딩 검사.
- `src/vlade/utils/colab_notebooks.py`: 현재 일반 소스 파일을 바탕으로 순서별 노트북과 파일별 SHA/revision manifest 생성.
- `scripts/prepare/generate_colab_notebooks.py`: 저장소 루트에서 인자 없이 실행하는 생성 진입점.
- `experiments/manifests/colab_source.json`: Colab에서 사용할 일반 파일 목록·SHA256·원본 revision을 확인하는 기준.
- `docs/environment/colab_kanban_handoff.md`: 순서·재개·미검토 목록과 실제 실행 전제.

핵심 데이터·학습 로직은 `src/vlade/`에 있으며 노트북은 실행 순서와 사용자 경로 설정을 제공합니다.

정본은 위 `src/`, `scripts/`, `experiments/`, `configs/`, `docs/`, `tests/`에 유지합니다.
`experiments/baselines/`와 원본 소스 manifests도 Colab 파일 목록에 포함합니다. 실제 upstream
복사 대상은 고정된 `third_party/ava_vla/`와 `third_party/openvla_oft/`이며 실행 환경은 별도입니다.

Drive에서는 `/content/drive/MyDrive/VLADE` 바로 아래에 저장소 구조를 둡니다. 노트북의
`ROOT`와 `PROJECT_ROOT`가 이 경로를 가리키며, manifest와 파일 SHA를 확인한 일반 소스만
`/content`의 실행 디렉터리로 복사합니다. `data/`, `checkpoints/`, `artifacts/`는 각각 실제
데이터·가중치·실행 결과를 보관합니다. 실행 결과를 수정할 소스 파일의 정본으로 사용하지 않습니다.

- `src/vlade/training/run_control.py`: epoch/query budget, 파일럿 통과 조건, gradient·checkpoint 공간·T1 보고서 검사.
- `src/vlade/training/debug_records.py`: 정확한 저장 checkpoint hash에 연결된 bounded 학습 debug 기록.
- `src/vlade/utils/filesystem.py`: 같은 parent의 완료 staging을 덮어쓰기 없이 게시하고 일시적인 잠금만 제한 재시도.
- `tests/unit/test_final_{data,cache,run}_review.py`: split/part/checkpoint/report/추출 재시작의 변조·실패 회귀 검사.
- `tests/unit/test_colab_source_files.py`: 일반 소스 파일 manifest·복사·누락/변조 거부와 revision 복원 검사.
