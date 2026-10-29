# Directory Structure

```text
vlade/
├─ .harness/
│  ├─ v0/                  # 과거 기록 보관본 (새 기록 금지)
│  └─ v1/                  # 활성 작업 기록·작업별 scratch (사용 후 정리)
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
│  └─ runs/                 # 실제 실행별 기록
├─ tests/                   # unit, integration, smoke test
├─ docs/                    # 의사결정·프로토콜·환경·논문 메모
├─ notebooks/               # 탐색 전용 분석
├─ third_party/             # 외부 원본 소스 연결
│  ├─ ava_vla/              # eastha10/AVA-VLA Git submodule (커밋 고정)
│  └─ openvla_oft/          # moojink/openvla-oft Git submodule (커밋 고정)
├─ data/                    # 설치된 데이터셋·전처리 데이터·Teacher cache (보존, Git 제외)
│  ├─ raw/                  # 출처 그대로 받은 데이터·분할 압축본
│  ├─ processed/            # 해제된 데이터·전처리 episode와 그 manifest
│  ├─ splits/               # 별도로 관리하는 데이터 분할 정의 (필요한 경우만)
│  └─ teacher_cache/        # 고정 Teacher의 학습용 출력과 provenance
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

## `data/` 디렉터리 운영 규약

### 보관 범위와 정본

`data/`는 실제 학습·평가에 사용하는 데이터의 보관 위치입니다. 로컬과 Google Drive의
`colab/VLADE/data/`에 동일한 보존 원칙을 적용합니다. 폴더 이름이나 수정 시각만으로
사용하지 않는 데이터라고 판단하지 않습니다. 아래 경로는 역할을 설명하며 빈 폴더를
미리 만들거나 설치된 자료를 이 구조로 재배치하라는 지시가 아닙니다.

| 위치 | 보관할 내용 | 관리 기준 |
|---|---|---|
| `data/raw/` | 원본 RLDS 자료, benchmark assets, 다운로드 압축본·분할 ZIP, 다운로드 manifest | 출처·revision·파일 hash를 보존합니다. 원본을 직접 수정하거나 압축 해제 후 임의로 삭제하지 않습니다. |
| `data/processed/` | 압축 해제된 TFDS, 전처리 episode·이미지·trajectory·query, 해당 데이터의 manifest·분할·정규화 통계 | 준비 절차와 설정이 식별되는 경로를 사용합니다. 기존 완료 데이터는 재사용하며 새 전처리 결과로 덮어쓰지 않습니다. |
| `data/splits/` | 실제로 별도 관리할 필요가 있는 train/validation/test 분할 정의 | 현재 episode 데이터의 분할 정본은 해당 `processed` 데이터 폴더의 `splits.json`입니다. 이를 다른 위치에 복제하여 서로 다른 정본을 만들지 않습니다. |
| `data/teacher_cache/` | 고정 Teacher에서 생성한 학습용 action·feature 등의 출력, cache manifest·provenance | Teacher checkpoint hash·데이터 revision·분할·정규화 통계와 연결합니다. 이름에 cache가 있어도 삭제 가능한 임시 파일로 취급하지 않습니다. |

- 현재 Colab 준비 절차의 경로 예시는 `data/raw/libero_plus_rlds/`,
  `data/processed/libero_plus_tfds/`, `data/processed/libero_plus_episode80/`입니다.
  실제 입력 경로는 사용하는 설정·노트북에서 확인하며, 다른 유효한 기존 경로를 임의로 바꾸지 않습니다.
- 데이터의 정체성은 출처·dataset revision·원본 hash·전처리 코드와 설정·seed로 식별합니다.
  `manifest.json`, `splits.json`, `normalization.json`과 완료 표시는 해당 데이터와 함께 보존합니다.
  현재 episode 준비 형식에서는 `COMPLETE`와 manifest·분할·정규화 hash의 일치를 확인합니다.
  완료 표시만 있거나 일부 파일만 남은 상태를 정상 데이터로 간주하지 않습니다.
- 분할과 정규화 통계를 임의로 다시 계산하지 않습니다. 현재 정규화 통계는 train 분할로만
  계산하며 validation/test와 섞지 않습니다. seed·분할·전처리 조건을 변경한 데이터는 별도 버전으로 식별합니다.
- 모델 가중치는 `checkpoints/`, 공식 학습·평가 결과는 지정된 실험 결과 경로에 둡니다.
  에이전트의 합성 테스트 데이터·임시 checkpoint·검토 메모·패키지는
  `.harness/v1/YYYY-MM-DD-작업명/scratch/` 등 AGENTS.md의 작업 경로를 사용합니다.
  `data/`에 작업 기록이나 테스트 부산물을 남기지 않습니다.

### 생성·변경·삭제 기준

- 기본 작업은 기존 자료의 조회·검증·재사용입니다. 데이터 생성·다운로드·전처리는 사용자가
  요청했거나 승인한 데이터 준비 작업의 범위 안에서만 수행합니다.
- 코드 수정·문서 정리·용량 확보·Git 푸시·Drive 갱신은 기존 데이터의 삭제·덮어쓰기·이동·
  이름 변경·재다운로드를 승인한 것으로 해석하지 않습니다. 원본·전처리 결과·Teacher cache와
  각 manifest를 포함해 보존합니다. 삭제가 필요하면 대상과 영향을 구체적으로 확인받습니다.
- 자료가 없거나 검증에 실패하면 경로와 오류를 보고합니다. 자동 삭제·자동 재설치로 해결하거나
  정상 데이터에 불완전한 결과를 병합하지 않습니다. 승인된 재생성이 필요하면 별도 출력 경로를 사용합니다.
- 데이터 준비 중의 미완료 파일은 완료 자료와 구분합니다. 승인된 준비 작업의 재시작·정리에서는
  해당 작업이 만든 파일임을 식별하고 절대 경계 경로를 확인한 임시 파일만 처리합니다.
  다른 작업의 파일·실행 중인 다운로드·소유자가 불명확한 부분 결과는 건드리지 않습니다.
- 재생성 가능한 결합 ZIP·일회성 테스트 파일도 목적과 소유 범위가 분명한 임시 경로를 사용하며,
  설치된 데이터셋을 `.harness/`나 작업용 복제본으로 복사하지 않습니다.

### Git과 Google Drive 갱신

- 데이터 본문·압축본·전처리 파일·Teacher cache는 Git에 추가하지 않습니다. 기존 `.gitignore`의
  `data/**` 제외를 유지합니다. 재현에 필요한 작은 설정·출처·버전 기록은 `configs/`,
  `experiments/manifests/`, `docs/`의 정식 파일에서 데이터 경로와 식별자를 참조합니다.
  데이터와 묶인 manifest를 Git에 넣기 위해 원래 위치에서 옮기거나 별도 정본으로 만들지 않습니다.
- Git 관리 파일 변경과 commit·push·merge에 따른 Drive 동기화에서는 `data/` 전체를 갱신 대상에서 제외합니다.
  `checkpoints/`, `hf_cache/`와 다른 위치에 설치된 데이터셋·다운로드 자료도 같은 원칙으로 보존합니다.
- 갱신할 소스·설정·문서·노트북·소스 manifest만 명시적으로 선택합니다. 프로젝트 폴더 전체 교체,
  삭제 전파, 폴더 미러링을 사용하지 않습니다. 로컬에 데이터가 없다는 이유로 Drive의 데이터를 삭제하지 않습니다.
- `experiments/manifests/colab_source.json`은 실행할 **소스 파일**의 hash 목록입니다.
  이를 갱신하는 작업은 `data/`의 데이터 manifest를 변경하거나 데이터셋을 다시 준비하는 작업과 구분합니다.

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
자료, 에이전트의 합성 검증 출력은 `.harness/v1/YYYY-MM-DD-작업명/scratch/outputs/`에 보관합니다. 대용량 생성물은 Git에 넣지 않습니다.
현재 공식 processor 자료만 준비했으며 사전학습 Student 가중치는 사용자 요청으로 보류했습니다.

## Colab 실행 연결 (2026-10-28)

- `experiments/colab_01_*.ipynb` → `colab_02_*.ipynb` → `colab_03_*.ipynb`: 다운로드·FFT/cache·Task-only 실행 순서.
- `src/vlade/data/colab_assets.py`, `episode_data.py`: Drive 자료 준비·공통 episode split/query/정규화.
- `src/vlade/training/prepared_fft.py`, `fixed_cache.py`, `student_task.py`: 데이터 어댑터·순차 Teacher cache·Student BC.
- `scripts/data/`, `scripts/train/`: 해당 기능의 얇은 CLI와 checkpoint 재로딩 검사.
- `src/vlade/utils/colab_notebooks.py`: 현재 일반 소스 파일을 바탕으로 순서별 노트북과 파일별 SHA/revision manifest 생성.
- `scripts/prepare/generate_colab_notebooks.py`: 저장소 루트에서 인자 없이 실행하는 생성 진입점.
- `experiments/manifests/colab_source.json`: Colab에서 사용할 일반 파일 목록·SHA256·원본 revision을 확인하는 기준.
- `docs/environment/colab_kanban_handoff.md`: 순서·재개·미검토 목록과 실제 실행 전제.

핵심 데이터·학습 로직은 `src/vlade/`에 있으며 노트북은 실행 순서와 사용자 경로 설정을 제공합니다.

정본은 위 `src/`, `scripts/`, `experiments/`, `configs/`, `docs/`, `tests/`에 유지합니다.
`experiments/baselines/`와 원본 소스 manifests도 Colab 파일 목록에 포함합니다. 실제 upstream
복사 대상은 `third_party/ava_vla/`이며 OpenVLA-OFT 실행 환경은 별도입니다.

Drive에서는 `/content/drive/MyDrive/colab/VLADE` 바로 아래에 저장소 구조를 둡니다. 노트북의
`ROOT`와 `PROJECT_ROOT`가 이 경로를 가리키며, manifest와 파일 SHA를 확인한 일반 소스만
`/content`의 실행 디렉터리로 복사합니다. `data/`, `checkpoints/`, `artifacts/`는 각각 실제
데이터·가중치·실행 결과를 보관합니다. 실행 결과를 수정할 소스 파일의 정본으로 사용하지 않습니다.

- `src/vlade/training/run_control.py`: epoch/query budget, 파일럿 통과 조건, gradient·checkpoint 공간·T1 보고서 검사.
- `src/vlade/training/debug_records.py`: 정확한 저장 checkpoint hash에 연결된 bounded 학습 debug 기록.
- `src/vlade/utils/filesystem.py`: 같은 parent의 완료 staging을 덮어쓰기 없이 게시하고 일시적인 잠금만 제한 재시도.
- `tests/unit/test_final_{data,cache,run}_review.py`: split/part/checkpoint/report/추출 재시작의 변조·실패 회귀 검사.
- `tests/unit/test_colab_source_files.py`: 일반 소스 파일 manifest·복사·누락/변조 거부와 revision 복원 검사.
