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
├─ scripts/                 # 사용자가 실행하는 진입점
├─ benchmarks/              # LIBERO, LIBERO+, CALVIN 연결 계층
├─ experiments/             # baseline, ablation, run manifest
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
- benchmark별 observation/action 차이는 `benchmarks/` 또는 `data/adapters/`에서만 흡수합니다.
- Teacher와 Student가 달라도 학습 코드는 공통 인터페이스만 보도록 구성합니다.
- quantization, pruning 등의 압축 확장은 증류 baseline이 고정된 뒤 별도 설정으로 추가합니다.
- 논문 보고값과 자체 재현값을 같은 결과로 덮어쓰지 않습니다.
