# VLADE

**Vision-Language-Action Distillation for Edge Deployment**

VLADE는 대규모 Vision-Language-Action(VLA) 로봇 제어 정책의 시각·언어 이해와 조작 능력을 경량 Student 모델에 전달하여, 제한된 연산 자원을 가진 엣지 디바이스에서도 높은 작업 성공률과 낮은 제어 지연시간을 함께 달성하는 것을 목표로 하는 연구 프로젝트입니다.

핵심 연구 범위는 action distillation, representation distillation, task supervision이며, LIBERO 계열과 CALVIN을 초기 벤치마크로 사용하고 Jetson Orin Nano급 환경에서 모델 크기, VRAM, 추론 지연시간과 제어 주기를 평가할 계획입니다.

기반 Teacher 및 baseline으로 AVA-VLA를 연결했습니다. 사용자 포크
`eastha10/AVA-VLA`의 원본 소스는 `third_party/ava_vla/` Git submodule로 관리하고,
VLADE 자체 구현은 `src/vlade/`에 분리합니다.

현재 단계는 원본 소스 이양과 설정 기록입니다. Teacher 어댑터, Student 및 증류 학습
코드는 아직 구현하지 않았으며, Hugging Face 가중치 다운로드와 실행 검증은 다음
단계에서 진행합니다. 연결 방법과 검토할 가중치는
[`docs/environment/ava_vla.md`](docs/environment/ava_vla.md)에 정리했습니다.

## 소스 받기

```bash
git clone --recurse-submodules https://github.com/eastha10/VLADE.git
cd VLADE
```

이미 clone한 저장소는 아래 명령으로 기록된 AVA-VLA 커밋을 받습니다.

```bash
git submodule update --init --recursive
```

Teacher 연결 설정은 `configs/models/teacher/ava_vla.json`, 원본·의존성 커밋과
검증 상태는 `experiments/manifests/ava_vla_source.json`에서 관리합니다.

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
3. 공식 baseline을 재현해 `experiments/baselines/`에 조건을 고정합니다.
4. action KD와 representation KD를 `src/vlade/distillation/`에서 한 요소씩 추가합니다.
5. LIBERO 계열과 CALVIN 평가는 `benchmarks/`와 `src/vlade/evaluation/`을 통해 수행합니다.
6. 최종 후보를 Jetson/TensorRT 경로로 옮겨 지연시간과 메모리를 측정합니다.

자세한 디렉터리 역할은 `STRUCTURE.md`를 참고하십시오.
