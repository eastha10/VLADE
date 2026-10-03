# Baseline registry

VLADE의 외부 baseline 실행 조건을 모델별로 분리해 관리합니다. 외부 원본은
`third_party/`의 고정된 Git submodule에 그대로 두고, 이 디렉터리에는 VLADE에서
사용할 평가 설정만 둡니다.

여기의 `libero.json`은 원본 checkpoint 재현용입니다. VLADE의 주 비교 실험은
`configs/experiments/libero_plus_fft.json`에 정의한 LIBERO+ FFT이며, 원본 LIBERO
재현 결과와 별도로 관리합니다.

| Baseline | LIBERO 설정 | 원본 |
| --- | --- | --- |
| AVA-VLA | `ava_vla/libero.json` | `third_party/ava_vla/` |
| OpenVLA-OFT | `openvla_oft/libero.json` | `third_party/openvla_oft/` |

두 원본은 같은 `prismatic` import 이름을 사용하지만 서로 다른 Transformers 포크를
요구하므로 반드시 별도 Python 환경에 설치합니다. 설정의 `runtime.environment_name`은
권장 환경 식별자이며, 실행할 때 해당 환경을 먼저 활성화해야 합니다.

## 명령 확인

대용량 모델을 받거나 평가를 시작하지 않고 최종 명령만 확인합니다.

```bash
python scripts/evaluate/run_baseline.py ava_vla --task-suite libero_10 --dry-run
python scripts/evaluate/run_baseline.py openvla_oft --task-suite libero_10 --dry-run
```

기본 실행은 설정에 기록된 로컬 체크포인트 경로만 사용합니다. 체크포인트를 검토해
해당 경로에 준비한 뒤 `--dry-run`을 제거합니다. Hugging Face 식별자를 사용해 실행
중 다운로드하는 경우에만 `--use-hub-checkpoint`를 명시합니다.

```bash
python scripts/evaluate/run_baseline.py ava_vla --task-suite libero_10
python scripts/evaluate/run_baseline.py openvla_oft --task-suite libero_10
```

평가 로그는 `artifacts/logs/baselines/<baseline>/` 아래에 생성됩니다. 논문 보고값은
이 디렉터리의 자체 재현 결과로 기록하지 않습니다. 실제 실행 결과에는 checkpoint
revision, dataset revision, seed, 하드웨어와 소프트웨어 환경을 별도 run manifest로
남겨야 합니다.
