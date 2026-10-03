# 로컬 자료 준비와 Colab A100 이관

2026-10-04 기준, AVA-VLA와 OpenVLA-OFT 공개 가중치는 로컬에 내려받아
SHA-256까지 검증했습니다. LIBERO+ 평가 assets와 RLDS 학습자료는 사용자 요청에
따라 로컬에서 삭제했으며, 필요하면 Colab에서 직접 다시 받습니다. Drive 이관,
Colab A100 추론 및 FFT 학습은 아직 실행하지 않았습니다.

## 로컬 자료 경로

| 자료 | 로컬 경로 | 상태 | 출처 |
| --- | --- | --- | --- |
| AVA-VLA LIBERO 4-in-1 | `checkpoints/teacher/avavla-libero-4in1/` | 로컬 검증 완료; Git 제외 | [LiAuto-DSR/avavla-libero-4in1](https://huggingface.co/LiAuto-DSR/avavla-libero-4in1) |
| OpenVLA-OFT LIBERO 4-suite | `checkpoints/baselines/openvla_oft/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/` | 로컬 검증 완료; Git 제외 | [moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10) |
| LIBERO+ 평가 assets | 없음 | 로컬 삭제; 평가할 때 Colab에서 다시 받기 | [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus) |
| LIBERO+ RLDS 학습자료 | 없음 | 로컬 삭제; FFT 학습할 때 Colab에서 다시 받기 | [Sylvest/libero_plus_rlds](https://huggingface.co/datasets/Sylvest/libero_plus_rlds) |

각 출처의 revision은 `configs/experiments/libero_plus_fft.json`에 고정했습니다.
이번 OpenVLA-OFT 다운로드는 LIBERO+ FFT의 초기 가중치로 사용할 4-suite
통합 체크포인트입니다. `experiments/baselines/openvla_oft/libero.json`에 등록된
원본 LIBERO의 suite별 네 체크포인트까지 받은 것은 아니며, 그 재현 여부와
LIBERO+ 주 실험을 구분합니다. 이전 로컬 다운로드 로그와 미완성 분할 ZIP도
사용자 요청에 따라 삭제했습니다. `scripts/setup/stage_public_assets.py`는 남아
있지만 자동 다운로드·재시작 작업은 중지했습니다.

## 이관 및 학습 전 확인

- Drive 업로드 대상은 VLADE 소스와 검증된 가중치입니다. 업로드 전에
  `.env`, 인증정보, 일시적 캐시와 미완료 `.part` 파일을 제외할 범위를 점검합니다.
- FFT에 필요한 RLDS 자료와 LIBERO+ 평가 assets는 Colab에서 별도로 받을 때
  압축 해제 경로와 필요 저장공간을 확인해야 합니다. VLADE FFT 코드와 데이터 이름
  등록은 추가했지만, 실제 압축 해제·TFDS/GPU 검증은 아직 수행하지 않았습니다.
- 데이터셋이나 평가 assets가 없는 단일 입력 추론 점검은
  [`experiments/colab_inference_smoke.ipynb`](../../experiments/colab_inference_smoke.ipynb)에
  준비했습니다. 노트북의 문법과 로컬 파일 경로만 확인했으며 GPU 실행은 하지 않았습니다.
- 두 원본은 서로 다른 커스텀 Transformers 포크를 요구하므로 Colab에서 별도
  Python 환경 또는 순차적인 의존성 설치가 필요합니다.
- 두 원본의 `vla-scripts/finetune.py`는 LoRA만 지원합니다. 요청한 Full Fine-Tuning은
  VLADE 측 `scripts/train/run_fft.py`로 분리했습니다. 사용법과 제한은
  [`libero_plus_fft_colab.md`](libero_plus_fft_colab.md)를 참고하십시오. A100의
  실제 VRAM 용량은 Colab 런타임에서 확인해야 합니다.
