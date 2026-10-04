# 로컬 자료 준비와 Colab A100 이관

2026-10-04 기준, AVA-VLA와 OpenVLA-OFT 공개 가중치는 로컬에 내려받아
SHA-256까지 검증했습니다. LIBERO+ 평가 assets와 RLDS 학습자료는 사용자 요청에
따라 로컬에서 삭제했으며, 필요하면 Colab에서 직접 다시 받습니다. Drive 이관과
Colab A100 단일 추론은 완료했습니다. 두 모델 모두 유한한 8×7 액션을 생성했으며,
FFT 학습과 benchmark 평가는 아직 검증하지 않았습니다.

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
  실행했습니다. NVIDIA A100-SXM4-40GB(39.5 GiB)에서 두 모델 모두 통과했고,
  결과 JSON과 패키지 버전은 Drive와 로컬의 `experiments/runs/inference_smoke/`에
  저장했습니다. OFT/AVA의 PyTorch 최대 할당 GPU 메모리는 각각 14.97/14.82 GiB입니다.
  검사 후 런타임을 종료하고 활성 세션 없음까지 확인했습니다.
- 두 원본은 서로 다른 커스텀 Transformers 포크를 요구하므로 Colab에서 별도
  Python 환경 또는 순차적인 의존성 설치가 필요합니다.
- 두 원본의 `vla-scripts/finetune.py`는 LoRA만 지원합니다. 요청한 Full Fine-Tuning은
  VLADE 측 `scripts/train/run_fft.py`로 분리했습니다. 사용법과 제한은
  [`libero_plus_fft_colab.md`](libero_plus_fft_colab.md)를 참고하십시오. A100의
  이번 추론 런타임의 실제 VRAM은 39.5 GiB였습니다. FFT 필요 메모리는 별도
  검증해야 합니다.

## 2026-10-04 단일 추론 실행 기록

- Drive 프로젝트 경로: `/content/drive/MyDrive/colab/VLADE/VLADE`.
- Python 3.11.16, PyTorch 2.2.0+cu121, CUDA 12.1, 모델별 Transformers 4.40.1 포크.
- TensorFlow Metadata를 `1.15.0`으로 고정해 protobuf import 오류를 해결했습니다.
  TensorFlow 이미지 전처리는 CPU, 모델 추론은 BF16 A100 GPU를 사용했습니다.
- 검은색 256×256 RGB 이미지 2장과 0으로 채운 8차원 상태, `pick up the object`,
  `libero_spatial_no_noops`, seed 7, batch 1, warm-up 0회, 모델별 추론 1회입니다.
- 결과: `openvla_oft_20261004T055453Z.json`, `ava_vla_20261004T060317Z.json`.
  소스·가중치 revision과 실제 패키지 Git commit은 각 JSON에 기록했습니다.
- [실행 노트북](https://colab.research.google.com/drive/1KEXpaFTeq7a1dKgAN601D_eKWdDKFN-Z),
  [Notion 결과](https://app.notion.com/p/3dcb5dd0687080199b5be414131b8f87).
