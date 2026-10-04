# Colab A100 단일 추론 스모크 테스트 · 2026-10-04

두 공개 체크포인트 모두 로드 및 유한한 8×7 액션 생성 검사를 통과했습니다.

| 모델 | 결과 파일 | PyTorch 최대 할당 GPU 메모리 | 모델 로드 + 추론 시간 |
| --- | --- | --- | --- |
| OpenVLA-OFT | [openvla_oft_20261004T055453Z.json](openvla_oft_20261004T055453Z.json) | 14.97 GiB | 236.03초 |
| AVA-VLA | [ava_vla_20261004T060317Z.json](ava_vla_20261004T060317Z.json) | 14.82 GiB | 254.64초 |

NVIDIA A100-SXM4-40GB(39.5 GiB), BF16, seed 7, batch 1, warm-up 0회,
검은색 256×256 RGB 이미지 2장, 0으로 채운 8차원 상태를 사용했습니다.
지시문은 `pick up the object`, normalization key는 `libero_spatial_no_noops`입니다.
AVA는 `temporal_context=None`인 첫 시점 경로를 사용했습니다.

시간은 Drive 읽기와 모델 초기화를 포함한 전체 벽시계 시간이며 추론 지연시간
벤치마크가 아닙니다. 메모리는 PyTorch 최대 할당량입니다. 소스·가중치 revision,
실제 패키지 버전 및 dependency Git commit은 각 JSON에 보존했습니다.

OFT 첫 import 실패는 `tensorflow-metadata==1.15.0` 고정으로 해결했습니다.
이미지 전처리용 TensorFlow는 CPU로 제한하고 모델별 Python 환경을 분리했습니다.

- [실행한 Colab 노트북](https://colab.research.google.com/drive/1KEXpaFTeq7a1dKgAN601D_eKWdDKFN-Z)
- [Notion 결과](https://app.notion.com/p/3dcb5dd0687080199b5be414131b8f87)
- [로컬 노트북](../../colab_inference_smoke.ipynb)

결과는 Drive와 로컬에 저장했습니다. Colab 런타임 연결 해제 및 삭제 후 활성 세션
없음을 확인했습니다(2026-10-04 15:05 KST 확인). LIBERO(+) 성공률, FFT
forward/backward/save/load, 실제 로봇 및 AVA 다음 시점 temporal-context는 미검증입니다.
