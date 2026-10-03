# LIBERO+ FFT 실행 경로 (Colab A100)

VLADE의 `scripts/train/run_fft.py`는 upstream의 LoRA 전용 학습 스크립트를
수정하지 않고 OpenVLA-OFT와 AVA-VLA의 **전체 VLA 가중치**를 학습합니다.
연속 행동 헤드와 proprio projector도 학습하며, AVA에서는 공개된
`vision_attn_weight_generator`를 초기화·학습·저장합니다. checkpoint는
`save_pretrained` 형식의 전체 VLA와 구성요소 `.pt`를 함께 보관합니다.

이 코드는 CPU 단위 테스트를 통과했지만, 아직 두 모델의 Colab GPU 학습을
실행하거나 LIBERO+ 성공률을 검증하지 않았습니다. 특히 AVA의 RLDS 학습 샘플은
이전 시점의 행동 특징을 제공하지 않습니다. 현재 FFT는 공개 AVA generator의
zero-context 경로를 학습하며, 에피소드 간 temporal context를 활용하는 원본
평가와 동등하다고 주장할 수 없습니다.

## 데이터 준비

LIBERO+ RLDS 자료는 로컬에서 사용자 요청으로 삭제했습니다. FFT를 진행할 때는
Colab에서 공식 분할 ZIP 3개를 직접 받은 다음
`data/processed/libero_plus_rlds/libero_plus_mixdata/` 디렉터리가 생기도록
압축 해제합니다. ZIP 내부 구조는 다운로드 완료 후 확인해야 합니다. 데이터가
해당 경로와 다르게 풀리면 JSON 설정의 `fft_defaults.data_root`를 바꾸십시오.
실행 전 압축 해제된 TFDS의 `dataset_info.json`과 shards가 있는지 확인하십시오.
원본 평가 assets는 학습 입력이 아니므로 이 단계에서 설치할 필요가 없습니다.

## 실행

두 upstream은 서로 다른 Transformers 포크를 사용합니다. 동일한 Python 런타임에
둘을 함께 설치하지 말고 모델별로 별도 Colab 런타임을 사용하십시오. 선택한
`third_party/<model>/pyproject.toml`의 의존성을 설치하고 `bitsandbytes`를
추가합니다. Colab의 Python·CUDA 버전과 해당 핀의 호환성을 먼저 확인해야
합니다. 아래 명령은 VLADE 루트에서 실행합니다.

```bash
python scripts/train/run_fft.py --model ava_vla --preflight
python scripts/train/run_fft.py --model ava_vla
```

별도 런타임에서 OpenVLA-OFT를 실행할 때는 `--model openvla_oft`로 바꿉니다.
실험값은 `configs/experiments/libero_plus_fft.json`의 `fft_defaults`와
`fft_runs`에서 조정합니다. `--max-steps 1 --save-steps 1`로 먼저 단일 step
smoke test를 수행한 뒤 본 학습으로 확장하십시오. 장시간 GPU 실행은 별도
승인이 필요합니다.

기본 설정은 BF16, batch 1, gradient accumulation 8, gradient checkpointing,
paged 8-bit AdamW입니다. 모두 전체 파라미터에 gradient를 계산하므로 LoRA가
아닙니다. 그러나 A100 40GB에서도 메모리 부족 가능성이 있으며, 80GB에서도
데이터/환경에 따라 다릅니다. 실제 GPU VRAM과 첫 step 최대 사용량을 확인해
설정을 확정하십시오. 체크포인트가 수십 GB가 될 수 있으므로 Colab 저장공간과
Drive 용량도 확인해야 합니다.

기본 체크포인트는 `artifacts/training/libero_plus_fft/<model>/step-XXXXXXX/`에
저장합니다. `COMPLETE` 파일이 있는 디렉터리만 완료된 저장으로 취급합니다.
기본값 `save_optimizer_state=false`에서는 **정확한 optimizer 상태 재개를
지원하지 않습니다**. 설정을 켜면 optimizer 상태도 저장하지만 재개 로더는
아직 없습니다. FFT 결과를 원본 LIBERO 재현이나 논문 결과와 혼동하지 마십시오.

## 남은 검증

- Colab에서 RLDS 분할 ZIP 다운로드·SHA-256 검증 및 압축 해제
- 실제 TFDS schema가 upstream LIBERO 변환과 맞는지 단일 batch 검사
- AVA·OFT 각각 Colab A100 단일 step forward/backward/save/load smoke test
- VRAM 사용량, 학습 시간, 체크포인트 크기와 평가 연결 검증
- AVA의 이전 시점 행동 특징을 사용하는 temporal-context 학습 확장
