# QwenVLA S0 Task-only

S0의 이번 설정은 QwenVLA 전체를 LIBERO+ D/seed42 정답 액션으로 학습합니다. BF16 autocast, FP32 파라미터 및 AdamW, 배치 1, 누적 8, seed 42, 총 8,000 optimizer step을 사용합니다. 연속 Query는 동일 에피소드의 상태를 전달하고 각 Query 경계에서 detach합니다. 에피소드가 바뀌면 상태를 초기화합니다. 체크포인트에는 모델, processor, optimizer, scheduler, RNG, query cursor와 현재 상태를 저장합니다.

## Colab 파일럿

이 변경을 포함한 Git commit을 Colab에서 checkout하고, GPU 런타임에서 시작합니다. 전체 모델의 가중치, gradient, AdamW moment만 약 14.1 GiB입니다. RTX 5060 8GB는 전체 학습에 부족합니다. A100에서 먼저 실제 데이터를 사용한 파일럿을 권장합니다.

Colab Secrets의 HF_TOKEN을 해당 런타임에서 환경 변수로 전달합니다. 토큰은 비공개 prepared 데이터 읽기와 팀 체크포인트 저장소 쓰기 권한이 필요합니다.

[Colab 노트북](../../experiments/colab_qwen_vla_s0.ipynb)의 기본 설정은 32스텝 파일럿입니다. 명령줄로 실행할 때는 아래 명령을 사용합니다. 이미 검증된 데이터와 가중치 캐시는 재사용합니다.

```bash
export PYTHONPATH="$PWD/src"
export VLADE_ASSET_CACHE=/content/vlade_hf_assets
export VLADE_HUB_REPO=VLADE-konkuk/VLADE-checkpoints
export VLADE_HUB_RUN_ID=qwen-vla-s0-seed42-r1
python -m pip install "torch==2.12.0" "torchvision==0.27.0" "transformers==5.9.0"
python -m vlade.training.qwen_vla_task \
  --config configs/experiments/qwen_vla_s0.json \
  --output /content/vlade_runs/qwen-vla-s0-seed42 \
  --stop-after 32
```

10~30스텝 구간의 elapsed_seconds 차이를 optimizer step 수로 나눠 학습 시간을 추정합니다. 초기 로딩, 검증, checkpoint 해시 및 Hub 업로드 시간은 별도로 계산합니다. 파일럿의 step 32는 중간 체크포인트입니다. 같은 런타임에서 이어갈 경우 위 학습 명령의 --stop-after 32를 아래 옵션으로 바꿉니다.

```bash
--resume /content/vlade_runs/qwen-vla-s0-seed42/step-0000032
```

다른 런타임에서는 기존 Hub 안내에 따라 같은 run과 고정 revision의 체크포인트를 검증해 복구하고, 같은 Git commit·설정·run group·output 폴더명을 사용합니다. 8,000스텝 완료 후의 checkpoint를 BF16 및 양자화 비교의 공통 입력으로 사용합니다. 첫 실행에는 prepared 데이터 다운로드·해제 및 FP32 모델과 AdamW 상태 저장 공간이 필요합니다.
