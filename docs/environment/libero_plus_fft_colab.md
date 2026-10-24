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
기본값 `save_optimizer_state=false`에서는 가중치만 저장합니다. 재개가 필요하면
`--save-training-state`를 사용하십시오. `trainer_state.pt`에 optimizer, scheduler,
Python/NumPy/PyTorch/CUDA RNG, 완료된 optimizer step과 microstep을 저장합니다.
이전 버전의 `optimizer.pt`만 있는 checkpoint는 자동 재개하지 않습니다.

## 보완한 Training loop (2026-10-24)

`src/vlade/training/loop.py`는 모델 종류와 독립적인 단일 디바이스 루프입니다.
FFT 어댑터가 모델·구성요소·데이터·loss·checkpoint callback을 제공합니다.
Student/KD에도 같은 루프를 사용할 수 있지만 실제 Student와 KD loss는 아직
연결하지 않았습니다. DDP/FSDP 및 FP16 GradScaler는 구현 범위에 포함하지 않았습니다.
기존 모델 어댑터는 BF16 CUDA를 사용하며 `device`로 GPU를 지정합니다.

- optimizer step 기준 gradient accumulation, LR schedule, 로그·평가·저장 주기를 사용합니다.
- 기존 constant LR를 기본으로 유지하고 선택적으로 warmup/cosine을 제공합니다.
- optimizer 업데이트 전에 유한한 scalar loss와 gradient를 확인하고 global norm을 clip합니다.
- 로그 loss는 전체 accumulation window의 평균이며 마지막 microbatch 값만 출력하지 않습니다.
- 데이터가 window 도중 끝나면 부분 gradient를 폐기하고 실패 처리합니다.
- 액션 target은 `[B,8,7]`, 각 샘플의 action token은 56개인지 검사합니다.
- upstream이 squeeze한 batch=1 proprio를 `[B,8]`로 복원하고 L1은 FP32로 계산합니다.
- 모든 모델 파라미터의 requires_grad를 확인합니다. L1에서 사용되지 않는 언어 출력
  head 등에는 gradient가 없을 수 있으며, 모든 파라미터가 매번 갱신된다는 뜻은 아닙니다.
- TensorFlow의 GPU 사용을 끄고 전처리를 CPU에 두어 Torch와 GPU 메모리 점유를 분리합니다.

`metrics-<run_id>.jsonl`에는 step, microsteps, loss, LR, clipping 전 grad norm,
peak Torch allocated bytes를 매 optimizer step 기록합니다. 이 메모리는 GPU 전체
점유량이나 정식 latency 측정값이 아닙니다. `run-<run_id>.json`에는 실행 상태,
설정, source/submodule/checkpoint/dataset revision, 학습 코드 SHA-256,
소프트웨어·GPU 정보를 기록하고 예외 시 failed/interrupted 상태를 남깁니다.
초기 경로 검사나 CUDA 가용성 검사에서 실패한 경우에는 run manifest 생성 전입니다.

저장은 숨김 `.incomplete` 디렉터리에 수행한 뒤 최종 `step-XXXXXXX`로 rename합니다.
최종 경로와 COMPLETE가 함께 있는 checkpoint만 선택하십시오. 숨김 임시 폴더는
실패 조사용이며 사용하지 않습니다. 기존 step은 덮어쓰지 않습니다.

### CPU 검증

사전학습 모델이나 RLDS 없이 작은 합성 액션 정책으로 backward/update/save/load/resume을
검사합니다. 아래 경로는 아직 없는 새 디렉터리여야 합니다.

```bash
python -m pytest tests/unit -q
python scripts/train/smoke_loop.py --output-dir artifacts/training/cpu-loop-smoke
```

이 검증은 실제 AVA/OFT forward, bitsandbytes, RLDS 또는 GPU 실행의 증거가 아닙니다.

### 실제 VLA 최소 학습과 재개

데이터와 모델별 의존성을 준비한 뒤 별도 출력 경로로 실행합니다.

```bash
python scripts/train/run_fft.py --model ava_vla --preflight
python scripts/train/run_fft.py --model ava_vla --max-steps 1 --save-steps 1 --save-training-state --output-dir artifacts/training/ava-one-step-smoke
python scripts/train/run_fft.py --model ava_vla --save-training-state
python scripts/train/run_fft.py --model ava_vla --resume artifacts/training/libero_plus_fft/ava_vla/step-0000500 --save-training-state --preflight
python scripts/train/run_fft.py --model ava_vla --resume artifacts/training/libero_plus_fft/ava_vla/step-0000500 --save-training-state
```

`max_steps`는 추가 실행 횟수가 아니라 최종 목표 optimizer step입니다. 재개 시 원본
설정의 max_steps, batch/accumulation, optimizer, schedule, seed, dataset/checkpoint
revision, upstream revision과 학습 코드 SHA-256가 일치해야 합니다. max_steps=1 smoke를
1000-step 학습으로 재개하면 거부됩니다. 장기 학습은 새 출력 경로에서 시작합니다.
source/데이터 경로가 다른 머신으로 옮겨졌다면 경로 호환성 검사에서 거부될 수 있습니다.

**RLDS stream은 새로 시작합니다.** TensorFlow shuffle, augmentation RNG와 iterator
커서는 저장하지 않으므로 optimizer/scheduler/PyTorch RNG를 복원해도 중단 전과
비트 단위로 동일한 샘플 순서나 결과를 보장하지 않습니다. stateful sampler/cache가
구현되기 전까지 manifest에도 이 제한을 남깁니다. 부분 accumulation은 저장하지 않습니다.

### 선택적 validation

`fft_defaults.eval_steps`를 양수로 설정하면 upstream `train=False`의 **val split**을
augmentation 없이 최대 `max_eval_batches`만 읽고 `val_l1`을 기록합니다. 기본값은 0입니다.
실제 TFDS에 val split이 있어야 합니다. 없으면 명시적으로 실패하며 train split을
검증용으로 재사용하지 않습니다. 데이터 분할은 episode 단위로 독립적인지 먼저 확인하십시오.
평균은 동일 batch 크기의 L1 평균이며 소수의 고정·대표 subset을 평가한 값으로 해석합니다.
검증 중 eval/no_grad를 사용하고 종료·예외 시 각 모듈의 모드와 host/Torch RNG를 복원합니다.
TensorFlow RNG와 validation 샘플 순서는 복원하지 않습니다. `val_l1`은 로봇 rollout
성공률이나 LIBERO+ macro average를 대신하지 않습니다.

## 남은 검증

- Colab에서 RLDS 분할 ZIP 다운로드·SHA-256 검증 및 압축 해제
- 실제 TFDS schema가 upstream LIBERO 변환과 맞는지 단일 batch 검사
- AVA·OFT 각각 Colab A100 단일 step forward/backward/save/load smoke test
- VRAM 사용량, 학습 시간, 체크포인트 크기와 평가 연결 검증
- AVA의 이전 시점 행동 특징을 사용하는 temporal-context 학습 확장
