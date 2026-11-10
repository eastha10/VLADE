# AVA FFT를 A100 8시간에 맞추는 실행 정책

날짜: 2026-10-31 (한국 시간). 사용자가 A100 최대 8시간을 지정했습니다.
정책: `configs/experiments/ava_fft_a100_8h.json`, 실행: `experiments/colab_02_ava_fft_and_fixed_cache.ipynb`.

2026-11-11 보완: 새 연구 CLI는 설정의 max_steps와 8시간 상한을 사용하며 AVA FFT는
[K4·detach2 학습](../environment/ava_sequence_training.md)으로 변경했습니다. 아래
단일 query·1 epoch 예산 산식은 당시 정책입니다. 새 batch 단위는 시퀀스이고 K4 GPU
파일럿으로 속도·메모리를 다시 확인해야 합니다. attention regularizer는 논문 LoRA 경로에만 있습니다.

[AVA-VLA v4, Appendix A 및 §3.4](https://arxiv.org/html/2511.18960v4)는 LoRA rank 32와
AVA 모듈 전체 학습, 관측 시퀀스 K=4와 두 번째/세 번째 시점 사이 gradient detach를 사용합니다.
LIBERO 설정은 batch 64, 40,000 gradient steps, LR 5e-4, 10% warm-up, cosine 및 clipping 1.0이며
action chunk는 8입니다. LIBERO+는 강건성 평가로 제시되어 있으므로 이것을 LIBERO+ FFT의
검증된 학습 예산으로 옮기지 않습니다. 원 논문은 A800 실험이며 단일 Colab A100 소요시간의 근거가 아닙니다.

우리의 기존 FFT 연구 조건을 유지합니다. 기존 AVA checkpoint에서 전체 파라미터를 학습하고
LR은 보수적 시작값 2e-5로 유지합니다. 논문의 LoRA LR 5e-4를 FFT에 그대로 적용하지 않습니다.
chunk 8·MAE·clipping을 유지하고 warm-up 10%/cosine만 학습 예산에 맞춰 적용합니다.
effective batch 8(1×8)과 이미지 증강 비활성은 현재 메모리·공통 prepared 입력 계약에 따른 조정이며
논문과 일치하거나 최적이라고 검증한 값이 아닙니다.

당시 FFT는 temporal zero-context였고 K=4 TBPTT와 논문의 attention regularizer를 재현하지 않았습니다.
현재 K4 추가만으로 논문 학습 조건이나 성능 재현을 완료했다고 해석하지 않습니다.

4 optimizer step 파일럿은 본 학습과 같은 accumulation을 사용합니다. 첫 step은 속도 추정에서
제외하고 나머지 3개 중 가장 느린 시간, 실제 validation/저장/모델 준비 시간을 사용합니다.
설치·파일럿·재로딩은 이미 경과한 시간으로 차감하고, 후속 작업용 1시간과 종료용 최소 5분,
추정 비용의 30% 여유를 확보합니다. 최대 1 epoch와 남은 시간에 가능한 step의 최솟값으로
목표를 고정합니다. 남은 예산이 2 optimizer step에도 부족하면 본 학습을 시작하지 않습니다.

본 학습은 중간·마지막 약 2회 저장/validation하고 optimizer·scheduler·RNG도 함께 저장합니다.
기존 checkpoint·cache를 자동 삭제하지 않습니다. budget stop은 완료와 구분하며, 중단 실행의
비교에는 실제 학습 step/query 수를 사용합니다. Student Task-only/KD와 다른 모델에 이 Teacher의
step 수를 자동 전파하지 않으며 각 비교군은 같은 데이터·유효 학습량·seed 조건을 별도로 고정합니다.
최종 test는 학습량·checkpoint 선정에 사용하지 않습니다. val L1만으로 T1을 자동 선정하지 않습니다.

[Colab runtime API](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py)의
`unassign()`으로 마감 시 런타임 반환을 요청합니다. GPU 명령 종료와 런타임 반환은 구분하며
서비스 오류까지 포함한 과금 중단을 보장하지 않습니다. 가격은 가정하지 않고 실제 CU/시간을 기록합니다.
GPU 학습, 실제 메모리 적합성, 최종 step 수·수렴·성공률은 아직 측정하지 않았습니다.
