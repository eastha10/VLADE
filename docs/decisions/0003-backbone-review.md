# 0003: Student backbone 재검토

검토일: 2026-10-24 (한국 시간). 상태: 후보 우선순위 제안, 교체 미실행.

## 현재 설계에서 유지할 것

로컬 HEAD는 `1aa11fb`이며 AVA-VLA와 OpenVLA-OFT의 고정 소스를 사용합니다.
`configs/experiments/libero_plus_fft.json`의 주 실험은 LIBERO+ FFT,
학습 환경은 Colab A100, 배포 목표는 Jetson Orin Nano급입니다.
실제 A100 VRAM과 목표 control Hz·지연시간 허용치는 아직 확정되어 있지 않습니다.
원본 LIBERO baseline 재현과 자체 LIBERO+ 결과는 계속 분리합니다.

로컬 AVA checkpoint의 `config.json`에서 Llama-2-7B, 두 이미지 224×224,
기존 tokenizer·norm_stats를 확인했습니다. 액션은 chunk 8×7, proprio는 8차원입니다.
VLADE Student, Teacher cache 생성기, KD loss는 아직 비어 있습니다.
기존 Teacher와 baseline을 최신 LLM으로 교체하면 기존 VLA 가중치와 비교점이
달라지므로 유지합니다. 학습 루프 보완은 기존 L1 FFT 경로에 적용했습니다.

## 선택 기준과 권고

일반 언어·수학 점수보다 visual grounding, 액션 정확도, 목표 기기 p95 지연시간,
전체 정책 메모리, temporal context와 정규화 규약, 제한된 데이터로 정렬 가능한지가
우선입니다. 아래 우선순위는 이 조건에 대한 설계 판단이며 자체 VLA 측정 결과가 아닙니다.

1. **LLM만 바꾸는 통제 실험:** Qwen3-0.6B를 첫 후보, 1.7B를 용량 비교점으로 둡니다.
   일반 Transformer 계열의 hidden-state KD·연속 action head 연결부터 검증하기
   용이하다는 판단입니다. 텍스트 사전학습만으로 시각 정렬이 확보되지는 않습니다.
   기존 DINO/SigLIP와 새 projector를 정렬해야 하며, 0.6B가 충분한 조작 성능을
   낸다고 주장하지 않습니다. Llama-3.2-1B도 구조적 호환성 비교 후보입니다.
2. **전체 정책의 엣지 효율 비교:** SmolVLM2-500M 기반 Student 또는 SmolVLA를
   독립 실험으로 우선 검토합니다. 시각·언어 사전학습과 작은 전체 모델의 장점이
   있으나 vision encoder·action expert도 바뀌므로 LLM 단독 교체의 효과로 해석하지 않습니다.
3. **최신 멀티모달 확장:** Qwen3.5-0.8B를 후속 후보로, 2B를 상위 용량으로 둡니다.
   native vision과 hybrid attention의 장점은 매력적이지만, 현 AVA의 Llama 전용
   attention 수정 및 Jetson export·실행 커널과의 호환성 검증이 먼저 필요합니다.
4. Gemma 4 E2B는 이름의 effective 2B와 실제 전체 파라미터가 달라 초기 후보에서
   후순위로 둡니다. 더 최신이라는 이유만으로 전체 메모리 제약을 충족한다고 보지 않습니다.

## 후보 비교: 공식 자료에서 확인한 사실과 적용 판단

| 후보 | 확인한 사실 | VLADE 연결·계산 측면의 판단 | 우선순위 |
|---|---|---|---|
| 현 Llama-2-7B | 기존 공개 VLA checkpoint의 언어부 | 유일하게 현재 baseline 가중치를 그대로 유지. 엣지 Student로는 큼 | Teacher/baseline 유지 |
| Qwen3-0.6B / 1.7B | 텍스트 causal LM, GQA, Apache 2.0. Qwen3는 Transformers 4.51 이상 필요 | vision 정렬·tokenizer·projector·action head 재구성 필요. 기존 fork 안에서 모델 ID만 바꾸면 안 됨 | 통제 실험 1순위 |
| Llama-3.2-1B | 텍스트 모델, Llama 3.2 라이선스 | Llama 계열 구조를 재사용하기 쉽다는 가설. vocab/hidden size/position·attention 구현은 달라 가중치 호환 아님 | 대조 후보 |
| SmolVLM2-500M | 이미지·비디오·텍스트 VLM, Apache 2.0 | 시각 정렬을 확보한 출발점. 기존 vision·projector와 함께 단순 교체 불가 | 전체 VLM 비교 1순위 |
| SmolVLA | 450M 사전학습 정책. multi-view·state·언어 → continuous action chunk | 로보틱스 직접 근거가 있음. LeRobot 데이터/flow action expert/정규화·주기 변환 필요 | 별도 edge baseline |
| Qwen3.5-0.8B / 2B | vision encoder를 포함한 VLM. 실제 config에 linear_attention와 full_attention 혼합, Apache 2.0 | 기존 AVA의 full attention 기반 훅 이식·마스크·recurrent state·export 부담 큼 | 2단계 실험 |
| Gemma 4 E2B | effective 2.3B, embeddings 포함 5.1B. image/audio 지원, Apache 2.0 | PLE와 modality 구성까지 포함한 실메모리 확인 필요. 2B dense 모델과 동급 메모리로 계산하면 안 됨 | 보류 |

Qwen3 사실은 [0.6B 공식 모델 카드](https://huggingface.co/Qwen/Qwen3-0.6B),
[1.7B 공식 카드](https://huggingface.co/Qwen/Qwen3-1.7B),
[0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)를 근거로 합니다.
Llama 후보의 modality·라이선스는 [Meta 공식 카드](https://huggingface.co/meta-llama/Llama-3.2-1B)를 확인했습니다.
모델명 규모와 Hub 자동 파라미터 표기가 일치하지 않는 경우가 있어 최종 메모리 예산은
고정 revision의 safetensors `numel` 합계로 다시 계산해야 합니다.

소형 VLM과 로봇 정책 근거는 [SmolVLM2 공식 카드](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
[SmolVLA 논문](https://arxiv.org/abs/2506.01844),
[LeRobot 공식 사용법](https://huggingface.co/docs/lerobot/smolvla)입니다.
SmolVLA의 공개 실험은 VLA 가능성의 근거이며 VLADE의 동일 LIBERO+ FFT 성공률 증거는 아닙니다.

새 후보는 [Qwen3.5-0.8B 공식 카드](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[2B 공식 카드](https://huggingface.co/Qwen/Qwen3.5-2B),
[실제 architecture config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json),
[Gemma 4 E2B 공식 카드](https://huggingface.co/google/gemma-4-E2B-it)에서 확인했습니다.
Qwen3.5 계열 소개의 MoE 설명을 이 소형 config의 MoE 사용 근거로 삼지 않았습니다.
실행 시점에는 main 링크 대신 모델 revision과 라이브러리 버전을 고정해야 합니다.
현재 모델 다운로드나 dependency 교체는 수행하지 않았습니다.

## 계산 자원 판단

가중치만의 이론적 BF16 메모리는 `2 × 파라미터 수 / 2^30 GiB`입니다.
예를 들어 명칭 기준 0.6B는 약 1.12 GiB, 1.7B는 3.17 GiB, 7B는 13.04 GiB입니다.
이는 직접 계산한 하한 추정이며 실제 GPU 메모리 측정이나 모델 전체 크기가 아닙니다.
텍스트 LLM이면 vision encoder·projector·action head를 별도로 더해야 합니다.
INT8/INT4도 scale·padding·커널 workspace·activation·KV/temporal state를 더하므로
이론상 절반/4분의 1만으로 배포 가능 여부를 결정하지 않습니다.

전통적인 mixed-precision AdamW를 가정하면 weights/gradients/FP32 moments/master
weights에 대략 16 bytes/parameter가 필요할 수 있습니다. 7B만으로 약 104 GiB이며
vision 및 activation은 제외한 값입니다. 현재 BF16+8-bit paged optimizer는 이보다
다른 저장 정책을 쓰므로 이 수치를 실제 요구량으로 단정하지 않습니다. 다만
batch=1·accumulation을 줄여도 optimizer 상태 메모리는 줄지 않는다는 점은 같습니다.
A100 40/80GB 중 어떤 GPU인지, host RAM과 paging 비용까지 첫 step에서 측정해야 합니다.

Orin Nano Super의 공식 사양은 8GB 공유 메모리입니다. 실제 대상 보드가 같은지 확인하고,
OS·카메라·로봇 제어·전처리·runtime을 포함한 전체 메모리에서 여유를 남겨야 합니다.
공식 peak TOPS를 VLA FPS로 환산하지 않습니다.
[NVIDIA 기기 사양](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/).

두 카메라의 visual token을 줄이는 설계가 decoder 크기 축소만큼 중요할 수 있습니다.
학습·추론 모두 동일한 이미지 해상도·patch/token 수에서 비교한 뒤 token compression을
별도 ablation으로 추가합니다. 8개 액션 chunk에서 일부만 실행하거나 비동기로 겹치면
환경 반응성도 달라지므로 단순 `8 / inference_latency`를 실제 제어 Hz로 보고하지 않습니다.

## 호환성 경계

- 현재 AVA/OFT는 각각 bidirectional/extra-attention을 위한 **서로 다른 Transformers fork**를
  요구합니다. 일반 최신 Transformers로 올려 baseline을 깨뜨리지 않습니다.
  Student는 별도 환경에서 구현하고 공통 output 계약으로 학습 루프와 연결합니다.
- 새 tokenizer에 기존 Llama action token ID·IGNORE_INDEX 규칙을 그대로 적용하지 않습니다.
  continuous action query를 명시적으로 정의하고 padding·causal/bidirectional mask와
  hidden-state 추출 위치를 테스트합니다. Text generation·thinking chain은 제어 경로에서
  생성하지 않고 고정 길이 action head 출력을 사용합니다.
- head/proprio/vision projector의 입력·출력 hidden size를 새 backbone에 맞춰 재초기화합니다.
  기존 head의 state_dict를 strict=False로 억지로 이식하지 않습니다.
- action KD는 동일 좌표계·정규화·gripper 규약의 `[B,8,7]`에서 비교합니다.
  representation KD는 projection 후 같은 의미의 action query/pooled feature를 맞춥니다.
  서로 다른 tokenizer의 text token이나 다른 vision grid를 index로 직접 정렬하지 않습니다.
- AVA Teacher cache는 episode 순서와 model query 주기를 지키며 episode 시작에서 temporal
  context를 reset해야 합니다. 임의 shuffle sample로 만든 zero-context cache를 원본
  temporal Teacher 출력과 동일하다고 표기하지 않습니다.

## 다음 실험의 채택 조건

1. 모델별 별도 환경에서 1 batch forward/backward/save/load와 유한 gradient를 확인합니다.
2. 소규모 고정 episode로 overfit해 vision·언어·proprio 입력이 예측에 영향을 주는지 확인합니다.
3. 같은 split·최적화 budget으로 task-only, action KD, representation KD를 순차 비교합니다.
   Teacher cache·정규화 통계·분할 해시와 실제 환경 버전을 manifest에 남깁니다.
4. LIBERO+ perturbation 차원별 성공률·macro average 및 별도 원본 LIBERO 결과를 기록합니다.
5. 목표 보드에서 batch=1, 동일 카메라 입력, 동일 실행 chunk 길이, dtype/quantization,
   warmup·반복 횟수·CUDA 동기화 방법을 고정합니다. preprocessing부터 action head까지
   p50/p95 지연시간, peak 전체 메모리, 전력 모드·온도와 실제 control cycle을 측정합니다.
6. 성공률과 p95·메모리 제약의 Pareto 비교 후 backbone을 확정합니다. 허용 성공률 감소,
   목표 Hz와 최대 p95는 아직 미정이므로 임의의 합격 숫자를 넣지 않았습니다.

이번에 자체 확인한 범위는 CPU 합성 학습과 코드·설정입니다. 실제 Student 성공률,
GPU FFT 메모리, TensorRT export, Jetson 지연시간은 측정하지 않았습니다.
