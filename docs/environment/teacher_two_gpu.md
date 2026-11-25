# Teacher만 두 GPU로 실행

[Teacher 실행 노트북](../../experiments/teacher_two_gpu.ipynb)은 Kaggle과 Colab을 선택할 수 있습니다.
**Student는 기존 단일 GPU 경로를 유지합니다.** 이 설정은 AVA/OFT Teacher FFT와 고정 AVA T1
cache 생성·추론을 위한 별도 실행 조건입니다. 실제 T4 두 장의 적합성·속도는 아직 미검증입니다.

## 실행 구조

한 프로세스에서 Teacher의 층을 나눕니다. GPU 0에는 vision·projector·proprio·AVA와 Llama 앞부분,
GPU 1에는 Llama 뒷부분·최종 norm·LM/action head를 둡니다. 경계의 hidden state, mask,
position/rotary 입력과 AVA 가중치는 autograd를 유지한 채 해당 GPU로 옮깁니다.
`split_layer=null`은 실제 모듈의 parameter bytes로 균형 경계를 선택합니다.
수동 경계는 `teacher_parallel.split_layer`로 지정합니다. 디바이스는 명시적인 서로 다른
두 CUDA index여야 하며, 첫 디바이스는 기존 FFT `device`와 일치해야 합니다.

FFT에서는 각 parameter의 backward가 끝나면 gradient를 CPU에 보관합니다.
microbatch 1과 누적 8을 완료한 뒤 전체 norm을 계산해 한 번 clipping하고,
GPU에 gradient 하나씩만 복원하여 **원래 PagedAdamW8bit**로 해당 parameter를 한 번 업데이트합니다.
모든 parameter group·원래 index·moment·update counter를 보존합니다.
가중치를 동결하거나 LoRA/QLoRA로 바꾸지 않습니다. GPU 간 통신과 CPU 왕복 비용이 있어
이 구현은 속도 향상을 보장하지 않습니다. 실제 장치가 없는 CPU 검증으로는 paging 적합성을 판정할 수 없습니다.

AVA의 K4, detach 2, episode-uniform sampling, batch 1×누적 8, L1 loss·mask·seed·LR·schedule·검증을
유지합니다. OFT의 원래 단일 query 경로도 유지합니다. 옵티마이저 경계는 두 GPU의 평균이 아니라
**기존과 같은 8 sequences(AVA 32 query 예측) 또는 OFT 8 queries**입니다.
두 GPU에 같은 모델을 복제하는 DDP가 아니며, GPU 두 장의 VRAM을 하나의 연속 메모리로 다루지 않습니다.

핵심 구현은 `src/vlade/training/teacher_parallel.py`, `gradient_offload.py`이며
기존 FFT·공통 loop·Hub 저장 모듈에 선택적으로 연결됩니다. Student에는 이 기능을 자동 적용하지 않습니다.

## T4의 정밀도와 메모리

T4의 Turing Tensor Core 경로에는 FP16을 사용합니다. 기존 L4용 BF16 설정은 보존합니다.
새 기본 profile은 **FP16 parameter·autocast·CPU gradient**이며 optimizer의 원래 8bit moment
알고리즘을 유지합니다. BF16과 exponent 범위·반올림이 다르므로 같은 수치 결과로 보고하지 않습니다.
FP32 master weight/전체 FP32 gradient를 새로 상주시킨 구성도 아닙니다.

초기 loss scale은 128입니다. scaled gradient overflow가 발생하면 **optimizer를 변경하기 전에**
같은 누적 window·같은 RNG로 scale을 줄여 재시도합니다. query cursor·optimizer step·LR은
성공한 window에만 진행합니다. 기본 재시도 한도는 8이며 모두 실패하면 오류로 종료합니다.
forward action/loss가 FP16 범위를 넘어 비정상이면 loss scaling으로 숨기지 않고 중단합니다.
scale·성공 counter는 checkpoint에 저장하고 재개합니다. 2,000회 성공마다 scale을 증가시킵니다.

Teacher backbone만 FP16 약 14.05 GiB이고 head·AVA 등이 추가됩니다.
단순 분할 상태에서 gradient와 optimizer를 모두 GPU에 상주시켜 16GB 두 장에 맞는다고
주장할 수 없습니다. 이 경로는 CPU gradient와 기존 optimizer의 paging으로 GPU 부담을 줄입니다.
CPU gradient도 대략 Teacher parameter bytes만큼 RAM이 필요하며, paged moment·데이터·현재 누적
입력·직렬화에는 추가 RAM/디스크가 필요합니다. 가중치 균형은 activation·일시 workspace 균형을 보장하지 않습니다.
각 GPU에 weight 외 4 GiB reserve를 사전 확인하지만 이는 전체 실행 적합성의 보장은 아닙니다.

원래 bitsandbytes **0.43.3** 환경을 사용합니다. 이 버전의 일반 optimizer loader는 저장된
moment를 일반 CUDA tensor로 옮기므로, 새 경로에서는 moment를 한 tensor씩 원래 paged buffer에
복원하고 quantization map의 dtype·값·step을 보존합니다. 기존 단일 GPU loader는 변경하지 않습니다.
새 경로의 trainer state는 memory mapping으로 읽고 복원 후 참조를 해제하여 CPU에 전체 복사본을
동시에 쌓지 않습니다. 모델·moment를 포함한 full checkpoint는 대략 수십 GB이며 입력 캐시와
이전 checkpoint를 함께 보관할 디스크가 필요합니다. 플랫폼의 저장 한도도 파일럿 전에 확인합니다.
다른 버전으로 자동 대체하지 않으며, 해당 paging/CUDA kernel은 실기기 검증이 필요합니다.

참고: [NVIDIA Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html),
[post-accumulate gradient hook](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html),
[고정 bitsandbytes 0.43.3 optimizer 구현](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.43.3/bitsandbytes/optim/optimizer.py).

## 설정과 실행

| 작업 | 설정 | 동작 |
|---|---|---|
| AVA Teacher full FFT | [ava_fft_tp2.json](../../configs/research/ava_fft_tp2.json) | 기존 K4·detach 2·전체 파라미터·paged AdamW8bit |
| OFT full FFT | [oft_fft_tp2.json](../../configs/research/oft_fft_tp2.json) | 원래 OFT 학습·전체 파라미터·paged AdamW8bit |
| 고정 AVA T1 cache | [cache_teacher_tp2.json](../../configs/research/cache_teacher_tp2.json) | 두 GPU 추론, 별도 FP16 cache 식별자 |

노트북에서 `PLATFORM='kaggle'` 또는 `'colab'`, `PROFILE`, `ACTION`, `RUN_GROUP`을 지정합니다.
소스는 Git commit을 확정하여 새 경로에 준비하며 실행 중인 소스를 바꾸지 않습니다.
해당 플랫폼 Secrets에 `HF_TOKEN`을 설정하고 비공개 입력 읽기·조직 저장소 쓰기 권한을 부여합니다.
토큰은 환경 변수로만 전달하며 출력·설정 파일에 저장하지 않습니다.
워크스페이스는 Kaggle의 `/kaggle/working/vlade_teacher/work` 또는 Colab의
`/content/vlade_teacher/work`입니다. 플랫폼의 GPU 할당량이나 추가 혜택을 이 코드가 늘리지 않습니다.
첨부 이미지의 Colab Pro+ 연동 주 60시간 혜택은 이번 공식 자료 조사에서 확인하지 못했습니다.
사용 가능한 GPU와 남은 시간은 실제 계정에서 확인해야 합니다.

기본 `ACTION='plan'`은 설정과 출력 경로만 표시합니다. `setup`은 기존 모델별 pinned runtime을
설치하고 `run`은 선택한 작업을 실행합니다. 새 유료 GPU 할당·자동 장시간 학습은 이 코드 작성으로 시작하지 않습니다.
새 runtime은 이 구현을 포함한 Git commit이 별도로 게시되어야 사용할 수 있습니다.

CLI 계획 예시는 다음과 같습니다.

```bash
python -B scripts/research/run.py --config configs/research/ava_fft_tp2.json \
  --workspace /kaggle/working/vlade_teacher/work --action plan
```

실제 FFT 전에는 별도 run group으로 optimizer 1 step의 파일럿을 수행합니다.
노트북 `OVERRIDES` 예시는 다음과 같습니다. 이 값은 파일럿용 별도 실험이며 본 실험 목표를 변경하는 것이 아닙니다.

```python
OVERRIDES = {
    'run_group': 'ava-tp2-fp16-pilot-seed42-r1',
    'train.max_steps': 1,
    'train.warmup_steps': 0,
    'train.save_steps': 1,
    'train.eval_steps': 1,
}
```

본 실행은 새 group과 원래 설정으로 시작합니다. 파일럿의 `final`은 그 1-step 파일럿 목표의 완료이며
1,000-step 본 학습의 final이 아닙니다. 본 실행을 재개할 때는 동일 group·source commit·입력·목표·
split/dtype/offload 설정을 유지하고 검증한 Hub checkpoint를 기존 `resume`에 지정합니다.
새 profile로 기존 BF16 단일 GPU 학습을 이어 붙이지 않습니다.

## cache와 온라인 Teacher

검증된 전체 cache가 있으면 기존처럼 재사용합니다. 새 cache 생성은 고정 HF T1·선정 보고서·
prepared split/정규화·전체 train coverage·replay 검증을 그대로 요구합니다.
FP16 실행을 BF16 cache와 혼합하지 않도록 asset 키를 **`teacher_cache_t1_fp16_tp2`**로 구분합니다.
`cache_config.json`과 완료 manifest에 실제 split·dtype·장치 정보를 기록하며,
기존 `teacher_cache_t1`이나 Drive 원본을 덮어쓰지 않습니다. 전체 게시 대상은 기존 비공개 dataset
`VLADE-konkuk/vlade-teacher-cache`입니다. pilot cache는 전체 cache로 게시하지 않습니다.

온라인 T1 worker도 `distillation.teacher_parallel`에 동일 구조를 전달할 수 있습니다.
L4의 `teacher_residency='exclusive'`를 함께 사용할 경우 Student를 CPU로 옮긴 뒤
Teacher를 두 GPU에 배치하고, 두 GPU의 Teacher와 episode context를 CPU로 돌려놓은 뒤
Student를 복원합니다. Student는 계속 단일 GPU입니다.
현재 Student의 BF16 요구를 T4용 FP16 Student 지원으로 변경한 것은 아니므로,
T4에서 Student도 함께 실행되는 것으로 해석하지 마십시오. cache를 통한 별도 플랫폼 실행과 구분합니다.
S2의 Plan 감독 이력·원문 probe·EOS 검증·같은 관측 계약도 유지하며 잘못된 Plan을 대체하지 않습니다.

## 기록과 검증 한계

FFT `run-*.json`에는 실제 split·dtype·장치별 weight bytes, `metrics-*.jsonl`에는 양쪽 GPU의
allocated/reserved/peak/free·CPU RSS·현재 loss scale을 기록합니다. cache 완료 manifest에도
양쪽 GPU 메모리를 기록합니다. warm-up·검증·저장·전송 비용을 포함해 실제 optimizer step 시간을
확인해야 하며 CPU 검사 시간을 T4 학습 속도로 보고하지 않습니다.
레이어 분할·정밀도·offload 설정과 scaler 상태가 seal에 포함되고 변경된 설정 재개는 거절합니다.
기존 모델 key·full checkpoint·optimizer/scheduler/RNG·cursor·Hub last/best/final/pending 계약은 보존합니다.

CPU에서 작은 native Llama의 layer hook·gradient·checkpoint 재계산, 누적 8·K4/TBPTT,
원래 AdamW와의 업데이트 일치, overflow 시 동일 window/RNG 재시도, scaler 저장/재개,
paged moment 재구성 경계와 설정 변경 거절을 검사합니다.
실제 두 CUDA GPU·bitsandbytes CUDA paging·7B full FFT·실제 AVA Transformers fork의
GPU attention·FP16 수렴·실제 T1 cache·Hub 전송은 미검증입니다.
노트북 재생성은 `python -B scripts/prepare/generate_teacher_parallel_notebook.py`입니다.
