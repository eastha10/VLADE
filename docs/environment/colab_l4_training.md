# Colab NVIDIA L4 24GB 학습

Teacher를 두 GPU에 나누는 별도 FFT·cache 경로는 [Teacher 분할 안내](teacher_two_gpu.md)를
따릅니다. 아래 내용은 기존 단일 L4 경로이며 Student 분할을 자동 적용하지 않습니다.

전용 실행본은 [colab_l4_training.ipynb](../../experiments/colab_l4_training.ipynb)입니다.
현재 구현은 CPU 축소 모델 검증 대상이며 **실제 L4의 적합성·속도·수렴은 미검증**입니다.
실행 코드를 포함한 Git commit을 확정한 뒤 새 run group에서 파일럿부터 실행합니다.
기존 실행의 소스·설정 identity 검사를 완화하거나 A100 checkpoint를 새 설정으로 재개하지 않습니다.

## 경로와 메모리 판단

| 경로 | 구현과 조건 | L4 선택 |
|---|---|---|
| S0 | native QwenVLA 전체 Task-only, GT masked L1, 8,000 optimizer step | `s0` |
| S1 | 같은 schema 2 S0 전체 학습, GT L1 + λ Teacher L1, 기본 100 step | 전체 검증 cache가 있으면 `s1_cache`; 없으면 `s1_online` |
| S2 | 같은 S0·T1·action loss + T1 접근 플랜을 Student 입력으로 사용 | `s2`; 감독 이력이 있는 T1과 probe 필수 |
| 기존 `qwen_task` / `qwen_kd` | `student_task.py`의 별도 query-head Student | 기존 통합 노트북 유지; 새 native S0/S1/S2와 혼용하지 않음 |
| Teacher cache 생성 | 고정 AVA T1만 로드하는 `fixed_cache.py` | 기존 `cache_teacher` 경로, Student 동시 로드 없음 |
| AVA FFT / Suite-balanced / OFT FFT | 약 7.5B backbone 및 원래 head·optimizer·시퀀스 | 현재 full FFT는 L4 24GB에 부적합 |
| AVA 논문 경로 | 원래 별도 rank-32 LoRA recipe | FFT의 자동 대체로 사용하지 않음 |

현재 native 구조는 backbone **852,985,920개**, 정책 구성요소 포함 **946,437,191개**
파라미터입니다. 로컬 S0의 shape 설정으로 메모리 없는 meta 모델을 구성해 센 수치이며,
실제 checkpoint 가중치·데이터를 로드하거나 GPU 학습한 결과가 아닙니다.
FP32 파라미터·gradient·AdamW 두 moment의 정적 합계는 **15,142,995,056 bytes ≈ 14.10 GiB**입니다.
activation·autocast 변환·attention mask·CUDA workspace·allocator·validation·저장 비용은 별도입니다.
파일럿에서 실제 모델의 파라미터 수와 free memory로 다시 검사합니다.
기본 2 GiB reserve는 사전 거절용 여유이며 최대 activation을 보장하는 값은 아닙니다.

고정 HF lock의 AVA/OFT backbone shard 합계는 각각 약 14.05 GiB입니다.
BF16 weight와 동일 크기의 gradient만 대략 **28.09 GiB**이므로 activation과 optimizer를
더하기 전부터 L4 용량을 넘습니다. 기존 paged AdamW 8bit의 optimizer paging으로
weight·gradient까지 자동 offload되는 것은 아닙니다. microbatch는 이미 1이며 K4/TBPTT를
줄이거나 누적을 늘려도 이 정적 하한은 사라지지 않습니다.
이 작업에서는 FFT를 LoRA/QLoRA로 바꾸거나 CPU optimizer·분산 학습기로 교체하지 않았습니다.
동일 FFT를 유지하려면 더 큰 기존 GPU 할당을 기다리거나 별도로 검증한 parameter/gradient
offload·분산 경로가 필요합니다. 이는 새 구현·파일럿·재개 계약 검증이 필요한 후속 작업입니다.

## L4 설정

| PROFILE | 설정 |
|---|---|
| `s0` | [qwen_vla_s0_l4.json](../../configs/experiments/qwen_vla_s0_l4.json) |
| `s1_cache` | [qwen_vla_action_kd_l4_cache.json](../../configs/experiments/qwen_vla_action_kd_l4_cache.json) |
| `s1_online` | [qwen_vla_action_kd_l4.json](../../configs/experiments/qwen_vla_action_kd_l4.json) |
| `s2` | [qwen_vla_plan_kd_l4.json](../../configs/experiments/qwen_vla_plan_kd_l4.json) |

원래 seed·학습률·weight decay·loss 가중치·학습 범위·목표 step·검증 query 수를 유지합니다.
microbatch **1 × accumulation 8 = effective batch 8 queries**입니다.
AVA FFT의 **8 sequences / 32 query 예측**과 다른 단위입니다.
이미 최소 batch이므로 누적을 줄여 메모리를 아끼거나, 학습량·schedule을 바꾸지 않습니다.

- FP32 파라미터·gradient·AdamW moment와 BF16 autocast를 유지합니다. BF16 weight-only
  학습으로 optimizer 정밀도를 바꾸지 않습니다. `foreach=False`도 기존 native trainer와 같습니다.
- native HF의 non-reentrant gradient checkpointing을 사용합니다. KV cache는 훈련에서 끕니다.
  autocast 범위는 각 forward로 제한하여 no-grad history replay의 BF16 cast가 다음
  gradient forward나 Teacher 교대 적재에 남지 않도록 합니다. episode 상태·detach 경계는 유지합니다.
- Qwen native vision/full attention에는 공개 `set_attn_implementation('sdpa')`를 적용합니다.
  AVA의 `log(omega)` additive mask와 대각선 가중치 1을 유지하고 DeltaNet 구조도 유지합니다.
  SDPA가 실제로 선택할 CUDA kernel과 절감량은 하드웨어에서 확인해야 합니다.
  FlashAttention을 강제하거나 추가 CUDA 패키지를 설치하지 않습니다. `eager`도 명시적으로 선택할 수 있습니다.
- `memory_profile`과 `teacher_residency`는 기존 settings identity 안에 저장됩니다.
  backend·누적·입력·scope 등을 바꾼 실행은 새 run group을 사용합니다.

공식 근거: [PyTorch AdamW의 foreach 메모리 설명](https://docs.pytorch.org/docs/main/generated/torch.optim.adamw.AdamW_class.html),
[HF attention backend 전환](https://huggingface.co/docs/transformers/attention_interface).
CUDA의 특정 fused kernel 지원을 이 문서만으로 보장하지 않습니다.

## Teacher cache 재사용과 온라인 적재

`s1_cache`는 `hfasset://teacher_cache_t1`을 공통 HF resolver로 검증한 뒤
기존 `CachedActions`로 전체 train coverage, T1 provenance, 데이터 revision·split·정규화,
관측·정답·mask·query 순서와 파일 무결성을 확인합니다. Teacher 모델과 AVA 환경은 로드하지 않습니다.
이미 있는 검증된 HF 캐시는 그대로 재사용합니다. 키가 없거나 검증이 실패하면 중단하며,
pilot cache·미등록 로컬 디렉터리·폴더의 COMPLETE만으로 우회하지 않습니다.

이번 로컬 조사에서 `data/teacher_cache/`에는 완료 cache가 없었고, 저장소 lock에도
`teacher_cache_t1`이 없었습니다. 설정의 고정 원격 registry 조회는 DNS 연결 실패로 완료하지
못했으므로 원격 cache의 존재 여부는 미확인입니다. 가중치·데이터는 내려받지 않았습니다.
cache가 이미 게시되어 있으면 이를 포함한 **불변** `workflow.registry_revision` 또는 검증된
런타임 registry를 지정합니다. 없는 경우 `s1_online`을 선택하거나 기존
[연구 cache 절차](research_runs.md)의 `select_teacher` → `cache_teacher`를 별도 실행합니다.
원래 검증 보고서·T1 선정·전체 train 생성·replay 검증·비공개 HF 게시 조건을 그대로 따릅니다.
새 GPU 작업과 대용량 전송은 이 문서 작성만으로 시작되지 않습니다.

온라인 S1/S2의 `teacher_residency='exclusive'`는 다음 순서입니다.

1. AVA worker가 CPU에 고정 BF16 T1을 준비합니다.
2. Teacher 요청 전에 Student weight·누적 gradient·optimizer state를 CPU로 옮깁니다.
3. worker가 T1과 episode context를 GPU로 옮겨 원래 causal replay·action·plan을 계산합니다.
4. T1과 context를 CPU로 되돌리고 CUDA 동기화·빈 allocator cache 해제를 마친 뒤 응답합니다.
5. Student와 optimizer·누적 gradient를 원래 device로 복원한 뒤 Student forward/backward를 수행합니다.

Parameter 객체와 optimizer 연결을 유지하고 loss나 episode detach 경계를 바꾸지 않습니다.
Teacher timeout·worker 오류는 worker 종료 후 Student를 복원합니다.
Teacher target은 Student의 autograd graph를 만들기 전에 요청합니다.
이 경로는 PCIe 왕복 비용이 크며 속도 향상을 보장하지 않습니다. 특히 S2의 validation replay에서도
플랜이 필요하므로 전송이 자주 일어납니다. **CPU RAM도 두 모델과 optimizer를 수용해야 합니다.**
Colab의 `/proc/meminfo` available RAM이 Student offload 추정량 + 2 GiB보다 작으면 중단합니다.
worker 시작 전에도 T1 직렬화 크기 + Student의 정적 AdamW 합계 + 2 GiB를 확인합니다.
이 구성에서는 대략 30 GiB 이상의 available RAM이 필요한 보수적 사전 조건입니다.
일반 RAM 런타임에서는 부족할 수 있으며, 가능한 기존 high-RAM 할당 또는 검증 cache를 사용해야 합니다.
Teacher의 최초 CPU 로드와 GPU inference 자체도 별도 여유가 필요합니다.

기존 action cache에는 플랜이 없습니다. S2를 `teacher_mode='cache'`로 바꾸거나
형식 검사만 통과한 문장으로 플랜을 대체하지 않습니다.
[Plan 감독 계약](teacher_plan_generation.md)을 갖춘 T1이 없으면 S2는 기존처럼 거절됩니다.

## 파일럿·측정·재개

노트북의 기본 `ACTION='plan'`은 설치·다운로드·GPU 실행·게시를 하지 않습니다.
새 L4 소스 commit을 사용하고 `PROFILE`, 고유 `RUN_GROUP`, 필요한 HF pin을 설정한 뒤
`plan → setup → preflight → pilot → train` 순서로 실행합니다.
S2는 `preflight → probe → pilot` 순서이며 probe 원문의 의미도 검토합니다.
profile을 바꿀 때는 새 RUN_GROUP을 지정합니다.

CLI 예시도 같은 모듈·Hub 저장 경로를 사용합니다.

```bash
python -B scripts/train/run_distillation.py \
  --config configs/experiments/qwen_vla_action_kd_l4.json \
  --workspace /content/vlade_l4 --run-group l4-s1-seed42-r1 --action plan
```

`pilot`은 목표 schedule·누적 8을 유지하는 optimizer **1 step**이며 validation·저장·Hub 전송까지
포함합니다. 이 결과만으로 시간 예산·수렴을 판단하지 않습니다. 충분한 실측을 얻기 전 장시간
본 학습을 시작하지 마십시오. 특히 AdamW state는 첫 update에서 생성되므로 forward-only로
적합성을 판정하지 않습니다. OOM이면 캐시/온라인/실제 profile과 측정 파일을 확인하며
자동으로 LoRA·부분 동결·짧은 시퀀스로 재시도하지 않습니다.

출력은 `<workspace>/runs/<run-group>/S0|S1|S2/`입니다.

- `invocation-*.json`: 실제 parameter 수·scope, FP32/BF16, backend·checkpointing·effective batch,
  정적 AdamW 추정량과 memory 로그 이름.
- `memory-*.jsonl`: Student 준비·offload·optimizer 경계(검증/저장 후 포함)의 allocated/reserved,
  allocator peak와 **device 전체 free/used**. 다른 프로세스 사용량은 device 수치에 포함됩니다.
- `memory-t1-*.jsonl`: 온라인 T1 query 완료와 CPU 복귀 시 같은 지표.
- `metrics-*.jsonl`: step loss·GT/Teacher L1·gradient norm·validation·저장 시간·메모리.
- 기존 `timings-*.csv`: CUDA 동기화 전후 측정, 첫 step/forward warm-up 표시,
  Student forward/backward/optimizer·검증·저장·T1 요청 및 CPU 왕복 시간.

메모리는 phase의 관측값과 프로세스별 peak입니다. 두 프로세스 peak를 더해 동시 사용량으로
해석하지 않습니다. 실제 startup·model load OOM은 학습 로그 생성 전 발생할 수도 있습니다.
속도 보고 시 GPU·dtype·seed·microbatch·누적·동기화·warm-up 제외 범위·실제 측정 step 수와
검증/저장/Hub 시간을 함께 남깁니다. 1-step 파일럿은 warm-up 이후 평균 속도를 제공하지 않습니다.

재개는 같은 source commit·profile·목표 step·데이터·정규화·T1/S0 identity·run group을 사용합니다.
원래 private `VLADE-konkuk/VLADE-checkpoints`의 불변 revision을 `RESUME_REVISIONS`에
`{'S0': '...', 'S1': '...', 'S2': '...'}` 중 선택한 모델만 지정합니다.
seal·optimizer/scheduler/RNG·query cursor와 episode state, `last/best/final`, pending/retry는
[기존 Hub 계약](hugging_face_checkpoints.md)을 그대로 사용합니다. 목표 step 완료 전 final은 만들지 않습니다.

## 검증 범위

로컬 CPU에서 tiny native Qwen vision·merger·hybrid language + 원래 AVA/proprio/action을 사용해
eager/SDPA 출력과 gradient, 실제 checkpoint recomputation, 누적 8·AdamW·seal·저장/재개,
HF 거절/수락과 Teacher/cache·plan 계약의 관련 회귀를 검사합니다.
CPU offload 왕복에서 객체·gradient·moment 보존 및 timeout 종료 순서도 확인합니다.
이 검증은 CUDA 메모리 해제나 BF16 CUDA kernel의 정확도·속도 검증을 대체하지 않습니다.
실제 L4, 0.8B 학습, 온라인 7B T1, 비공개 Hub 실제 전송·복구, 수렴과 LIBERO+ 성공률은 미검증입니다.
노트북 재생성은 `python -B -m scripts.prepare.generate_l4_notebook`으로 수행합니다.
