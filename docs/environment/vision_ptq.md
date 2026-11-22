# Vision Encoder 전용 PTQ

Teacher T1의 SigLIP·DINOv2 attention/MLP Linear에 가중치 전용 Fake Quantization을
적용합니다. `patch_embed.proj` Conv2d는 선택 사항입니다. LLM, vision projector,
AVA, action/proprio head, LayerNorm, bias는 양자화하지 않습니다.
이는 INT4/INT8 커널 또는 추론 가속 구현이 아닙니다. 원본 모델을 복제하거나
수정하여 저장하지 않으며 적용 중 예외가 발생해도 선택된 가중치를 FP로 복원합니다.

`configs/quantization/vision_ptq.json`의 A0~A6은 각각 FP, W8, W4, Random Mixed,
Visual Mixed, AVA Mixed, Combined Mixed입니다. `ablations`에 method와
alpha/beta 항목을 추가하면 Python 수정 없이 조건을 추가할 수 있습니다.
A3~A6은 동일한 대상 그룹과 `target_avg_bits`를 사용합니다.
`--method mixed`는 Combined Mixed의 별칭입니다.

Per-output-channel symmetric quantization에서 범위는 ±(2^(b-1)-1)이며
연산은 float32, 저장 정밀도는 원래 dtype을 유지합니다. 그룹은 Linear/Conv
가중치 하나입니다. 민감도 측정 때 해당 그룹만 W4 또는 W8로 바꾸고
다른 그룹은 FP로 둡니다. Visual loss는 FP feature Frobenius norm으로 정규화한
제곱 오차, AVA loss는 log 중요도의 평균 제곱 오차입니다. epsilon은 1e-8입니다.
AVA loss에는 실제 attention에 사용되는 sink 처리 후 중요도를 사용합니다.
FP query의 pixels, text, 이전 FP 시간 문맥을 한 번 저장하고 모든 probe에
같이 사용합니다. 양자화 probe로 시간 문맥을 갱신하지 않습니다.
Episode마다 FP 문맥을 초기화하고 첫 query부터 순서대로 생성합니다.
FP 문맥 준비에는 기존 Teacher 추론을 사용하지만 action 지표는 계산하지 않습니다.

`batch_size`는 동시에 보관하는 FP reference query 수입니다. 서로 다른 길이의
text와 문맥을 유지하기 위해 내부 forward는 query별로 수행합니다.
`calibration_episodes`와 `max_queries_per_episode`는 기존 Train 내부의
`is_calibration=true` episode와 앞쪽 연속 query에만 적용됩니다.
Split·정규화·이미지 전처리는 기존 구현을 사용합니다.

비트 예산은 선택된 가중치의 파라미터 수 n에 대한 `sum(n*b)`입니다.
선택 밖의 FP 파라미터·bias·scale 메타데이터는 이 예산에 포함하지 않습니다.
모두 W4에서 시작해 `(S4-S8)/(4*n)`이 큰 그룹부터 W8로 올리며,
예산에 들어가지 않는 그룹을 건너뛰고 다음 그룹을 확인합니다.
음수 또는 0 이득은 업그레이드하지 않습니다. Greedy 근사이며 전역 최적해를
보장하지 않습니다. Random Mixed는 동일 예산 안에서 seed로 섞은 순서로 배정합니다.
그룹 크기 때문에 최종 평균 비트와 예산 사용량은 조건마다 다를 수 있습니다.

## HF 입력과 검증

기존 `configs/storage/hf_assets.lock.json`은 변경하지 않습니다.
2026-11-21 확인한 원격 registry revision
`195d32e54c7f48960411c32963d91746183de488`에서 T1과 prepared_d_seed42 항목만
`configs/quantization/hf_assets.json`에 고정했습니다.
T1은 `VLADE-konkuk/VLADE-checkpoints`의 `assets/T1`, revision
`1fdc672416f2129624b22022e6977345a3b429e3`이며 step 1975입니다.
데이터 revision은 `6d2542577647a9e291554524af4e1dd0928e9df6`입니다.
기존 `ava_fft_step_1975` 경로의 유효성을 가정하거나 T0로 대체하지 않습니다.
기존 asset도 revision `c685b31d7593128ac2f45ff91cf4629e57ba266a`에서
유효함을 메타데이터로 확인했으며, 기존 asset과 현재 T1의 파일 29개
크기·SHA256 목록은 동일합니다. 경로와 revision을 함께 구분하여 사용합니다.

`--dry-run`은 HF 메타데이터와 작은 config/index/봉인 JSON만 읽습니다.
가중치·데이터 다운로드, 모델 생성, GPU 연산, 결과 디렉터리 생성은 하지 않습니다.
revision, 파일 크기, LFS SHA256, metadata SHA256, T1 step/봉인과 calibration
identity 및 대상 모듈 이름을 검사합니다. 모든 payload의 실제 SHA256 검증은
실행 시 기존 `resolve_input`과 `teacher_identity`가 수행합니다.
`--dry-run --offline`은 설정과 로컬 registry만 검사하며 원격 입력 검증을
완료했다고 해석하면 안 됩니다. HF 인증은 환경 변수 또는 Colab Secrets
`HF_TOKEN`을 사용합니다. 다운로드된 기존 캐시는 수정하지 않습니다.

```bash
python scripts/quantization/run_vision_ptq.py --config configs/quantization/vision_ptq.json --dry-run
python scripts/quantization/run_vision_ptq.py --config configs/quantization/vision_ptq.json --dry-run --offline --all
```

## Colab A100 파일럿

기존 Colab의 Git 소스·HF 인증·AVA 환경 준비 절차를 사용하고, 이번 변경이
포함된 `dev/fish13031e`의 commit을 고정합니다. 실행 중인 소스는 교체하지 않습니다.
노트북을 수정할 필요가 없습니다. 체크포인트와 데이터는 동일한 고정 HF 입력을
사용하며, 최초 실행은 기존 resolver의 캐시 준비 다운로드를 수반할 수 있습니다.
로컬 검증에서는 이 다운로드와 GPU 실행을 수행하지 않았습니다.

```bash
python scripts/quantization/run_vision_ptq.py --config configs/quantization/vision_ptq.json --pilot --method combined --run-group vision-ptq-a100-seed42-pilot
```

`--pilot`은 최대 2 episode × 2 query로 제한합니다. 전체 설정의 단일 방법은
`--method visual`처럼 선택하며, A0~A6은 `--all`로 각각 독립 실행합니다.
예: `--pilot --all`은 각 ablation을 소량 입력으로 실행합니다.
각 실행은 `run_group`에 ablation ID(전체 실행 시), UTC 시각, 무작위 접미사를
붙여 기존 출력을 덮어쓰지 않습니다. 기본 출력은
`runs/quantization/<run_group>/`의 config, manifest, sensitivity,
bit_allocation, quantization_summary JSON 다섯 개입니다.
고정 FP checkpoint와 bit allocation을 다시 적용하면 결과를 재구성할 수 있어
가중치는 중복 저장하지 않습니다. FP 결과의 bit allocation은 빈 객체이며
민감도를 계산하지 않는 방법의 sensitivity도 빈 객체입니다.

`upload=false`가 기본입니다. 명시적으로 true로 설정하면 결과 JSON만
비공개 checkpoint 저장소의 `runs/quantization/<run_group>/`에 게시합니다.
원본 checkpoint 경로는 변경하지 않습니다.

CPU 합성 검사는 `.harness/v1/2026-11-21-vision-ptq/checks/`에 있습니다.
실제 GPU 실행, 전체 Calibration, Action MAE, rollout/성공률, 속도와 최종
모델 성능 비교는 이번 구현 검증 범위에 포함하지 않습니다.
