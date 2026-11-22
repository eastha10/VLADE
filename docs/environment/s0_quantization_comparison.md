# S0 독립 양자화 조건표와 실행기

같은 비양자화 QwenVLA S0에서 각 조건을 새로 시작하는 파일럿 실행기입니다.
GT-only/GT+KD로 추가 학습한 Student도 별도 checkpoint 키와 run으로 평가합니다.
HF에 게시하고 불변 revision·전체 파일 SHA256을 고정한 `hub_checkpoint` record를
registry에 넣고 `checkpoint` 키를 선택합니다. 기존 S0 결과를 새 Student 결과로
재사용하지 않으며, 각 Student마다 BF16과 양자화 조건을 같은 원본에서 다시 시작합니다.
여러 조건의 simulator 자산 추출본을 공유하려면 `VLADE_BENCHMARK_WORK`에 입력
모델·데이터 밖의 runtime 경로를 지정합니다. 기존 archive 지문·추출 파일 검증은
유지하며, 조건별 보고서와 모델 상태는 공유하지 않습니다.
기존 세 스위치를 모두 켜는 순차 조합 평가와 구분합니다. 새로운 공식 연구
결론이나 논문 전체 재현을 주장하지 않으며, validation으로 선정한 최종 S*의
양자화 비교를 S0 결과로 대체하지 않습니다. T1 Vision-only A0~A6 실험도 별도입니다.

정본 설정은 `configs/quantization/s0_comparison.json`입니다. 코드 수정 없이
조건을 추가·제외하거나 seed·calibration/evaluation budget을 변경할 수 있습니다.
기본값 1 seed, calibration 8 query, validation 16 query, warmup 2, rollout 0은
기존 실행기와 연결되는 파일럿 기본값입니다. 본 실험의 확정 budget이 아닙니다.

## 조건표

| ID | 방법 | 양자화 범위 | 대상 가중치 예산 | 비교 묶음 |
|---|---|---|---|---|
| BF16 | 비양자화 원본 | 없음 | 원래 BF16 | 전체 공통 기준선 |
| L8-Uniform | Uniform W8 | language model Linear | 8 bit/weight | 언어부 8비트 |
| L8-ActQuant | ActQuant 파일럿 | 같은 language model Linear | 평균 ≤8 bit/weight | 언어부 8비트 |
| L8-QVLA | QVLA 파일럿 | 같은 language model Linear | 평균 ≤8 bit/weight | 언어부 8비트 |
| L8-QuantVLA | smoothing + W8 파일럿 | 같은 language model Linear | 8 bit/weight | 언어부 8비트 |
| LV4-Uniform | Uniform W4 | language model + visual.blocks Linear | 4 bit/weight | 언어·비전부 4비트 |
| LV4-ActQuant | ActQuant 파일럿 | 같은 language + visual.blocks Linear | 평균 ≤4 bit/weight | 언어·비전부 4비트 |
| LV4-QVLA | QVLA 파일럿 | 같은 language + visual.blocks Linear | 평균 ≤4 bit/weight | 언어·비전부 4비트 |

ActQuant/QVLA는 기존 2/4/8/16비트 할당을 사용합니다. QuantVLA 파일럿은
언어부 W8만 지원하고 DuQuant 회전은 생략합니다. 현재 없는 QuantVLA W4·vision
지원을 만들어 비교하지 않습니다. 기본 조건은 모두 weight-only fake quantization,
activation은 BF16입니다. AVA·action head·proprio projector·vision merger와
선택 밖의 모듈·bias는 원래 정밀도를 유지합니다.

평균 비트 예산은 대상 가중치 파라미터 수로 가중합니다. scale/index·부가 metadata는
제외하므로 파일 저장 크기나 전체 모델 메모리와 같지 않습니다. 결과에 요청 예산과
실제 달성 비트, 대상 모듈/파라미터 수, 비양자화 파라미터를 포함한 전체 모델의
이론적 weight bpw를 따로 남깁니다. 이산적인 할당 때문에 예산을 전부 사용하지
않을 수 있습니다. 비교 묶음이 다르면 단순 순위를 매기지 않습니다.

## 먼저 계획 확인

```bash
python scripts/quantization/run_s0_comparison.py --config configs/quantization/s0_comparison.json --dry-run
```

옵션 없는 기본 동작도 계획 출력입니다. 설정과 로컬 registry의 불변 revision·입력
형식을 확인하며 HF 접속, 모델/데이터 다운로드, 모델 생성, CUDA 연산이나 결과
디렉터리 생성은 하지 않습니다. 원격 payload 확인은 실제 실행 시 기존 HF
resolver가 수행합니다. dry-run 성공을 실제 checkpoint 로딩 성공으로 해석하지 않습니다.

## 명시적인 GPU 실행

연구 소스를 commit한 뒤 기존 CUDA BF16·QwenVLA 환경에서 실행합니다.
`--run`만 실제 입력 준비와 GPU 평가를 허용합니다. 최초 캐시 준비에는 고정 HF
모델·데이터 다운로드가 필요할 수 있습니다. 인증은 환경 변수/Colab Secrets의
HF_TOKEN 또는 기존 Hub 인증을 사용하며 토큰을 설정에 넣지 않습니다.

```bash
# 기본 8개 조건을 각각 독립 실행
python scripts/quantization/run_s0_comparison.py --config configs/quantization/s0_comparison.json --run

# QuantVLA와 Uniform W8만 선택. BF16 기준선은 자동 포함
python scripts/quantization/run_s0_comparison.py --config configs/quantization/s0_comparison.json --condition L8-Uniform --condition L8-QuantVLA --run
```

각 seed의 BF16을 먼저 한 번 실행한 후 각 조건을 별도 Python 프로세스에서
같은 원본 checkpoint로 로드합니다. BF16 action과 query ID를 결과에 저장해
다른 프로세스에서도 같은 validation query의 action 오차를 계산합니다.
Calibration은 원래 Train의 is_calibration episode에 한정하며 기존 split,
이미지 전처리와 normalization은 변경하지 않습니다. query는 episode 앞부터
순차 사용하고 episode 경계에서 문맥을 초기화합니다. 기본 calibration 8개가
한 episode에 집중될 수 있으므로 대표성·seed 수는 본 실험 전에 재설계해야 합니다.

소스 Git SHA/clean 여부, HF model/dataset revision, checkpoint 봉인과 데이터
manifest·split·normalization hash, software/GPU, seed·query ID·비트 배정을 기록합니다.
조건 간 provenance, validation 순서, 같은 비교 묶음의 실제 대상 모듈을 대조하고
할당표에서 실제 평균 비트를 다시 계산합니다. NaN/Infinity는 성공 JSON으로 쓰지 않습니다.

기존 `configs/storage/hf_assets.lock.json`은 보존합니다. 별도
`configs/quantization/s0_assets.json`은 remote registry
`195d32e54c7f48960411c32963d91746183de488`의 S0 revision
`380f8fb1bd828d42e7f396d6979d699b0b7d0cff`와 prepared_d_seed42 revision
`6d2542577647a9e291554524af4e1dd0928e9df6`를 고정합니다. 선택적 benchmark
asset은 기존 storage Lock의 `libero_plus_assets` revision
`dd2bd61b7d9a6fef1abc52d606e983b41886a149`를 그대로 사용합니다.

기존 S0와 prepared 데이터가 이미 있다면 [읽기 전용 입력 등록](local_hf_inputs.md)으로
전수 검증한 뒤 원본 경로를 그대로 연결할 수 있습니다. `VLADE_LOCAL_INPUTS`를 지정하면
worker도 같은 등록을 상속하고 그 hash를 비교 identity에 기록합니다. 등록 정보가 바뀌면
이전 source/config가 같아도 새 실행으로 분리해야 합니다.

## 저장과 재개

기본 결과는 Git 제외 경로 `runs/quantization/s0-comparison/<고유 group>/`입니다.
모델 가중치를 복제하거나 원본 checkpoint를 덮어쓰지 않습니다.

```text
config.json
conditions.json
manifest.json
summary.json
seed-42/BF16/attempt-001/report.json
seed-42/BF16/attempt-001/worker.log
seed-42/L8-Uniform/attempt-001/report.json
...
```

manifest는 조건 시작·완료·실패마다, summary는 각 실행 후 저장합니다. 원자적 JSON
교체로 완료 결과를 보존합니다. 실패 조건이 있어도 다른 독립 조건은 계속 실행하되,
유효한 BF16 기준선이 없으면 종속 비교를 blocked로 남깁니다. 일부 실패를 전체 성공으로
표시하지 않으며 실행기는 exit code 1을 반환합니다.

```bash
python scripts/quantization/run_s0_comparison.py --config configs/quantization/s0_comparison.json --run --resume runs/quantization/s0-comparison/실제-기존-group
```

같은 설정·조건 선택·seed·소스 commit·HF revision에서만 재개합니다. 완료 보고서의
hash와 pairing을 재검증해 건너뛰며 실패 조건만 새 attempt에 실행합니다. 기존 attempt는
덮어쓰지 않습니다. `--output`은 새 출력 디렉터리 지정용이며 resume와 함께 쓰지 않습니다.
모델/데이터/HF 캐시 내부 출력은 거부합니다.

## Rollout과 최종 선정의 경계

기본값은 MAE 파일럿입니다. 설정의 rollouts를 양수로 바꾸면 기존 Spatial paired
rollout을 요청합니다. 고정 `configs/runtime/student_eval.json`의 benchmark source를
기존 runtime 검증기로 검사하고 HF asset을 기존 검증기로 준비합니다. 설치·source·asset
준비 실패나 rollout 실패는 completed와 구분합니다. source commit이 같아도 실제 import가
다른 checkout을 가리키면 거부합니다. 이 파일럿은 LIBERO+ 7개 교란 전체 평가가 아닙니다.

Fake quantization의 latency/메모리는 BF16 실행 및 smoothing hook 비용의 관측값입니다.
packed INT4/INT8 배포 속도나 저장 크기 개선을 뜻하지 않습니다. summary는 자동 승자를
고르지 않으며 selection_eligible=false입니다. 성공률 허용 저하, seed·episode 예산,
target board의 메모리·p95 한계는 본 실험 전에 사전 확정해야 합니다. validation에서
후보를 선정하고 최종 test는 이후 보고용으로 사용합니다.

기존 `scripts/evaluate/run_s0_quantization.py`와 UI의 스위치는 호환성을 위해 순차
조합 동작을 유지합니다. 독립 비교에는 이 문서의 새 실행기를 사용합니다. 기존 실행기도
BF16 완료 결과를 먼저 저장하므로 이후 calibration 실패 시 원인과 BF16 결과를 보존합니다.
