# 설정으로 실행하는 연구 절차

유지 관리되는 Colab 실행본은 [colab_vlade.ipynb](../../experiments/colab_vlade.ipynb) 하나입니다. 노트북은 Git commit을 확정하고 HF 인증·설정·계획 확인·설치·실행만 호출합니다.
모델, 데이터 처리, 학습, cache, rollout 로직은 프로젝트 Python 모듈에 있습니다.
`third_party/`는 원본 출처를 확인하는 보관본이며 이 실행 경로에는 필요하지 않습니다.
PyTorch·timm·Transformers·시뮬레이터 등 설치된 실행 의존성은 필요합니다.

AVA FFT·Suite-balanced·논문 학습의 연속 구간, 시퀀스별 훈련 상태 초기화와 K4 설정은
[AVA 시퀀스 학습](ava_sequence_training.md)을 따릅니다. 기본 FFT batch 1×누적 8은
8 sequences / 32 query 예측이며, 기존 단일 query 실행과 노출량이 다릅니다.

## Colab

기존 개별 실행 노트북 10개를 통합했습니다. 소스·인증 셀을 실행한 다음
설정 셀의 `PROFILE`을 선택하고 계획·설치·실행 셀을 사용합니다.
18개 `configs/research/*.json`을 모두 선택할 수 있습니다. 모든 작업을 차례로 자동 실행하지 않습니다.

`EXPERIMENT_ID`는 셀 재실행 시 유지되며 기본 run group은 `<EXPERIMENT_ID>-<PROFILE>`입니다.
따라서 같은 task 종류를 사용하는 데이터 준비 작업이나 Student 실험도 출력이 섞이지 않습니다.
Suite pilot4/main·논문 pilot/main은 같은 `EXPERIMENT_ID`를 유지합니다.
새 실험은 새 ID, 재개는 `RUN_GROUP_OVERRIDE`에 최초 run group을 지정합니다.
`OVERRIDES`에는 실제 변경할 설정만 넣습니다. 변경 후 계획 셀을 실행하지 않으면 setup/run은 거부합니다.
실행할 준비가 됐을 때만 `RUN_SETUP`·`RUN_EXPERIMENT`를 True로 바꿉니다.

단일 Episode 진단은 `prepare_one_episode` 이후 같은 WORKSPACE의 `ava_one_step`을 실행합니다.
`fullframes`는 별도 공개 게시 작업입니다. 이미 완료된 데이터 이용에는 재실행이 필요 없습니다.
Teacher 선택·cache는 `select_teacher`·`cache_teacher`를 선택합니다. 실제 검증 보고서와 고정 T1이 필요합니다.


노트북의 소스·인증 셀을 실행한 뒤 다음처럼 사용합니다. `WORKSPACE`는 현재 런타임 SSD입니다.
`RUN_GROUP`은 새 실험마다 고유하게 정하고 재개 시 같은 값을 사용합니다.

```python
from vlade.research.client import job
CONFIG = CODE / 'configs/research/ava_fft.json'
OVERRIDES = {'run_group': RUN_GROUP, 'train.max_steps': 1000, 'train.warmup_steps': 100}
job(CONFIG, action='plan', workspace=WORKSPACE, overrides=OVERRIDES)
job(CONFIG, action='setup', workspace=WORKSPACE, overrides=OVERRIDES)
job(CONFIG, action='run', workspace=WORKSPACE, overrides=OVERRIDES)
```

`plan`은 설치·다운로드·실험을 시작하지 않습니다. 배포된 노트북의 설치·실행 스위치는 기본 False입니다.
실험 실행 전 코드·설정은 Git에 commit되어야 하며, 실행 중 소스를 교체하지 않습니다.
로컬의 미커밋 구현 검증과 정식 연구 실행을 구분합니다. 이 작업에서 commit/push를 자동 수행하지 않습니다.

CLI는 저장소 루트에서 같은 설정을 사용합니다.

```bash
python -m scripts.research.run --config configs/research/ava_suite.json --workspace /content/vlade_work --action plan
```

설치할 때는 `--action setup`, 실행할 때는 `--action run`을 사용합니다.
새 실험 run group은 설정 또는 `--set`으로 지정합니다. 노트북 생성은
`python -m scripts.prepare.generate_colab_notebooks`이며 개별 노트북을 재생성하지 않습니다.
참조용 `notebooks/references/`의 사용자 전달본은 실행본과 별도로 보존합니다.

## AVA K4를 4,000스텝 학습하는 방법

### Suite-balanced: 파일럿 후 본 학습

처음에는 "작업 선택과 설정" 셀의 해당 값을 아래처럼 바꿉니다.
소스·HF 인증 이후 설정 → 계획 확인 → 환경 설치 → 실행 셀 순서로 실행합니다.

```python
PROFILE = "ava_suite"
OVERRIDES = {}  # 기본 pilot4: optimizer 4 step
RUN_SETUP = True
RUN_EXPERIMENT = True
```

pilot4가 완료되면 같은 SOURCE_COMMIT·WORKSPACE·EXPERIMENT_ID·RUN_GROUP_OVERRIDE를
유지한 채, 같은 설정 셀의 값을 아래처럼 바꿉니다.
설정 → 계획 확인 → 실행 셀 순서로 실행합니다. 설치가 끝났으므로 setup은 생략합니다.

```python
PROFILE = "ava_suite"
OVERRIDES = {
    "settings.phase": "main",
    "train.max_steps": 4000,
    "train.warmup_steps": 400,
    "train.save_steps": 250,
    "train.eval_steps": 100,
}
RUN_SETUP = False
RUN_EXPERIMENT = True
```

warmup 400·저장 주기 250·검증 주기 100은 이 실행 예시의 값입니다.
파일럿 4 step과 본 학습 4,000 step은 별도 실행입니다.
main은 초기 T0에서 시작하므로 pilot checkpoint를 resume에 넣지 않습니다.
기존 1,000-step main을 시작한 동일 run group의 목표만 4,000으로 변경하지 마십시오.
그 경우 새 EXPERIMENT_ID로 파일럿부터 실행합니다.

### 일반 AVA FFT를 사용하는 경우

새 실험 ID에서 아래 값을 설정하고 계획 → 환경 설치(필요할 때) → 실행합니다.
일반 FFT는 Suite의 pilot 보고서를 요구하지 않으므로 최소 GPU 파일럿을 먼저 별도 실행으로 확인하십시오.

```python
PROFILE = "ava_fft"
OVERRIDES = {
    "train.max_steps": 4000,
    "train.warmup_steps": 400,
    "train.save_steps": 250,
    "train.eval_steps": 100,
}
RUN_SETUP = False  # 아직 환경이 없으면 True
RUN_EXPERIMENT = True
```

K4·batch 1·누적 8은 기존 기본값을 사용합니다. 4,000은 optimizer step 수이며,
이 설정에서 총 32,000개 시퀀스·128,000개 query 예측에 해당합니다.
복원 추출이므로 고유 데이터 수나 epoch 수를 뜻하지 않습니다.
기본 세션 상한은 설치 포함 8시간입니다. 4,000 step을 그 시간 안에 완료한다는 보장은 없습니다.
예산 종료 시 final로 기록하지 않으며, 같은 4,000-step 목표·원래 설정·run group으로
검증된 Hub checkpoint에서 재개합니다. 기존 settings 비교·Suite pilot 복구 제약도 유지합니다.

## 설정 선택

| `configs/research/` 파일 | 실제 작업 |
|---|---|
| `prepare_inputs.json` | 고정 HF 데이터·가중치 검증 및 캐시 준비 |
| `prepare_one_episode.json` | 기존 canonical Train 한 episode 준비 |
| `fullframes.json` | 승인된 별도 공개 D/seed42 전체 프레임 처리·게시 |
| `ava_one_step.json` | 한 step FFT와 별도 프로세스 재로딩 검사 |
| `ava_fft.json`, `oft_fft.json` | 원래 FFT optimizer·학습·Hub 저장/재개 |
| `ava_paper.json` | AVA K=4/2-frame TBPTT·native LoRA |
| `ava_suite.json` | 명시적 suite label에 따른 균형 샘플링·pilot4/main |
| `select_teacher.json` | 측정 보고서와 이유에 근거한 불변 T1 선택 |
| `cache_teacher.json` | 선택한 T1의 순차 train cache 생성·선택적 전체 게시 |
| `qwen_task.json`, `qwen_kd.json` | 기존 query-head Qwen Task-only / action KD |
| `evaluate_ava.json`, `evaluate_oft.json`, `evaluate_qwen.json` | 고정 LIBERO+ 구현의 paired rollout |
| `evaluate_libero_ava.json`, `evaluate_libero_oft.json` | 원본 LIBERO 조건의 별도 평가 |
| `inference_ava.json` | 실제 가중치의 단일 입력 추론 |

기존 Colab에서 사용하던 Qwen query-head trainer를 유지합니다. 별도 `QwenVLA` 모델 구현을
이 trainer로 학습한 것으로 간주하지 않습니다. CALVIN·representation KD·Jetson 배포는 후속 연구 범위입니다.

훈련량은 `train.max_steps`, 누적은 `train.accumulation_steps`, 저장·검증 주기는
`train.save_steps`·`train.eval_steps`, 학습률·warmup·schedule은 해당 `train` 항목으로 조절합니다.
전체 목표 step은 재개 도중 바꾸지 않습니다. Student는 microbatch 1과 주기적 validation을 유지합니다.
Student 실행은 실제 query 재로딩·단일 episode 과적합을 통과한 뒤 본 학습으로 진행합니다.
`pilot.overfit_steps`, `pilot.max_ratio`, `pilot.only`로 파일럿을 제어하며 실패 보고서를 통과로 바꾸지 않습니다.

평가는 `evaluation.rollouts_per_task`, `tasks_per_suite`(null이면 전체), `task_suites`,
`max_steps`(null이면 suite별 기존 horizon), `open_loop_steps`(1~8), `settle_steps`,
`environment_seed`와 최상위 `seed`로 제어합니다. `checkpoints`에 비교 후보의 HF 키를 지정합니다.
같은 task와 초기 상태로 각 후보를 평가하며 요청 rollout 수를 모두 수행해야 완료입니다.
중단 결과는 미완료로 기록하고 같은 설정으로 재개합니다. 단일 추론 결과를 성공률로 기록하지 않습니다.

## 파일럿과 본 학습

논문 경로는 같은 run group·설정으로 `pilot=true`를 먼저 실행하고 `pilot=false`로 본 학습을 실행합니다.
원래 recipe는 100,000 step·effective batch 256이며 step·학습률 등을 바꾸면 그 변경 조건의 실험입니다.
K=4·detach 2·초기 pretrained OpenVLA 및 action/proprio 계약은 유지합니다.

Suite-balanced는 `settings.phase='pilot4'`와 4 step으로 먼저 실행합니다. 본 학습은
`settings.phase='main'`과 원하는 `train.max_steps`·`warmup_steps`·`save_steps`를 지정합니다.
파일럿 checkpoint는 `suite/pilot4/checkpoints`, 본 학습은 `suite/checkpoints`로 분리되며
Hub run group의 `-pilot4`/`-main` 구분을 유지합니다.

Suite-balanced FFT는 suite별 quota와 선택한 query 집합을 보존하면서 optimizer update 안에
동일한 궤적 그룹이 반복되는 것을 줄입니다. 각 query는 현재 관측만 사용합니다.
Validation·순차 Teacher cache·Student의 기존 시간 순서는 유지합니다. seed와 완료 step으로
새 순서를 재현하며, 이전 순서의 checkpoint는 당시 고정 소스로 재개합니다.

`settings.suite_map='hfasset://suite_map_seed42'`는 작업 지시문과 `spatial/object/goal/libero_10`을
명시적으로 연결한 입력입니다. task 이름에서 임의로 추정하지 않습니다. 기존 Drive 자료의 해당 파일은
확인해 비공개 model 저장소 `VLADE-konkuk/VLADE-checkpoints`에 게시했습니다.
고정 revision은 `ca06ce16238d46387b2b972bedc35a35259a4a4c`이며 HF lock에 등록되어 있습니다.
2914 bytes의 원본 SHA256은 `06de171dd29b07bf5e4fb1283a3d1355a2e96a4169c07c850e9ae45253002efa`입니다.
기존 Drive 원본은 보존합니다. 전체 prepared 데이터와의 task coverage는 실제 학습 사전 검사에서 확인합니다.

## 입력·저장·재개

모든 외부 가중치·데이터·cache는 `resolve_input`으로 HF revision과 무결성을 확인합니다.
Drive나 미등록 로컬 파일로 우회하지 않습니다. 새 HF asset은 게시 후 얻은 레지스트리의 불변
commit을 `registry_revision`으로 지정하면 런타임 전용 레지스트리로 읽습니다.
코드에 이미 고정된 기존 asset은 새 레지스트리로 덮어쓰지 않습니다.
`teacher_cache_t1`, 새 후보 checkpoint 등은 실제 검증·게시가 선행되어야 합니다.

훈련은 비공개 `VLADE-konkuk/VLADE-checkpoints`의 원래 Hub 연결을 유지합니다.
`HF_TOKEN`은 Secrets/외부 환경으로 전달하며 설정 파일에 넣지 않습니다.
cache는 전체 train split의 완료·replay 검증이 끝난 경우에만 비공개 cache dataset으로 게시할 수 있습니다.
선택/평가 입력으로 쓸 checkpoint와 보고서는 검증 후 기존 HF 게시 명령으로 등록하십시오.
게시·등록 자체는 `run`을 실행했다는 이유만으로 성공으로 간주하지 않습니다.

재개는 [Hub 안내](hugging_face_checkpoints.md)의 `download --revision <불변 commit>`으로
검증한 step을 `resume`에 지정합니다. 같은 source commit·설정·run group·출력 폴더를 사용합니다.
실행 세션 예산은 설정의 `budget.hours`이며 설치부터 계산합니다. 예산 종료·실패는 final이 아닙니다.
새 세션에서 같은 실행을 재개할 때는 새 런타임 workspace에 checkpoint를 복구합니다.

## 검증 범위

로컬 CPU 검사는 모델 구성요소·축소 Llama/Qwen·저장/재개·입력 거부/수락·명령 전달과 합성 rollout 계수를 확인합니다.
실제 7B 비전/언어 모델·Colab 패키지 설치·GPU 학습·시뮬레이터 성공률은 별도 실제 실행 검증이 필요합니다.
에이전트 검증 소스는 `.harness/v1/2026-11-11-ava-teacher-code/checks/`에 있으며
실행 캐시·임시 checkpoint는 같은 작업의 `scratch/`에만 둡니다.

Benchmark 소스는 [LIBERO+ 고정 commit](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182)과
[원본 LIBERO 고정 commit](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)을
런타임 전용 Git checkout으로 준비합니다. namespace와 BDDL/초기 상태 파일을 함께 연결하고
원본 10-task suite와 LIBERO+의 수천 variant suite를 task 수·revision으로 구분합니다.
