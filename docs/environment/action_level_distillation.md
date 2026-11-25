# T1 → S0 Action-level Distillation

QwenVLA schema 2 S0에서 시작해 LIBERO+ 시범 액션과 고정 AVA T1의 Action Chunk로 학습합니다.
두 손실은 동일한 정규화 공간의 `[B, 8, 7]` masked L1이며 에피소드 끝의 패딩을 제외합니다.
기본 손실은 `GT L1 + λ × Teacher L1`입니다.

현재 S2의 **중간 표현 + Action Distillation**은
[표현 증류 안내](s2_representation_distillation.md)와
[기존 A100 세션 셀](../../experiments/colab_s2_representation_kd_existing_session.ipynb)을 따릅니다.
S1과 같은 S0·데이터·액션 조건에서 내부 표현 손실만 추가하고 Plan 생성·입력은 사용하지 않습니다.
아래 CoT/Plan 실행 안내는 별도 자연어 Plan 연구 경로입니다.

L4 24GB에서 같은 연구 조건으로 새 실행을 준비하려면 [L4 설정·파일럿 안내](colab_l4_training.md)를
따릅니다. cache가 있으면 검증하여 재사용하고, 온라인 T1은 Student와 번갈아 GPU에 적재합니다.

| 실험 | JSON 설정 | Colab |
|---|---|---|
| S1 Action-only 훈련 | [qwen_vla_action_kd.json](../../configs/experiments/qwen_vla_action_kd.json) | [Action KD](../../experiments/colab_action_distillation.ipynb) |
| S2 Action+CoT 훈련 | [qwen_vla_plan_kd.json](../../configs/experiments/qwen_vla_plan_kd.json) | [CoT+Action KD](../../experiments/colab_teacher_plan_kd.ipynb) |

## S2 준비 노트북

[CoT+Action KD 준비 노트북](../../experiments/colab_cot_action_kd_ready.ipynb)은 설정·재개 메모,
T1 probe, checkpoint/Hub metadata 검사와 작은 결과 ZIP을 포함합니다. 실행 소스는
`4b24ece9dbdc5cb690a1bab189c121af85d95967`로 고정되어 있으며 현재 로컬의 미커밋 변경은 사용하지 않습니다.

1. Colab의 BF16 GPU 런타임과 비공개 저장소 읽기·쓰기 권한이 있는 Secrets `HF_TOKEN`을 준비합니다.
   기본 실행은 A100급에서 메모리를 확인합니다. 이 노트북은 L4 전용 설정을 선택하지 않습니다.
2. Plan 지도학습 이력이 있는 게시된 T1을 `TEACHER_CHECKPOINT_PIN`의 `repo_id`, 불변 `revision`,
   `prefix`로 지정합니다. 빈 설정은 고정 registry의 T1을 검사하며, Action-only T1이면 중단합니다.
   준비 노트북 자체가 새 Teacher를 학습하거나 Plan 주석을 만들지는 않습니다.
3. 설정 → 소스 → 계획/재개 메모 → setup → preflight → probe 순서로 실행합니다.
   probe는 Train·Validation 양쪽의 플랜과 Train의 `[8, 7]` 액션 형태를 검사합니다.
4. probe 원문을 확인한 뒤 `RUN_PILOT=True`로 1-step 파일럿을 실행합니다.
   로컬 봉인·손실 항·gradient와 HF의 고정 revision에 있는 pointer·완료 표시·manifest를 대조합니다.
   원격 가중치 전체를 재다운로드해 검사하는 것은 아닙니다.
5. 파일럿 검사 통과 후 `RUN_TRAIN=True`로 같은 run의 목표 step까지 이어갑니다.
   재개 메모와 최종 `CHECKPOINT_PINS`를 보관합니다. 업로드 실패는 원본을 보존하고 retry 셀로 복구합니다.

`RUN_PILOT`과 `RUN_TRAIN`은 기본 False입니다. 노트북의 CPU 검사 통과는 실제 GPU 학습이나
T1 플랜의 의미 품질을 입증하지 않습니다. 아래 CoT 조건과 별도 rollout 평가를 함께 확인하십시오.

## 실행 및 재개

두 노트북은 Git commit을 확정하고 설정을 읽어 [run_distillation.py](../../scripts/train/run_distillation.py)를 호출합니다.
Colab Secrets의 `HF_TOKEN`에 비공개 모델 저장소 읽기·쓰기 권한을 부여하십시오. 로컬에서는 `hf auth login` 또는 주입된 `HF_TOKEN`을 사용합니다.
Colab에서 사용할 `SOURCE_REF`에는 실행 코드가 포함된 원격 브랜치 또는 commit SHA를 지정합니다.

```bash
python -B scripts/train/run_distillation.py --config configs/experiments/qwen_vla_action_kd.json --workspace /content/vlade_action_kd --run-group action-kd-seed42-r1 --action plan
```

| 액션 | 동작 |
|---|---|
| `plan` | 적용 설정과 출력 경로를 표시합니다. 기본 액션입니다. |
| `setup` | Student 환경과 온라인 Teacher에 필요한 AVA 환경을 준비합니다. |
| `preflight` | 고정 HF 입력의 무결성, 모델·데이터 계약, 비공개 Hub 연결을 확인합니다. |
| `probe` | CoT 설정에서 Student를 로드하지 않고 T1의 train·validation 첫 Episode 쿼리를 각각 최대 2개 확인합니다. 플랜 원문과 train 액션 형태를 기록합니다. |
| `pilot` | GPU에서 optimizer 1 step, 검증, 저장을 실행합니다. |
| `train` | 같은 run의 마지막 봉인 체크포인트에서 이어서 목표 step까지 학습합니다. 최초 실행은 `pilot`을 먼저 실행합니다. |
| `retry` | 원본 체크포인트를 사용해 실패한 Hub 전송을 재시도합니다. |

설정에서 `initial_checkpoint`는 S0, `distillation.teacher_checkpoint`는 T1, `distillation.action_kd_weight`는 λ입니다.
`workflow.registry_revision`에 모델 저장소의 불변 registry commit을 지정합니다.
HF `runs/`에 있는 체크포인트는 `workflow.checkpoint_pins`에 asset 키별 `repo_id`, `revision`, `prefix`를 넣어 연결할 수 있습니다.
`prefix`는 `runs/<group>/<output>/checkpoints/step-0000100`과 같은 형식입니다.
`VLADE_ASSET_REGISTRY`를 공급할 때는 `workflow.registry_revision`을 `null`로 설정합니다.
이미 검증한 로컬 입력은 [읽기 전용 입력 연결](local_hf_inputs.md)을 따릅니다.

S1과 S2는 비공개 HF 저장소 `VLADE-konkuk/VLADE-checkpoints`의
`runs/<run-group>/S1` 및 `runs/<run-group>/S2`에 각각 저장합니다.
각 모델의 목표 step 업로드가 끝나면 `final.json`과 평가용 `CHECKPOINT_PINS`가 생성·출력됩니다.
`CHECKPOINT_PINS`를 S1·S2 평가 노트북에 복사해 같은 가중치를 사용합니다.
새 런타임에서 재개할 때는 기존 `SOURCE_COMMIT`, `RUN_GROUP`, 설정을 유지하고
노트북의 `RESUME_REVISIONS`에 `{'S1': '<HF commit SHA>'}` 또는 `{'S2': '<HF commit SHA>'}`를 지정합니다.
CLI에서는 같은 JSON을 `--resume-revisions`에 전달합니다. `last`에서 복원합니다.
코드·입력·설정이 바뀌면 새 run group을 사용합니다. 목표 step을 완료한 체크포인트가 `final`입니다.

## CoT 조건과 검증

현재 액션 L1로 학습한 T1은 Plan 지도학습이 확인된 체크포인트가 아닙니다. CoT 실행은
호환되는 봉인 `plan_supervision`이 없으면 중단합니다. 원인 범위, 선택적 Plan CE·주석 입력과
새 학습/재개 절차는 [Teacher Plan 안내](teacher_plan_generation.md)를 따릅니다.
기존 T1에서 새 Plan-supervised Teacher를 준비하는 실행본은
[T1 Plan 학습 노트북](../../experiments/colab_teacher_plan_training.ipynb)과 [학습 안내](teacher_plan_training.md)입니다.

T1은 같은 시점의 관측에서 Action Chunk와 짧은 접근 플랜을 제공합니다. Student는 플랜 문자열을 입력으로 받고
액션 L1 손실로 학습합니다. `teacher_plan_max_new_tokens`의 기본값은 Action-only에서 0, CoT+Action에서 48입니다.
T1의 응답 전체가 `<plan>…</plan>`으로 감싸져 있으면 본문을 추출해 기존 평문 규칙으로 검증합니다.
생성 시 첫 EOS를 허용하고 BOS·UNK·PAD 등 EOS 외 특수 토큰과 액션 토큰·추가 vocabulary 행을 억제합니다.
EOS-only와 EOS 없이 예산에 도달한 응답은 거절합니다. EOS를 막아 텍스트를 강제로 만들지 않습니다.
빈 응답·미완성 또는 중첩 태그는 거절합니다. 거절 시 디코딩 원문 앞 256자와 이유를 오류에 포함합니다.
생성된 token ID와 Episode·timestep도 오류에 포함해 조기 종료와 잘못된 텍스트를 구분합니다.
학습에서는 오류를 `teacher-plans-*.jsonl`에 기록하고 중단합니다. 플랜 평가에서는 `plan=None`으로
계속 진행하면서 task ID·timestep·거절 이유를 stderr에 출력하므로, `None`을 유효한 플랜으로 해석하지 마십시오.
CoT+Action은 온라인 T1을 사용합니다. 단독 rollout과 양자화 평가 경로에는 플랜 공급 기능이 필요합니다.

Validation은 held-out GT L1을 계산하며 같은 에피소드에서 건너뛴 쿼리도 순서대로 재생해 recurrent state를 복원합니다.
CoT 실행은 `teacher-plans-*.jsonl`에 시점, 지시, 플랜과 GT·Teacher L1을 기록합니다.
Validation의 Teacher L1은 `null`입니다. 단계별 지표는 `metrics-*.jsonl`, 시간은 `timings-*.csv`에 저장됩니다.
플랜 품질은 파일럿 로그의 원문을 확인하십시오.

CoT는 `preflight → probe → pilot → train` 순서로 실행합니다. `preflight`는 모델 생성을 실행하지 않습니다.
`probe`는 같은 설정의 고정 HF 입력과 AVA 런타임을 사용하며 `teacher-plan-probe-*.jsonl`을
`<workspace>/runs/<run-group>/`에 저장합니다. 형식이 틀리면 오류와 관측 위치를 기록하고 중단합니다.
이 단계는 optimizer·Student 로드·Hub checkpoint 게시를 실행하지 않습니다.
기존 T1 FFT의 최적화 목표는 액션 L1이며 자연어 플랜 생성 손실은 없습니다. 새 선택적 Plan CE는
검증된 별도 주석과 새 run이 필요하며 기존 T1에 소급 적용되지 않습니다.
형식 검사 통과만으로 플랜의 의미·액션과의 일치가 보장되지 않으므로 probe 원문과 아래 rollout 평가를 확인합니다.
생성 실패 시 지시문이나 고정 문장으로 플랜을 대체하지 않습니다.

플랜의 적합성과 효과를 평가하려면 두 LIBERO+ 노트북을 사용합니다.

- [T1 플랜–액션 일치도](../../experiments/colab_t1_plan_alignment.ipynb)는 같은 관측의 플랜과 실제 8-step 시뮬레이터 동작을 기록합니다. `review.csv`, 두 카메라 GIF, 손끝·그리퍼·물체 좌표가 포함된 `chunks.jsonl`을 생성합니다.
- [S1·S2 플랜 효과](../../experiments/colab_s1_s2_plan_effect.ipynb)는 S2의 정상 플랜·과거 플랜·플랜 생략을 S1과 비교합니다. 동일 초기 조건의 성공률 차이는 `paired_success.csv`, 동일 관측·이전 상태에서 플랜만 바꾼 액션 차이는 `action_probes.csv`에 기록합니다.

두 노트북은 `plan → setup → run` 순서로 실행합니다. 기본 평가는 LIBERO+ Spatial의 고정 시드 변형을 사용합니다.
S0 평가에는 같은 시작 S0와 같은 조건으로 훈련한 두 체크포인트를 지정합니다. 결과와 실험 입력 식별자는
`<workspace>/evaluations/<run-name>`의 `results.json`과 `manifest.json`에 저장합니다.

같은 실행 조건에서 λ를 비교할 때는 서로 다른 run group을 사용합니다. 각 최종 step 디렉터리를 전달하면
봉인과 조건을 대조해 held-out GT L1이 가장 낮은 후보를 선택합니다.

```bash
python -B scripts/train/select_plan_kd_weight.py /path/to/run-a/gt_kd/step-0000100 /path/to/run-b/gt_kd/step-0000100
```

기존 통합 노트북의 `qwen_kd` 프로필은 schema 1 QwenStudent용입니다.
그 실행 설정은 [기존 Qwen Action KD 프로필 안내](qwen_action_distillation_colab.md)를 참고하십시오.
