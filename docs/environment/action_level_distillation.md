# T1 추론 → S0 Action-level Distillation

HF 연결부터 실행까지는 [증류 전용 노트북](../../experiments/colab_action_distillation.ipynb)을 사용합니다.
생성 정본은 `src/vlade/utils/action_kd_notebook.py`이며
`python -B scripts/prepare/generate_action_kd_notebook.py`로 재생성합니다.

현재 `qwen_kd`는 HF에서 선택한 **비양자화 AVA T1**을 고정하고 매 train query의
Action Chunk를 추론하여 **전체 S0 checkpoint**에서 시작한 Student를 학습합니다.
Teacher cache는 필요하지 않습니다. 기존 cache 방식은 `teacher_mode=cache`로 사용할 수 있습니다.

근거는 노션 [11-06 Tasks · Action-Level Distillation](https://www.notion.so/40fb5dd0687080008d39d86c89c7af45)의 수식입니다.

\[
L = L_{BC}(A_S,A_{GT}) + \lambda L_{KD}(A_S,A_T)
\]

두 항은 동일한 정규화 액션 `[B,8,7]`의 masked L1입니다. 유효한 액션 원소만 평균하며
Episode 말미 padding은 제외합니다. `task_weight=1.0`, `action_kd_weight=λ`이고 기본 λ는 1.0입니다.
이는 튜닝 결과가 아닌 시작 설정입니다. T1은 `eval`, `requires_grad=False`, `inference_mode`로
호출하고 타깃을 detach합니다. Action-only 설정의 validation은 held-out GT L1을 계산합니다.

## 타임스텝별 Teacher 플랜과 Action KD

CoT 설정에서는 정책 쿼리마다 T1이 접근 플랜과 action chunk를 제공합니다.
쿼리 간격은 8이며 각 \(F_T(t)\)는 \(t\)에서 두 카메라와 proprio, 작업 지시를 보고
생성한 다음 8개 제어 스텝의 짧은 접근 계획입니다. Student의 언어 입력에 해당
타임스텝과 플랜을 함께 넣고, 최종 action head에 기존 증류 손실을 적용합니다.

\[
A_S(t) = S_\theta(I_t, L, F_T(t); p_t), \qquad
\mathcal L = \mathcal L_{gt}(A_S(t), A_{GT}(t)) +
\lambda\mathcal L_{KD}(A_S(t), A_T(t)).
\]

T1의 Llama 디코더가 관측에 조건부로 플랜 텍스트를 생성합니다. AVA의 기존 action
경로는 동일 쿼리에서 연속 action chunk를 계산합니다. \(p_t\)는 기존 proprio 입력이며
모델에 따라 이미지 히스토리나 AVA 상태도 사용합니다. 플랜은 조건 입력으로 쓰고,
텍스트 생성 손실은 추가하지 않습니다. Validation은 같은 방식으로 생성한 플랜을 입력에
포함하고 GT L1만 기록합니다. 모델의 플랜 생성 품질은 실제 T1에서 별도로 확인해야 합니다.

- Native schema 2 S0는 [CoT+KD 설정](../../configs/experiments/qwen_vla_plan_kd.json)을 사용합니다.
  `teacher_plan_max_new_tokens`의 시작값은 48입니다. 기존 Action-only 설정에서 이 값은 0입니다.
- [CoT+Action KD Colab](../../experiments/colab_teacher_plan_kd.ipynb)은
  `VARIANTS='gt_kd'`, `TEACHER_PLAN_MAX_NEW_TOKENS=48`을 기본으로 사용합니다.
  단독 롤아웃과 양자화 평가 진입점에는 플랜 공급 기능이 아직 없습니다.
- 기존 schema 1의 `qwen_kd`에서도 `distillation.teacher_plan_max_new_tokens=48`로
  켤 수 있습니다. 플랜 경로는 온라인 T1을 사용합니다.

### 플랜 품질 확인

Colab의 `Teacher 플랜 로그` 셀에서 파일럿 실행 직후부터 split별 쿼리 수, 생성 오류
건수, 평균 GT L1 및 Teacher L1과 플랜 원문을 확인할 수 있습니다.
`teacher-plans-*.jsonl`의 각 행에는 에피소드, 타임스텝, 지시문, 플랜과 해당 쿼리의
action 오차가 들어갑니다. 검증 행의 `action_kd_l1`은 `null`입니다. Action L1은
Student가 예측한 action의 오차이므로, 플랜이 지시와 관측 장면에 맞는지는 원문을
보고 판단합니다. `DOWNLOAD_LOGS=True`로 플랜 로그, step별 학습 지표와 시간 CSV를
ZIP으로 내려받을 수 있습니다.

### λ 선택

같은 초기 S0, T1, 데이터, seed, 학습 범위와 step 수로 `0.25`, `0.5`, `1.0`을
서로 다른 run group과 output에 실행합니다. 각 실행은 훈련 전에 고정된 λ를 사용합니다.
Colab에서는 새 `RUN_GROUP`에 원하는 `KD_WEIGHT`를 지정합니다. 같은 Colab 런타임에
후보 checkpoint가 있는 경우 `완료한 λ 후보 비교` 셀에 후보 그룹을 입력합니다. Native CLI에서는
공통 설정에 `--action-kd-weight`로 각 값을 지정합니다. 각 실행의 목표 step 경로를
전달하면 일치하는 실행 조건과 checkpoint 봉인을 확인하고 validation GT L1이 가장
낮은 λ를 선택합니다.

```bash
python -B scripts/train/select_plan_kd_weight.py \
  experiments/runs/plan-kd-025/step-0000100 \
  experiments/runs/plan-kd-05/step-0000100 \
  experiments/runs/plan-kd-10/step-0000100
```

## 가중치 선택

`configs/research/qwen_kd.json`에서 다음 값을 바꿉니다.

| 설정 | 의미 | 기본값 |
|---|---|---|
| `checkpoint` | 전체 S0 가중치·학습된 액션 헤드·processor | `hfasset://student_s0` |
| `initialization` | S0 전체 가중치로 새 증류 실행 시작 | `student_checkpoint` |
| `distillation.teacher_checkpoint` | 추론에 사용할 T1 전체 FFT 가중치 | `hfasset://teacher_t1` |
| `distillation.teacher_mode` | 고정 Teacher 직접 추론 | `inference` |
| `distillation.task_weight` | Demonstration L1 계수 | `1.0` |
| `distillation.action_kd_weight` | Teacher L1 계수 λ | `1.0` |
| `distillation.teacher_timeout_seconds` | Teacher 로딩·개별 추론 응답 제한 | `600` |
| `distillation.teacher_device` | Teacher를 올릴 장치 (기존 AVA BF16) | `cuda:0` |
| `teacher_runtime` | 별도 AVA 추론 환경 | `ava` |

HF의 실제 `repo_id`, **불변 commit revision**, 파일 경로·크기·SHA256은 기존
`configs/storage/hf_assets.lock.json` 또는 `registry_revision`으로 고정한 원격 registry에 등록합니다.
위 두 asset 키는 선택 자리이며 **현재 committed lock에는 등록되어 있지 않습니다**.
전용 노트북에서는 기존 HF `runs/<group>/<output>/checkpoints/step-...` 경로와 불변 revision을
입력하여 `pin_checkpoint_inputs`로 런타임 registry에 연결할 수 있습니다. 작은 seal·metadata와
원격 LFS hash를 대조하며 모델 다운로드·재업로드·원격 registry 변경을 하지 않습니다.
실제 모델 입력은 이후 `resolve_input`이 `hub_checkpoint` 전체 파일의 크기·SHA256을 다시 검증합니다.
이미 게시된 asset 키를 선택하는 기존 방식도 유지합니다. 임의 revision이나
T0/사전학습 Qwen을 T1/S0로 대체하지 않습니다. [HF 입력 안내](hf_inputs_and_git_source.md)를 따릅니다.
URI와 이미 받은 로컬 캐시 모두 기존 `resolve_input`을 통과합니다.

S0는 기존 `QwenStudent` schema 1 (`model/backbone`, `model/action_head.pt`,
`model/student.json`, `processor`, `identity.json`, 봉인·`COMPLETE`) 형식입니다.
schema 2의 별도 `QwenVLA` 모델은 다른 구조이므로 명확히 거부합니다.
S0의 모델 설정·데이터 identity와 현재 입력을 대조합니다. T1은 완료된 AVA FFT checkpoint이며
데이터·split·정규화가 일치해야 합니다. 이 검사 자체가 validation 성능에 따른 T1 선정을 대신하지 않습니다.

S0 초기화에는 과거 optimizer·scheduler·query cursor를 복원하지 않습니다. 기존 KD 실행을
이어갈 때만 `resume`을 사용하며, 그 경우 원래 설정·초기 S0·T1·코드·데이터 identity 검사를 유지합니다.

## 실행

통합 `experiments/colab_vlade.ipynb`에서 `PROFILE='qwen_kd'`를 선택합니다.
setup은 Student와 AVA 환경을 각각 준비합니다. 두 모델은 별도 프로세스로 실행하여
서로 다른 Transformers 의존성을 유지하며, 같은 GPU에 동시에 올라갑니다.
실제 실행 전 파일럿에서 합산 VRAM과 추론 시간을 확인해야 합니다.

```python
PROFILE = 'qwen_kd'
OVERRIDES = {
    'checkpoint': 'hfasset://student_s0',
    'distillation.teacher_checkpoint': 'hfasset://teacher_t1',
    'distillation.action_kd_weight': 1.0,
}
```

실제 등록 키·registry revision을 확인한 뒤 계획 → setup → 파일럿 → 본 학습 순서를 따릅니다.
스위치 기본값은 False이고 설정만으로 다운로드·GPU 실행을 시작하지 않습니다.

Teacher는 같은 Episode 안에서만 시간 상태를 유지합니다. shuffle로 Episode가 바뀌거나
같은 Episode의 첫 query로 돌아가면 reset합니다. 재개 시 중간 timestep부터 요청되면
Episode 시작부터 해당 query까지 Teacher만 재추론합니다. validation/test는 Teacher action 타깃을 생성하지 않습니다.
Teacher 입력에는 원본 지시문·두 카메라·proprio만 사용하며 GT 액션은 forward에 넣지 않습니다.

기존 공통 optimizer loop·gradient accumulation·validation·봉인 checkpoint·private Hub 전송을 사용합니다.
실제 T1/S0 가중치 다운로드, GPU forward/backward, VRAM, 성공률 검증은 미실행입니다.
