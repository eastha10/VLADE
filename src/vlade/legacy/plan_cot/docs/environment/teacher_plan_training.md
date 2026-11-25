# T1 Plan 학습 실행

[전용 Colab 노트북](../../experiments/colab_teacher_plan_training.ipynb)은 기존 Action-only T1을
초기 가중치로 사용해 새 AVA FFT 실행에서 `Action L1 + λ × Plan CE`를 학습합니다.
학습 구현은 기존 `training/fft.py`와 `teacher_plan_supervision.py`이며 새 실행기는 설정·주석·
환경·Hub 연결을 담당합니다. Student S2 학습은 [CoT+Action KD 준비 노트북](../../experiments/colab_cot_action_kd_ready.ipynb)에서 별도로 실행합니다.

## 실행 전에 필요한 것

- 이 구현을 포함한 원격 Git commit과 BF16 GPU Colab 런타임. 새 노트북의 `SOURCE_REF`에
  해당 commit 또는 포함된 브랜치를 지정합니다. 실행 시 확정한 `SOURCE_COMMIT`을 보관합니다.
  기존 실행 소스는 덮어쓰지 않습니다. 현재 로컬 변경만으로 원격 Colab 소스가 갱신되지는 않습니다.
- Colab Secrets `HF_TOKEN`의 비공개 model `VLADE-konkuk/VLADE-checkpoints`와
  dataset `VLADE-konkuk/libero-plus-prepared` 접근·쓰기 권한.
- 고정 HF의 `T1`, `prepared_d_seed42`, 검토된 Plan 주석 자산.
  기본 주석 키는 `teacher_plans_d_seed42`이며 실제 주석을 포함한 registry의 불변 revision을 지정합니다.
  주석이 없으면 아래 준비 절차를 먼저 실행합니다. 기본 registry 설정만으로 주석이 만들어지지는 않습니다.

전체 7B Teacher와 추가 LM forward를 사용합니다. 실제 GPU·RAM·디스크 적합성은 파일럿에서
확인해야 합니다. 이 실행은 단일 BF16 GPU용이며 앞선 두 GPU FP16 경로를 자동 선택하지 않습니다.

## 1. 실제 Plan 주석 준비

설정 → 계획 → setup을 실행한 뒤 `EXPORT_TEMPLATE=True`로 빈 Train 주석을 내보냅니다.
출력은 `<workspace>/annotations/<run-group>/`입니다.

| 파일 | 내용 |
|---|---|
| `plans.jsonl` | 모든 Train query의 split·episode_id·timestep·원문 instruction·빈 plan |
| `observations.jsonl` | 같은 시점의 두 카메라 상대 경로·정규화 proprio·원본 HF 입력 참조 |
| `manifest.json` | prepared manifest·split·normalization 식별자, `annotation_source=unreviewed` |

원본 데이터·이미지를 복제하지 않습니다. 검증된 prepared 캐시에 있는 해당 이미지와 지시를
확인하여 영어 접근 플랜을 작성합니다. 현재 시점의 관측으로 다음 8 control step의 목표·이동·
그리퍼 의도를 설명합니다. 미래 프레임·미래 액션·Validation/Test 정답을 주석 생성에 쓰지 않습니다.
실패한 T1 출력이나 지시문 복제를 정답으로 채우지 않습니다.

각 Plan 본문과 EOS는 실제 Teacher tokenizer에서 128 token 이내여야 하며 학습 시 특수·액션
토큰과 초과 길이는 거절됩니다. 문장 형식·coverage 검사만으로 의미 품질을 보장하지 않습니다.
현재 생성 예산 48을 쓰는 S2에서도 충분히 생성되도록 짧은 플랜을 작성하고 이후 probe로 확인합니다.

검토 후 `REVIEW_SOURCE='human_reviewed'` 또는 `'externally_reviewed'`를 지정하고
`PUBLISH_REVIEWED=True`로 게시합니다. 모든 Train query coverage, instruction 일치, 중복·누락·
범위 밖 시점·빈 플랜을 확인한 뒤 기존 private asset 업로더를 호출합니다. 이 값은 실제 검토
이력의 사용자 선언이며 자동 의미 평가가 아닙니다. 관측 참조 파일은 게시하지 않습니다.
출력된 `ANNOTATIONS_REGISTRY_REVISION`으로 설정을 바꾸고 **새 학습 run group**을 사용합니다.
변경된 주석은 새 asset 키로 게시하여 기존 HF 입력을 보존합니다.

독립 주석 CLI도 같은 구현을 사용합니다.

```bash
python -B src/vlade/legacy/plan_cot/scripts/data/prepare_teacher_plans.py template --data hfasset://prepared_d_seed42 --draft <새-작성-폴더>
python -B src/vlade/legacy/plan_cot/scripts/data/prepare_teacher_plans.py validate --data hfasset://prepared_d_seed42 --draft <작성-폴더> --review-source human_reviewed
python -B src/vlade/legacy/plan_cot/scripts/data/prepare_teacher_plans.py publish --data hfasset://prepared_d_seed42 --draft <작성-폴더> --review-source human_reviewed --asset-key teacher_plans_d_seed42
```

외부 prepared 입력은 읽기·출력 생성 전에 `resolve_input`으로 확인합니다. 작성 중인 Plan은
현재 준비 절차의 편집 출력이며, 학습에는 게시·고정 HF 검증을 마친 주석만 전달합니다.
주석 CLI의 publish는 사용자 실행 시 HF dataset과 model registry에 씁니다.

## 2. T1 초기화와 사전 검사

학습 설정은 [teacher_plan_training.json](../../configs/experiments/teacher_plan_training.json)입니다.
`initial_checkpoint='hfasset://T1'`은 이전 action 가중치를 가져오는 **새 목적의 초기화**입니다.
원본 T1을 `--resume`으로 넘기지 않습니다. 원본에 Plan 학습 이력이 없어도 초기화 입력으로는 허용합니다.

`preflight`는 데이터·주석·T1의 고정 HF 무결성, 원문 instruction과 전체 Train 주석 coverage,
기존 canonical 분할·정규화 및 private Hub 연결을 검사합니다. 주석이 registry에 없으면 모델
다운로드 전에 중단합니다. 기존 S2의 Plan 감독 검사·입력 검증·seal/재개 조건은 유지합니다.

```bash
python -B src/vlade/legacy/plan_cot/scripts/train/run_teacher_plan.py --workspace <workspace> --run-group <고유-run-group> --action plan
```

같은 명령에 `--action setup`, `template`, `publish-annotations`, `preflight`, `pilot`, `train`,
`retry`, `export-pin`을 선택합니다. `--overrides`는 JSON의 기존 키를 점 표기로 바꿉니다.
registry revision을 바꿀 때는 `{"registry_revision":"<주석을 포함한 HF commit SHA>"}`를 사용합니다.
설정·소스 변경은 새 run group입니다. plan은 다운로드·설치·GPU 실행·출력 생성을 하지 않습니다.

## 3. 별도 파일럿과 본 학습

| 실행 | 초기화·목표 | 비공개 model의 게시 위치 |
|---|---|---|
| `pilot` | 원래 T1에서 1 optimizer step, warmup 0, 저장·검증 1 | `runs/<group>-pilot/T1_plan_pilot` |
| `train` | 원래 T1에서 설정한 목표 step까지 새 학습 | `runs/<group>/T1_plan` |

파일럿은 본 학습을 시작하기 위한 실행·저장 확인용입니다. 본 학습은 pilot optimizer나
변경된 1-step schedule을 이어받지 않습니다. 새 main은 pilot의 로컬 seal과 HF 완료 게시를
확인한 뒤 원래 T1에서 시작합니다. 동일 main의 재개에서는 기존 optimizer/scheduler/RNG와
K4 cursor·주석 identity 검사를 그대로 사용합니다.

기본 시작값은 λ=1, LR=2e-5, 누적 8, 1,000 optimizer step, K4/detach2,
cosine/warmup100, 저장·검증 100 step입니다. 최적값이나 수렴이 확인된 설정이 아닙니다.
기존 full scope·paged AdamW8bit·액션 손실은 유지하며 별도의 Plan CE를 더합니다.
검증 `val_l1`과 best는 held-out **액션 L1**입니다. Train loss는 복합 손실이며 Plan 의미나
액션과의 일치·rollout 성공률을 직접 측정하지 않습니다.

## 4. 저장·재개·S2 연결

학습 가중치와 trainer state, 실제 Plan 감독 이력을 기존 seal·Hub checkpoint에 함께 저장합니다.
본 학습의 마지막 목표 step만 main의 final입니다. 예산 종료·실패 실행을 final로 처리하지 않습니다.
전송 실패는 원본과 `hub_pending.json`을 보존하고 `retry`로 해당 pilot/main의 큐를 재시도합니다.

새 런타임에서는 같은 `SOURCE_COMMIT`, `RUN_GROUP`, 설정과 workspace를 사용하고
`RESUME_REVISION`에 main의 last를 포함한 불변 HF SHA를 지정합니다.
setup → preflight → train 순서로 복구합니다. 복구된 main checkpoint가 있으면 pilot을 다시
학습하지 않습니다. 다른 설정이나 Action-only 목적의 optimizer 재개는 허용하지 않습니다.

main 완료 후 `export-pin`은 로컬 봉인·Plan 감독 계약과 고정 원격 revision의 final pointer,
manifest·file SHA256/LFS identity를 검사합니다. 원격 가중치 전체 재다운로드 검사는 아닙니다.
`<workspace>/runs/<group>/T1_plan-handoff.json`에 다음 형식으로 실제 결과를 저장·출력합니다.

```python
TEACHER_CHECKPOINT_PIN = {
    'repo_id': 'VLADE-konkuk/VLADE-checkpoints',
    'revision': '<검증된 불변 HF commit SHA>',
    'prefix': 'runs/<group>/T1_plan/checkpoints/step-0001000',
}
```

이를 새 S2 run의 준비 노트북에 지정하고 preflight → **실제 Plan probe** → pilot → train을
실행합니다. 파일럿 checkpoint를 main 완료 Teacher처럼 연결하지 않습니다.
Plan 학습 이력은 의미 품질의 증명이 아니므로 [Plan–액션·S2 효과 평가](action_level_distillation.md)의
held-out 생성·rollout을 별도로 수행합니다.

이 작업에서 실제 Plan 주석 작성·HF 게시·GPU 학습은 실행하지 않았습니다.
