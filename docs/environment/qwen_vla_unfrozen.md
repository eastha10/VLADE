# QwenVLA 백본 학습 범위 비교

[`colab_qwen_vla_unfrozen.ipynb`](../../experiments/colab_qwen_vla_unfrozen.ipynb)는 schema 2 S0에서 시작하는 GT-only/온라인 GT+KD 비교용입니다. 기본값은 `full`, GT-only 100 optimizer step, 누적 8, LR 2e-5, 25스텝 저장입니다. 두 실행을 비교하려면 `VARIANTS='both'`를 선택합니다. 비교군은 같은 S0·데이터·seed·학습 범위·예산을 사용하고 KD 항만 다릅니다. 두 실행은 GPU 메모리를 공유하지 않도록 순차 별도 프로세스로 실행합니다.

| `trainable_scope` | 학습 대상 | `top_layer_count` |
|---|---|---|
| `full` | 백본 포함 전체 Student | 0 |
| `policy` | AVA·proprio projector·액션 헤드, 백본 동결 | 0 |
| `top_layers` | 정책 모듈 + 마지막 N개 language decoder 층과 final norm | N ≥ 1 |

설정에 범위 키가 없으면 기존 동작을 유지합니다. S0에서 초기화한 GT/KD는 policy, pretrained에서 시작하는 S0는 full입니다. 새 노트북은 full을 명시합니다. schema 차이 때문에 동결하는 것이 아니며 모델 구조나 저장 schema는 변경하지 않습니다.

## 실행

Teacher의 타임스텝별 접근 플랜까지 Student에 입력하려면 `VARIANTS='gt_kd'`와
`TEACHER_PLAN_MAX_NEW_TOKENS=48`을 지정합니다. λ 선택 방법과 평가 조건은
[CoT+Action KD 안내](action_level_distillation.md#타임스텝별-teacher-플랜과-action-kd)를 따릅니다.

1. 새 범위 선택과 시간 계측 소스가 포함된 Git ref를 `SOURCE_REF`에 지정합니다. branch는 처음 한 번 commit으로 확정하며 출력된 `SOURCE_COMMIT`을 보관합니다. 미배포 소스에서는 입력 다운로드 전에 기능 검사가 실패합니다. 미커밋 코드를 노트북에 내장하지 않습니다.
2. Colab A100에서 Secrets `HF_TOKEN`을 연결합니다. 팀 비공개 모델 저장소의 읽기/쓰기 권한이 필요합니다. 기본 registry는 S0 게시 완료 revision `65799b213837eeb9b931a08d8fe6625a5fce43e4`, asset은 `S0`, `T1`, `prepared_d_seed42`입니다. 모든 외부 입력은 기존 `resolve_input` 검증을 통과합니다.
3. Student 환경 및 선택 시 AVA 환경을 준비하고 실행 계약/입력 검사를 수행합니다. 기존 HF checkpoint 업로드와 검증을 사용하며 Drive 데이터·가중치를 대신 연결하지 않습니다.
4. 실제 optimizer 1-step 파일럿 셀을 먼저 실행합니다. 로그의 `trainable_scope`, `backbone_trainable_parameters`, 메모리/손실 및 Hub pending을 확인한 뒤 본 학습 셀로 이어갑니다. full GT+KD가 OOM이면 실행 조건을 재검토하며 자동으로 동결 모드로 바꾸지 않습니다.

full FP32 AdamW는 백본 동결보다 메모리를 더 사용합니다. RTX 5060 8GB 전체 학습을 보장하지 않습니다. 100스텝은 비교의 시작 예산이며 성공·수렴·전체 S0 학습 완료를 뜻하지 않습니다. 이 노트북은 롤아웃 평가를 실행하지 않습니다.

## 재개와 CSV

새 실험은 새 `RUN_GROUP`을 사용합니다. 같은 실험 재개는 최초의 그룹·Git commit·registry·설정을 유지합니다. 로컬 런타임에는 검증할 최신 checkpoint가 있으면 이어서 사용합니다. 새 Colab에서는 `RESUME_HF[variant]`에 기존 `<group>-<variant>/<variant>`와 불변 HF commit SHA를 지정합니다. `last`를 복구한 뒤 기존 trainer가 설정/소스/정규화/optimizer identity를 검사합니다. 이미 목표 step까지 완료된 checkpoint도 봉인을 검증한 뒤 건너뜁니다.

기존 동결 실행을 full 실행으로 바꾸어 재개하지 마십시오. 범위 설정도 기존 run identity에 포함되어 호환되지 않는 재개는 거부됩니다. mode 변경은 동일 S0 가중치에서 시작하는 새 실행입니다. 기존 실행의 optimizer/scheduler를 다른 모드에 억지로 복구하지 않습니다.

로그와 CSV는 런타임의 `WORK/runs/RUN_GROUP/<variant>/`에 저장합니다. [시간 기록 안내](qwen_timing.md)의 T1 모델 추론/요청, S0 학습/검증/저장 구간을 사용합니다. 재개마다 새 CSV를 생성하고 기존 파일을 보존합니다. 마지막 셀에서 CSV만 ZIP으로 내려받을 수 있습니다. 체크포인트는 기존 비공개 Hub 저장을 유지하며, timing CSV는 별도 다운로드 전까지 Colab 런타임 파일입니다. T1 CSV는 GT+KD를 실제 실행했을 때 생성됩니다.

Git 소스 배포, Drive 노트북 반영, 실제 GPU 실행은 각각 별도 상태입니다. 노트북 생성/로컬 검사만으로 A100 실행·모델 성공률을 검증했다고 볼 수 없습니다.
