# Qwen 오프라인 액션 증류 · colab05

현재 실행본은 [통합 노트북](../../experiments/colab_vlade.ipynb)이며 `PROFILE`은 `qwen_kd`입니다. 설치·실행 셀은 [연구 실행 안내](research_runs.md)를 따릅니다. 아래의 과거 번호·셀 이름은 이전 실행 기록과 구분하십시오.

> 2026-11-04 이후 로컬/Git 노트북의 실행 기준은 [Git 소스·HF 입력 안내](hf_inputs_and_git_source.md)입니다. 아래 Drive 기반 절차·실행 기록은 기존 버전의 맥락입니다. 진행 중인 세션은 그대로 유지하며 Drive 배포는 별도 명령 후 수행합니다.

`experiments/colab_vlade.ipynb`는 02에서 생성한 고정 T1 cache로
Qwen3.5-0.8B를 학습합니다. 03의 Task-only와 초기 모델 revision, seed, 입력,
episode shuffle/query cursor, accumulation, LR 및 epoch 예산을 공유하며 실행 결과는 분리합니다.
실제 GPU 학습·성공률은 미검증입니다. 04 adapter checkpoint의 cache 생성 연결은 포함하지 않습니다.

## 준비물과 실행 순서

1. 01의 `data/processed/libero_plus_episode80/`와 원본 query 이미지를 유지합니다.
2. 02에서 validation으로 선정한 T1을 사용하여 `RUN_FULL_CACHE=True`로 전체 train cache를 생성합니다.
   기본 입력은 `data/teacher_cache/T1-full-action/`입니다. `T1-full-hidden/`도 액션만 읽어 사용할 수 있습니다.
3. 05의 실행 스위치 셀에서 cache 경로와 loss 가중치를 확인합니다.
4. 환경이 없으면 `RUN_INSTALL=True`로 설치 셀을 실행합니다. 기존 Qwen 모델이 없을 때만
   `RUN_DOWNLOAD=True`로 다운로드 셀을 실행합니다. 현재 작업에서 설치·다운로드는 실행하지 않았습니다.
5. `RUN_SMOKE=True`로 1-step 학습/봉인 저장/재로딩 검사를 실행합니다.
6. `RUN_OVERFIT=True`로 한 train episode의 50-step loss 감소 검사를 실행합니다.
7. 결과 확인 후 `RUN_MAIN_KD=True`로 본 학습 셀을 실행합니다. 최신 완료 checkpoint가 있으면
   같은 설정·데이터·코드·cache identity에서 query cursor/optimizer/RNG를 복원합니다.

모든 실행 스위치는 기본 `False`입니다. 설정 변경 시 `RUN_NAME`을 새로 지정하며 기존 결과를 덮어쓰지 않습니다.
본 학습은 새 Qwen 초기값에서 시작하고 smoke/overfit 가중치를 이어받지 않습니다.
`T1-pilot-action/`은 일부 episode뿐이므로 전체 학습에는 사용할 수 없습니다.
pilot cache를 점검하려면 별도 설정에서 실제 포함 범위에 맞는 `episode_limit`을 명시해야 합니다.

## 학습 목적과 검증

`loss = task_weight × masked_L1(Student, GT) + action_kd_weight × masked_L1(Student, T1)`

`configs/experiments/qwen_action_kd.json`의 기본 가중치는 각각 1.0이며 검증된 최적값이 아닙니다.
액션은 동일 정규화 공간의 `[B,8,7]`이며 `action_valid_mask`로 episode 말미 padding을 제외합니다.
Teacher target은 detach하며 실제 Teacher 모델이나 그 GPU 환경을 로드하지 않습니다.
액션과 원본 episode/timestep, 지시문, 두 카메라 참조, proprio, GT, mask를 대조합니다.
완료 cache/part의 hash와 데이터 revision·split·정규화·Teacher identity도 검사합니다.
한 번에 한 episode의 액션 tensor만 메모리에 보관합니다. hidden이 있는 JSON은 검사 시 읽지만 특징 증류에는 사용하지 않습니다.

로그에는 합산 `loss`, `task_l1`, `action_kd_l1`을 기록합니다. `validation_l1`은
Teacher cache를 사용하지 않는 held-out GT 손실이며 rollout 성공률이 아닙니다.
representation KD와 Teacher token ID 증류는 이 경로에 포함되지 않습니다.

## 저장 위치와 실행 진입점

Drive 프로젝트 루트 `내 드라이브/colab/VLADE/` 기준:

- 설정: `run-configs/qwen-action-kd-epoch-seed42*.json`
- 결과: `runs/student_action_kd/qwen-action-kd-epoch-seed42*/`
- checkpoint: `step-XXXXXXX/`의 model, processor, trainer state, identity 및 완료/무결성 표시

노트북이 생성한 절대 경로 설정으로 별도 Student 환경에서 실행합니다.

```text
python scripts/train/run_student_kd.py --config <run-config.json> --preflight
python scripts/train/run_student_kd.py --config <run-config.json>
python scripts/train/run_student_kd.py --config <run-config.json> --resume <completed-checkpoint>
```
