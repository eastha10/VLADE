# S1 Action KD와 현재 S2 표현 증류

현재 QwenVLA는 같은 S0와 LIBERO+ 전처리 입력에서 다음 두 실험을 구분합니다.

| 실험 | 설정 | 실행 셀 |
|---|---|---|
| S1 · GT + Action KD | [S1 설정](../../configs/experiments/qwen_vla_action_kd.json) | [Action KD](../../experiments/colab_action_distillation.ipynb) |
| S2 · GT + Action + Representation KD | [S2 설정](../../configs/experiments/qwen_vla_representation_kd.json) | [기존 A100 세션](../../experiments/colab_s2_representation_kd_existing_session.ipynb) |

정규화된 `[B,8,7]` 액션과 유효 mask를 공유합니다. S1은 `GT L1 + λa × Teacher L1`,
현재 S2는 여기에 `λr × Representation MSE`를 더합니다. 현재 S2의 토큰·차원 정렬,
실제 S1 조건 상속, 기존 런타임 재사용과 검증 범위는 [S2 안내](s2_representation_distillation.md)를 따릅니다.

## 실행과 저장

`scripts/train/run_distillation.py`에서 `plan → setup → preflight → pilot → train` 순서로 실행합니다.
기존 설치 환경이 있으면 setup을 반복하지 않습니다. 현재 A100 세션은 위 전용 셀을 사용합니다.
`pilot`은 원래 누적·목표 schedule을 유지하는 1 optimizer step이며, `train`은 같은 run을 재개합니다.
설정·코드·입력이 달라지면 새 run group을 사용합니다.

```bash
python -B scripts/train/run_distillation.py --config configs/experiments/qwen_vla_action_kd.json --workspace /content/vlade_distillation --run-group action-kd-seed42-r1 --action plan
```

S0/T1과 데이터는 고정 HF revision·무결성·checkpoint identity 검사를 통과해야 합니다.
Validation/best 기준은 held-out GT L1입니다. private model 저장소
`VLADE-konkuk/VLADE-checkpoints`의 `runs/<group>/S1` 또는 `runs/<group>/S2`에
봉인된 checkpoint와 pointer를 기존 구현으로 게시합니다. 목표 step을 완료했을 때만 final입니다.
전송 실패 시 원본·`hub_pending.json`을 보존하고 `retry`로 복구합니다.
인증과 불변 revision을 사용한 재개는 [Hub 안내](hugging_face_checkpoints.md)를 따릅니다.

## 과거 자연어 Plan 경로

Plan 생성·주석·Teacher Plan CE·Plan 입력 CoT KD·플랜 효과 평가는
[legacy/plan_cot](../../src/vlade/legacy/plan_cot/README.md)에 모았습니다.
공유 학습기의 선택적 legacy 연결은 보존하지만 현재 S1/S2 설정은 Plan을 사용하지 않습니다.
현재 S2 결과를 과거 자연어 CoT 결과와 같은 실험으로 해석하지 않습니다.
