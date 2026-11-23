# QwenVLA 백본 학습 범위 비교

[`colab_qwen_vla_unfrozen.ipynb`](../../experiments/colab_qwen_vla_unfrozen.ipynb)는 schema 2 S0로 시작합니다.
기본값은 `full`, GT-only 100 optimizer step, 누적 8, LR 2e-5, 25스텝 저장입니다.
`OVERRIDES`에서 `workflow.variant`를 `both`로 바꾸면 같은 S0·데이터·seed·학습 범위로 GT-only와 Action KD를 순차 실행합니다.

| `trainable_scope` | 학습 대상 | `top_layer_count` |
|---|---|---|
| `full` | 백본과 정책 헤드 전체 | 0 |
| `policy` | AVA·proprio projector·액션 헤드 | 0 |
| `top_layers` | 정책 모듈과 마지막 N개 언어 디코더 층, final norm | N ≥ 1 |

범위 설정은 `OVERRIDES`에 지정합니다. 범위를 바꾸는 실험은 새 run group으로 시작합니다.
Colab에서 `HF_TOKEN`을 설정하고 `ACTION`을 `plan → setup → preflight → pilot → train` 순서로 바꿔 실행합니다.
파일럿 로그의 학습 파라미터 수, 메모리, 손실과 Hub 전송 상태를 확인하십시오.
full FP32 AdamW는 Colab A100을 기준으로 하며, GPU 메모리 적합성은 파일럿에서 확인합니다.

재개할 때 원래 실행의 `SOURCE_COMMIT`, `RUN_GROUP`, 설정을 사용합니다.
새 런타임에서 복구할 때 `RESUME_REVISIONS`에 variant별 불변 HF commit SHA를 지정합니다.
단계별 학습 로그와 시간 CSV는 `WORKSPACE/runs/RUN_GROUP/<variant>/`에 저장됩니다.
로그 경로와 CoT 실행은 [Distillation 실행 안내](action_level_distillation.md)를 따릅니다.
