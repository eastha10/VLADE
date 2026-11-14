# Baseline registry

AVA-VLA와 OpenVLA-OFT의 원본 LIBERO 조건을 기록합니다. source 경로와 commit은 출처 정보이며,
실행할 때 `third_party/`를 읽지 않습니다. 프로젝트 모델·processor·관측/액션 변환·평가 진입점을 사용합니다.

| 모델 | 원본 LIBERO 환경 |
|---|---|
| AVA-VLA | `configs/runtime/ava_libero_eval.json` |
| OpenVLA-OFT | `configs/runtime/oft_libero_eval.json` |

현재 실행은 [연구 안내](../../docs/environment/research_runs.md)를 따릅니다.
원본 LIBERO는 `evaluate_libero_ava.json` / `evaluate_libero_oft.json`, 주 연구 LIBERO+는
`evaluate_ava.json` / `evaluate_oft.json`을 선택합니다. benchmark commit·task 수를 검사하여
원본 10-task suite와 LIBERO+ variant 결과를 구분합니다.

```bash
python scripts/evaluate/run_baseline.py ava_vla --task-suite libero_10 --dry-run
python scripts/evaluate/run_baseline.py openvla_oft --task-suite libero_10 --dry-run
```

호환 CLI의 실제 실행에는 위 독립 평가 환경을 먼저 준비하고 해당 Python을 사용합니다.
checkpoint는 고정 HF 키 또는 검증된 캐시로 받습니다. `--use-hub-checkpoint`도 실행 시
등록된 HF asset으로 검증합니다. 훈련·평가 설정은 `configs/research/`에 있으며,
평가 횟수는 `evaluation.rollouts_per_task`, horizon은 `evaluation.max_steps`로 제어합니다.
실제 성공률은 측정 결과와 구분하며 단일 추론·합성 검사로 baseline 재현 완료를 선언하지 않습니다.
