# Legacy · 자연어 Plan 기반 CoT

T1이 자연어 Plan을 생성하고 Student 입력에 전달하던 CoT 경로를 이 폴더에 모았습니다.
전용 Python 모듈·CLI·설정·Colab 노트북·문서를 함께 보존합니다. 현재 S2는
[중간 표현 + 액션 증류](../../../../docs/environment/s2_representation_distillation.md)이며,
이 폴더의 과거 S2/CoT 표기와 구분합니다.

- `distillation/`, `training/`, `data/`, `evaluation/`, `research/`, `utils/`: Plan 전용 구현.
- `scripts/`: 주석 준비·Teacher Plan 학습·노트북 생성·Plan λ 선택 진입점.
- `configs/experiments/`: 과거 Plan KD·Teacher Plan 학습·Plan 평가 설정.
- `experiments/`: Plan 학습·진단·비교 노트북.
- `docs/environment/`: 기존 Plan 생성·학습·CoT KD 안내.

공유 BC/KD·FFT·Teacher worker·Student processor는 원래 위치에 남습니다.
그 안의 선택적 Plan 기능은 `vlade.legacy.plan_cot` 모듈을 명시적으로 참조합니다.
S1과 현재 S2의 기본 설정은 Plan을 요청하지 않습니다. 현재 S2 노트북은 이곳으로 옮기지 않습니다.
실제 데이터·가중치·실험 결과와 `.harness/v0`, 기존 v1 작업 기록도 이동하지 않습니다.

옛 `vlade.distillation.teacher_plan` 등의 import와 옛 Plan 전용 파일 경로는 제거했습니다.
새 Python import는 `vlade.legacy.plan_cot.*`입니다. 예를 들어 다음 명령은 설정 계획만 표시합니다.

```bash
python -B src/vlade/legacy/plan_cot/scripts/train/run_teacher_plan.py --workspace /content/vlade_distillation --run-group legacy-teacher-plan --action plan
```

노트북은 이 이동이 포함된 Git commit을 지정해야 합니다. 실행 중인 Colab의 고정 소스나
checkpoint 계약을 자동 변경하지 않으며, 과거 실행 재현에는 그 실행의 원래 commit을 사용합니다.
HF 입력 검증·봉인·private Hub·재개 계약을 완화하지 않습니다.

Legacy로 보존했다는 것은 자연어 생성 품질이나 GPU 학습 성공을 보증한다는 뜻이 아닙니다.
현재 제공된 T0/T1 진단에서는 Plan 생성이 실패했으므로 이 경로를 현재 S2 기본값으로 사용하지 않습니다.
