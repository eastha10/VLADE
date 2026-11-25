# CoT + Action KD 실행 노트북

- 정본: `experiments/colab_cot_action_kd_ready.ipynb`.
- GitHub `main`의 `4b24ece9dbdc5cb690a1bab189c121af85d95967` CoT KD 노트북·공식 job을 기준으로, 사용자 지정 `VLADE_Action_KD_Ready_20261125.ipynb`의 설정·재개 메모·로그·checkpoint/Hub metadata 검사·결과 ZIP 기능을 합쳤습니다.
- 추가: Train/Validation T1 probe, probe 이후 pilot/train 실행, S2 출력·복구·평가 pin, 별도 `T1_plan` 입력 키의 고정 HF checkpoint 선택. 기존 registry의 T1을 덮어쓰지 않습니다.
- 변경: RUN_TRAIN 기본값을 False로 설정하고, preflight 재실행 시 성공 상태를 먼저 초기화하여 실패 뒤 이전 상태가 남지 않게 했습니다. 생성 Python 검사 파일 대신 자식 Python의 `-c`로 checkpoint 검사를 수행합니다.
- 기존 로컬 수정·브랜치·submodule은 변경하지 않았습니다. commit/push/Drive 배포/HF 업로드는 실행하지 않았습니다.

## 검증

`python -B -X utf8 .harness/v1/2026-11-25-cot-kd-notebook/checks/check_notebook.py`

- 최종 8개 검사 통과: 코드 셀 및 내장 audit 문법, 출력 초기화, 고정 commit의 실제 load_settings 연결, CoT token 예산, 별도 Teacher pin 키, S1 재개 거부, 정상/빈/미완료 probe, probe 이전 훈련 거부, 실패 preflight 상태 초기화, 결과 묶음의 플랜 로그 포함.
- 처음 검사 코드 실행에서 생성 AST의 lineno 누락으로 실패했습니다. 검사 코드에 fix_missing_locations를 적용한 뒤 전체 8개를 다시 실행해 통과했습니다.
- 실행 환경에 nbformat/pytest가 없어 표준 라이브러리 unittest와 compile을 사용했습니다. nbformat 스키마 검증은 미실행입니다.
- 실제 GPU 생성·파일럿·학습·HF 원격 checkpoint 검사는 미실행입니다. 새로운 학습과 대용량 입력 다운로드를 시작하지 않았습니다.
- 기존 Action-only T1은 Plan 지도학습 이력이 없어 CoT 실행에서 거절됩니다. 실제 실행에는 봉인된 Plan-supervised T1이 필요합니다. 이 노트북 생성으로 Teacher 학습이나 Plan 의미 품질을 입증하지 않습니다.
- 원본 파일 다운로드는 DNS 오류로 실패하여, Library의 전체 텍스트 읽기로 20개 원본 셀을 확인했습니다. 최종 노트북은 22개 셀입니다.
- 이번 작업의 재생성 가능한 helper·임시 파일은 scratch에서 정리했습니다.

## 실행 준비 후속 확인

- 사용자 요청: CoT+Action KD 노트북 준비. 기존 정본의 설정·소스 pin·probe·저장/재개 연결을 다시 확인했습니다.
- `README.md`, `STRUCTURE.md`, `docs/environment/action_level_distillation.md`에 준비 노트북 링크와 실제 실행 순서·Plan-supervised T1 입력 조건을 연결했습니다. 노트북의 고정 소스와 기존 다른 작업의 변경은 유지했습니다.
- `python -B -X utf8 .harness/v1/2026-11-25-cot-kd-notebook/checks/check_notebook.py`: 8개 통과.
- 기존 Anaconda Python의 `nbformat.read(..., as_version=4)` 및 `nbformat.validate(...)`: 22개 셀의 스키마 검증 통과. 앞선 nbformat 미실행 범위를 이번 확인으로 보완했습니다. 설치는 하지 않았습니다.
- GPU 실행·외부 입력 다운로드·HF 검사/게시·Git commit/push는 미실행입니다. Plan-supervised T1의 실제 원격 가용성은 확인하지 못했으며 실행 전 해당 checkpoint pin이 필요합니다.
- 후속 문서의 로컬 링크, 코드 셀 문법·무출력, `git diff --check --ignore-submodules=all`, 보존 용량 한도를 확인했습니다. 이번 검사에서 만든 임시 fixture는 정리됐습니다. 기존 `scratch/helpers/`의 전달 도우미 2개는 이번 실행의 생성물이 아니어서 보존했습니다. scratch 제외 규칙도 확인했습니다.
