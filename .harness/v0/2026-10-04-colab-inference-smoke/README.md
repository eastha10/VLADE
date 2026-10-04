# Colab 단일 추론 실행 작업 기록

2026-10-04 업로드 완료 후 사용자 요청에 따라 OFT와 AVA의 A100 단일 추론을
실행하고 기존 Notion 작업 페이지에 결과를 기록했습니다. 노트북의 Drive 경로,
TensorFlow Metadata 호환성 및 CPU 전처리를 수정했습니다. 원본 submodule과
가중치는 수정하지 않았습니다. 기준 VLADE commit은
`82480532420340d57e39529a7416991c78ce2744`입니다.

공식 결과와 재현 정보는 `experiments/runs/inference_smoke/`의 JSON 2개 및
README에 있습니다. 이 폴더는 초기 import 실패와 실제 실행 출력, 종료 증빙을
보관합니다. 이전 `.harness/v0/2026-10-03-basic-inference-smoke/README.md`의
사용자 변경은 유지했습니다.

- `logs/oft-initial-import-error.txt`: TensorFlow Metadata/protobuf 초기 실패의 UI 출력.
- `logs/oft-inference-output.txt`: 통과한 OFT 실제 추론 출력.
- `logs/ava-install-output.txt`: AVA 환경 설치와 import 검사 출력.
- `logs/ava-inference-output.txt`: 통과한 AVA 실제 추론 출력.
- `outputs/runtime-ended.jpg`: Colab 세션 관리의 활성 세션 없음 확인 화면.
- `outputs/tests-complete-session-ended.jpg`: 두 모델 통과 결과와 종료 상태 요약 화면.

검증: 두 모델에서 shape=(8,7), NaN/Inf 없음; 결과 JSON 문법 및 notebook 코드
문법 확인. Colab 런타임은 종료했으며 노션에 완료 결과·범위·재현 정보·세션
종료를 기록하고 재조회로 확인했습니다. FFT 및 벤치마크 검증은 포함하지 않습니다.

Colab UI의 실행 노트북 다운로드 이벤트는 대기 시간 내 반환되지 않았습니다.
로컬의 기존 노트북에 실제 사용한 코드 수정과 완료 결과를 반영했고,
실제 실행 출력과 결과 JSON은 별도로 보존했습니다.
