# 에이전트 저장 규약 재작성과 v1 전환

- 날짜: 2026-10-29 (한국 표준시).
- 요청: 에이전트 부산물의 잘못된 저장과 작업 기록 비대화를 방지하고 v1을 시작합니다.
- 정본 변경: `AGENTS.md`, `README.md`, `STRUCTURE.md`, `docs/environment/teacher_records_and_student.md`, `.gitignore`.
- 규약: 목적에 따른 저장 경계, v1 단일 활성 경로, v0 보관, 명령 실행 전 출력·캐시 경로 확인, 전체 저장소 재귀 복사 금지, scratch 정리, 보존 파일당 1 MiB·작업당 5 MiB 상한.
- `v0/`의 파일은 이동·복제·삭제·수정하지 않았습니다. 기존 사용량의 실제 감축은 수행하지 않았습니다.
- 이번 작업은 문서·저장 규약 변경이며 학습·GPU·환경 설치를 실행하지 않습니다. 검증 결과는 아래와 같습니다.

## 검증 결과

- Colab manifest 164개 파일의 SHA256 일치 확인.
- 생성 노트북 3개의 manifest 참조 hash와 코드 셀 문법 확인.
- AGENTS·README·STRUCTURE에 활성 경로로 v0를 쓰는 예시가 없음을 확인.
- v1은 이 README 한 개만 보존하며 파일당·작업당 용량 상한 충족, scratch 없음.
- `.gitignore`의 v1 scratch 제외와 Git diff 공백 검사 확인.
- 애플리케이션 소스 변경 없음. 학습·전체 회귀 테스트는 실행하지 않음.

## Git 변경 파일 과다 표시 조치

- 푸시 전 현재 HEAD의 실제 커밋 수는 11개입니다.
- Git에서 제외되던 artifacts의 테스트 출력·합성 checkpoint·임시 torchvision 패키지를 v0로 옮긴 뒤 목적지의 제외 규칙이 누락됐습니다. 파일 이동 시 Git 제외 유지 확인을 AGENTS.md에 추가했습니다.
- 이전 정리 폴더의 testing·student만 .gitignore에서 제외합니다. 파일을 삭제하거나 과거 기록을 수정하지 않습니다.
- 전체 추가 대신 검토한 규약·문서·manifest·노트북과 작은 안내 기록만 명시적으로 커밋합니다.

- 추가 규약: 실제 파일 저장 시에만 필요한 폴더 생성, 빈 폴더·placeholder 금지, 완료 시 이번 작업의 빈 폴더 정리, 실제 수행한 변경·명령·관측 결과만 기록.
- Git 조사: 변경으로 표시되던 미추적 항목 51,606개 중 테스트 파일 51,401개·Student 임시 파일 202개였습니다. Git에서 제외되던 경로를 옮긴 뒤 목적지 제외 규칙을 누락한 것이 원인입니다. 로컬 보관본은 51,958개 파일·106,135,324바이트이며 삭제하지 않았습니다.

- 사용자의 삭제 승인 후 재생성 가능한 테스트 파일·임시 의존성·합성 checkpoint를 제거했습니다. 정리 직후 기존 요약과 result.json 두 개의 합계는 3153바이트였으며, 정리 전 대비 106132171바이트를 줄였습니다.
- Git 푸시 시 Drive colab/VLADE 정본 동기화를 함께 수행하도록 규약을 추가했습니다. 브라우저에서 대상 폴더 ID와 src·scripts·configs·experiments 등의 실제 프로젝트 구조를 확인했습니다. Colab 기본 ROOT와 관련 안내도 이 경로로 맞췄습니다.

- Drive ROOT 기본값 변경 후 `python -B -m pytest tests/unit/test_colab_source_files.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-10-29-harness-policy/scratch/pytest-drive-sync --tb=short`: 12 passed (0.56s). 생성 노트북 3개의 문법·ROOT·manifest 참조와 정본 SHA256도 확인했으며 테스트 scratch는 삭제했습니다.
