# Git 소스 배포 · HF 입력 개편 (2026-11-04)

## 범위와 변경

- `dev/eastha`를 당시 main `3ed6710`까지 fast-forward한 뒤 작업했습니다. 기존 미추적 skeleton 및 다른 작업 기록은 변경·추가 대상에서 제외합니다.
- `src/vlade/utils/source.py`: Git ref를 불변 commit으로 확정하고 실행 사본을 분리합니다. 부모 commit에 고정된 AVA submodule을 준비하며 변경된 사본을 거부합니다. Windows 경로 길이 제한을 피하면서 전체 commit을 대조합니다.
- 노트북 생성기·Colab 01~05·Suite·추론 smoke: 소스 파일별 SHA256 목록과 Drive 입력 준비/내장 코드 사본을 제거했습니다. `colab_source.json`은 schema 2 Git 배포 설정으로 전환했습니다.
- `src/vlade/storage/assets.py`, FFT·Student·논문 AVA·전처리·평가 진입점: 외부 입력은 HF 고정 자료 키 또는 검증된 HF 캐시로 제한합니다. 현재 실행 출력·checkpoint 재개 경로는 로컬에 둘 수 있습니다. 프로세스 내 동일 입력의 중복 해시를 줄이고, 입력이 변경되면 재검증합니다.
- 다운로드·데이터 분할·정규화·Teacher 출처·checkpoint 무결성 검사는 유지합니다. Student 모델의 중복 해시 루프는 공통 HF 입력 검증으로 통합했습니다.
- `AGENTS.md`, README·STRUCTURE·환경 안내: Git 우선 반영 및 명시적 명령 후 Drive 배포 원칙과 새 실행 경로를 반영했습니다.
- Notion Project Tasks: `담당자` People 속성과 `기간` Date 속성으로 중복 속성을 통합했습니다. 최초 41개 항목을 확인했고 날짜가 달랐던 5건은 원래 값을 본문에 보존했습니다. 마감 수식·관련 보기를 갱신하고 담당 계정의 보존을 확인했습니다. 작업 중 외부에서 추가·삭제·상태 변경된 항목은 되돌리지 않았습니다.

## 실제 검증

- `python -B -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=.harness/v1/2026-11-04-hf-input-refactor/scratch/pytest-final-07`: **201 passed, 2 skipped, 67.88초**.
- 로컬 Git 소스 배포 검사: **10 passed**. 서로 다른 commit의 사본 분리·변경된 소스 거부·AVA submodule commit 보존과 변경 감지를 확인했습니다.
- 전체 Python 소스 및 생성된 Colab 코드 셀의 문법 검사를 통과했습니다. `git diff --check`를 통과했습니다.
- Windows 기본 실행 환경은 임시 Git clone이 제한되어 해당 검사는 승인된 실행 권한으로 로컬 fixture만 사용했습니다. 실제 GPU 학습·대용량 다운로드는 실행하지 않았습니다.

## 배포 경계와 남은 입력

- 현재 Colab 런타임과 Google Drive는 조회·수정·업로드하지 않았습니다. HF 자료 게시나 기존 자료 변경도 수행하지 않았습니다.
- `teacher_cache_t1`, `ava_calvin_t0` 등 미게시 입력은 실제 게시·검증·레지스트리 등록 후 사용해야 합니다. 다른 모델·데이터나 Drive 경로로 우회하지 않습니다.
- Git commit·원격 push·main 병합 상태는 Git 이력과 완료 응답에서 확인합니다. Drive는 사용자의 후속 명령 전까지 반영하지 않습니다.
- 검사 임시 파일과 로컬 Git fixture는 완료 전 본 작업의 scratch에서 정리합니다.
