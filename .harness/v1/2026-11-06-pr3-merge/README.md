# PR #3 충돌 해결과 검증

2026-11-06 사용자가 현재 PR 검토와 병합을 지시했습니다.
대상은 `eastha10/VLADE#3`, 원래 head `7d367bd470e129d7806569b4e7cea13a8108dfc6`,
main `e9d30b7c4278670b6d580e991eafcd885c68ca51`입니다. 원격 CI와 기존 리뷰는 없었습니다.

- 원본 작업 공간의 main 및 미추적 사용자 파일을 보존하고 scratch의 분리 worktree에서 검토했습니다.
- 16개 충돌 파일은 줄바꿈 차이를 정리한 뒤 내용을 통합했습니다. Git source schema 2,
  검증된 HF 입력, AVA attention 패치, 예산·checkpoint Hub 연결과 schema 1/2 loader 호환성을 유지했습니다.
- PR의 canonical D/seed42 manifest·정규화, 전처리 재개·동기 백업, 단일 Train Episode 준비,
  FFT 유한값·메모리·시간 보고 및 별도 프로세스 재로딩 검사를 통합했습니다.
- 기존 고정 HF prepared 자료가 정규화 JSON의 원본 바이트 SHA를 유지하는 경우 통계값의 완전한
  일치와 HF 입력 무결성을 모두 검사합니다. prepared 원본 identity와 재개 검사는 유지합니다.
- 새 OFT·AVA 점검 노트북은 Git/HF 실행 방식으로 생성합니다. 새 AVA 노트북은 기존 검증된
  Transformers 패치 설치 셀을 공유합니다. Qwen 및 KD 입력은 기존 HF prepared 키를 유지합니다.
- 유지 관리되는 Suite-balanced 실행본은 보존했습니다. PR의 전달본 Suite-balanced·paired rollout
  노트북은 `notebooks/references/`에 원본 코드·설명 그대로 보존하고 안내·구조 문서를 갱신했습니다.
- 새 normalization CLI는 기존 TFRecord CRC reader와 protobuf schema를 재사용합니다.

검증 환경은 기존 Anaconda Python과 CPU Torch입니다. 부족한 google-crc32c 1.7.1은
이 작업의 scratch/deps에만 설치했습니다. 모델·실제 데이터 다운로드, GPU 학습,
Drive 배포, HF 자료 게시는 실행하지 않았습니다.

- 첫 집중 검사: 128 passed, 5 failed. 실패는 미초기화 submodule 1건과 실행 접근 제한에 따른
  로컬 Git fixture 4건이었습니다. 고정 submodule을 준비하고 허용된 실행에서 다시 확인했습니다.
- 전체 `tests`: 255 passed, 1 failed (199.61초). 실패는 생성 노트북 수 6→8의 기존 기대값이며 수정했습니다.
- 수정 및 추가 검사: 12 passed (6.04초). 생성 기대값·AVA 설치 셀 공유·기존 HF 정규화 직렬화
  호환성과 통계 변조 거부·canonical manifest·TFRecord CRC 거부를 확인했습니다.
- 최종 생성 검사: 3 passed (0.77초). 모든 실행 노트북의 Git/HF 계약·Python 문법과
  Qwen/KD의 이전 로컬 전처리 경로 회귀를 확인했습니다.
- 변경 Python 21개 AST·JSON 문법, 실행 노트북 코드 문법, `git diff --check`를 확인했습니다.

검증 명령은 `python -B -m pytest ... -q --tb=short -p no:cacheprovider`와
작업별 고유 `scratch/pytest-*` basetemp를 사용했습니다. 재생성 가능한 fixture·의존성·worktree는
병합 후 정리하며, 원격 병합 결과는 작업 루트의 `reports/merged-pr.json`으로 확인합니다.
