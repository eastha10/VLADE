# Student 뼈대

2026-11-04 한국 시간. 사용자가 논문을 모두 검토한 뒤 직접 구조를 판단하여 구현하기로 했으며,
이번 범위는 뼈대만 만드는 것입니다.

- `src/vlade/models/students/skeleton.py`: `StudentSkeleton.forward(observations, proprio) -> PolicyOutput`와 TODO만 추가.
  파라미터가 없고 호출 시 `NotImplementedError`. 백본·융합·시간 이력·헤드·증류 목적은 미정이며 학습 진입점에 연결하지 않았습니다.
- README/STRUCTURE에 위 범위를 안내했습니다. 기존 Qwen·학습·증류 코드는 변경하지 않았습니다.
- 소스 manifest에 새 파일과 변경 문서의 SHA256을 반영하고 01~05 노트북의 bootstrap hash만 갱신했습니다.
  생성 시 줄바꿈 변경을 원래 형식으로 맞추면서 발생한 manifest/bootstrap hash 불일치는 재계산 후 검증했습니다.

검증 환경은 기존 Anaconda Python, torch 2.12.0+cpu, transformers 5.9.0입니다. 패키지를 설치하지 않았습니다.

- `python -B -m pytest tests/unit/test_student_contract.py tests/unit/test_action_distillation.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-04-student-model/scratch/pytest-contract-01`: 15 passed in 47.39s. 뼈대 추가 전 기존 코드 확인입니다.
- `python -B -m pytest tests/unit/test_colab_source_files.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-04-student-model/scratch/pytest-source-01`: 12 passed in 1.23s.
- 뼈대 compile/import, 파라미터 0개, 미구현 호출 거부 확인. 소스 목록 전체 SHA256과 5개 노트북 구문·bootstrap hash 검사 통과.
- `git diff --check --ignore-submodules=all` 통과. 외부 submodule metadata 접근 제한으로 상태 확인은 submodule을 제외했습니다.
- 다운로드·실제 가중치 로딩·GPU 학습은 미실행입니다. 기존 데이터와 가중치·사용자 변경은 보존했으며 commit/push는 하지 않았습니다.

Drive의 `내 드라이브/colab/VLADE`와 `src/vlade/models/students` 폴더 계층 및 기존 파일을 조회했습니다.
새 뼈대 업로드는 자동 승인 검토에서 거부되었습니다. 사유는 비공개 소스의 외부 전송 승인과
목적지의 소유권·접근 범위가 충분히 확인되지 않았다는 것입니다. 우회하지 않았으며 Drive 변경은 없습니다.
이후 사용자가 "일단 스켈레톤이니까 완성되기 전까지 올리지마"라고 지시했습니다.
이 작업의 로컬 정본 9개(뼈대, README, STRUCTURE, source manifest, 01~05 노트북)는
Student 완성 전까지 로컬에만 보관하고 Drive 업로드·동기화 및 Git push를 보류합니다.
일반 Drive 동기화 지침보다 이번 사용자의 명시적 보류 지시를 우선합니다. 승인 대기 요청은 철회했습니다.
