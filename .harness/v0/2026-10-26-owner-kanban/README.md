# 김동하 담당 10월 25일 마감 작업

실행일: 2026-10-26 한국 시간. 저장소: Desktop/vlade,
기반 commit `600f304292700d2408e226e75cb895e1c1b735c8` 위의 미커밋 변경입니다.

## 요청과 범위

- VLADE 칸반에서 주 담당 `김동하`, 2026-10-25까지 마감인 미완료 카드만 대상으로 했습니다.
- Teacher action/chunk 저장 코드: https://app.notion.com/p/3e7b5dd06870807384e4f2c0c53b30d3
- Qwen Student 계약: https://app.notion.com/p/403b5dd06870816aa6dec171db863bd1
- 집필 준비: https://app.notion.com/p/403b5dd0687081068f65d2aae66074b0
- 사용자가 세 카드 모두 In progress를 요청하여 결과 기록 후에도 해당 상태를 유지했습니다.
- 모델 가중치 검토는 추후로 미루라는 요청을 반영했습니다. 실제 0.8B 가중치를
  다운로드하지 않았으며 Hub에서 불러오는 인터페이스만 준비했습니다.
- 주 담당이 다른 논문 적용 매핑·Teacher adapter·실험 budget 카드는 수정하지 않았습니다.

## 구현과 산출물

- `src/vlade/data/teacher_records.py`: bounded JSONL writer/reader, dtype/shape,
  checkpoint·정규화·tokenizer 식별, debug/KD 분리, temporal 순서·무결성 검증.
- `src/vlade/models/contracts.py`, `students/qwen.py`: native Qwen VLM과 별도 continuous
  action query head, 2 camera × 2 query history, 정규화 8×7 액션, mask loss.
- `configs/models/student/qwen3_5_0_8b.json`: 고정 revision과 검증 범위.
- `scripts/train/smoke_student.py`: 다운로드 없는 축소 native Qwen 검증 CLI.
- `docs/environment/teacher_records_and_student.md`: API·실행·환경·실제 검증 제한.
- `docs/paper/preparation.md`, `experiment-record.template.json`: 버전별 원문 5편과
  재현/변형 경계, outline·조건표·결과표·그림 양식. 성능 숫자는 미실행으로 유지.
- README·STRUCTURE에 새 구현의 위치·역할·현재 상태를 반영했습니다.

## 검증

환경: Windows, Anaconda Python 3.13.5, torch 2.12.0+cpu, transformers 5.9.0.
torchvision 0.27.0은 호환 의존성을 공식 PyPI metadata에서 확인하고
`artifacts/student/python-deps`에만 설치했습니다. Teacher fork·전역 환경은 변경하지 않았습니다.

- `python -m pytest tests/unit -q -p no:cacheprovider --basetemp=artifacts/testing/kanban-2026-10-26-run3 --tb=short`: **55 passed**.
- `smoke_student.py --processor-dir checkpoints/students/qwen3_5_0_8b`:
  **공식 tokenizer/processor + 축소 무작위 Qwen3.5**에서 입력 330 tokens, 4 images,
  출력 `[1,8,7]`; vision/language/proprio/action nonzero 유한 gradient,
  입력 변화에 대한 반응, 1 SGD update, save/load exact 일치 통과.
- 결과 JSON: `artifacts/student/contract-verified-2026-10-26/result.json`.
- Python AST 18개와 신규 JSON 구문, 문서 상대 링크·경로·Git diff 공백을 확인했습니다.
- 테스트의 초기 시스템 임시 경로/부모 디렉터리 생성 오류는 프로젝트 내부 임시 경로를
  준비해 해소했습니다. 최종 55개는 모두 실행·통과했습니다.
- Git 상태 확인 시 submodule 내부 권한 오류가 있어 `--ignore-submodules=all`로
  상위 저장소 변경을 확인했습니다. submodule은 수정·갱신하지 않았습니다.

## 로컬 자료와 남은 범위

`checkpoints/students/qwen3_5_0_8b`에는 고정 revision의 config/tokenizer/processor/라이선스
약 23MB와 파일별 SHA256 manifest만 있습니다. 사전학습 모델 가중치는 없습니다.
검증 중 생성한 tiny random checkpoint는 `artifacts/` 아래에만 두며 Git에 넣지 않습니다.

실제 pretrained 0.8B·real RLDS batch·GPU FFT/Student·고정 T1 cache·KD 학습·양자화·
LIBERO(+) 성공률·Jetson 실측은 미검증입니다. 지금의 CPU 결과를 성능 근거로 사용하지 않습니다.
기존 `.harness/v0/2026-10-03-basic-inference-smoke/README.md` 사용자 변경을 보존했으며
commit/push/브랜치 변경을 하지 않았습니다. Notion 카드 3개에 결과를 기록하고 재조회했습니다.
