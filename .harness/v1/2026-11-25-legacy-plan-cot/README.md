# Plan CoT의 legacy 이관

2026-11-25 KST. 사용자는 자연어 Plan을 전달하는 CoT 관련 자료를 legacy 폴더 한곳에
모으도록 허용했습니다. 기준 HEAD는 `3218235804cfcecc489a29cb08c5357502c9db38`입니다.
시작 시 별도 Plan 변경·진단 셀과 이전 S2 기록이 수정/미추적 상태였으며, Plan 정본의 현재
내용을 이동했고 과거 `.harness/v0`, 기존 v1 기록과 S2 작업 기록은 수정하지 않았습니다.

## 실제 변경

- `src/vlade/legacy/plan_cot/`에 전용 Python 모듈 9개, CLI/생성기 6개, 설정 5개,
  노트북 6개, 문서 3개를 이동했습니다. 이동 목록은 `checks/migrate_plan_cot.py`의
  명시적 SOURCES 목록이며 실제 프로젝트 루트/대상 경계·기존 목적지 부재·Git 제외 규칙을
  확인한 뒤 필요한 파일만 이동했습니다. 저장소 전체 복사나 데이터/가중치 이동은 없습니다.
- Python namespace는 `vlade.legacy.plan_cot.*`입니다. 기존 9개 모듈 위치는 제거하고
  공유 Teacher/Student·FFT·BC/KD·평가 코드의 import와 상대 import를 새 경로로 바꿨습니다.
  공유 학습·모델 코드와 선택적인 Plan 연결은 유지합니다.
- 이동한 research/CLI의 프로젝트 루트·기본 설정·subprocess 모듈 경로, 노트북 생성기·
  notebook source의 설정/문서 경로와 상대 링크를 맞췄습니다. 별도 호환 alias는 만들지 않았습니다.
- 현재 `docs/environment/action_level_distillation.md`는 S1 Action KD와 현재 S2 표현 증류
  중심으로 작성했고, 이전 전체 CoT 안내는 legacy 문서로 보존했습니다. README/STRUCTURE를
  같은 경계로 정리했습니다. L4의 예전 Plan 선택은 `s2_legacy_plan`으로 명시했습니다.
- 현재 S2 설정·기존 A100 세션 notebook은 원래 위치와 조건을 유지합니다.
  HF 입력 revision·seal·private Hub·재개 검사를 완화하지 않았습니다.

## 검증

기존 Anaconda CPU Python/Torch/Transformers 환경을 사용했고 설치·다운로드는 하지 않았습니다.
`-B`, PYTHONDONTWRITEBYTECODE, PYTEST_DISABLE_PLUGIN_AUTOLOAD, HF/Transformers offline을
사용했습니다. HF/Torch/TEMP/TMP와 pytest 임시는 이 작업 scratch로 지정했습니다.

```powershell
python -B -X utf8 -m pytest -q -x -p no:cacheprovider --basetemp=.harness/v1/2026-11-25-legacy-plan-cot/scratch/pytest-relocated .harness/v1/2026-11-25-legacy-plan-cot/checks/test_legacy_layout.py .harness/v1/2026-11-25-legacy-plan-cot/checks/test_legacy_annotation_contracts.py .harness/v1/2026-11-25-s2-representation-kd/checks/test_representation_kd.py .harness/v1/2026-11-22-s0-local-run/checks/test_local_inputs.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hf_assets.py
```

**102 passed, 2 SWIG deprecation warnings, 22.19초**였습니다. 이관된 29개 경로, 9개 모듈의
새 import·repository root, legacy CLI의 plan 무부작용, notebook 생성/compile/스키마와 문서
링크를 확인했습니다. 현재 S1/S2 설정 일치·S2 native 축소 모델의 gradient·checkpoint/
optimizer 저장·재개, 기존 HF/cache 수락·미등록/변조/revision 변경 거절도 확인했습니다.

기존 Teacher Plan 검사는 원문 기록을 수정하지 않고 이번 checks에서 import/파일 경로만
매핑해 실행했습니다. 첫 두 시도에서 검사 어댑터의 옛 notebook 경로 누락과 Windows의
write_text CRLF/이동본 LF 차이를 발견했습니다. 경로를 보완하고 legacy 생성기의 줄바꿈을
명시했습니다. 최종 보존본은 원래 notebook의 CRLF를 유지해 이동 diff를 줄였습니다.
실제 Plan/학습 조건은 바꾸지 않았습니다.

공유 notebook 생성기로 현재 Action KD/L4 클라이언트를 생성했고, legacy Teacher Plan KD
생성 진입점도 실행했습니다. 재생성 중 생긴 범위 밖 unfrozen notebook의 주석 차이는
해당 한 줄만 원래대로 보존했습니다. `git diff --check`를 통과했고 v0 추적 변경은 없습니다.
최종 notebook 줄바꿈 보존 후 관련 layout·주석/학습 계약 검사를 고유
`scratch/pytest-notebook-bytes` 경로에서 다시 실행해 **46 passed, 5.18초**였습니다.

## 실행 범위

실제 GPU/Colab 학습·rollout·HF 게시·Drive 배포는 미실행입니다. 실행 중인 Colab의 고정
checkout·패키지·캐시·데이터·가중치는 변경하지 않았습니다. 과거 실행 재현에는 원래 Git commit이
필요하며 새 구조는 이 이동이 포함된 commit에서 사용합니다.
이번 작업 scratch는 완료 시 정리하고 작은 검사 소스와 이 기록만 보존합니다.
