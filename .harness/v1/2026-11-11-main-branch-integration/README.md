# 사용자 브랜치 푸시·main 통합

2026-11-11 KST. 시작 main `ddc86c38632fcfafe48fc4317bda6334a47dffbd`. 사용자 지시로 현재 연구 코드 정리를 dev/eastha에 commit/push하고, 모든 원격 브랜치를 main에 통합하며 fish13031e의 데이터 샘플링 개선을 반영합니다.

원격 브랜치는 main, dev/eastha, dev/fish13031e, codex/ava-transformers-patch-a입니다. eastha와 codex는 시작 main의 조상이며 fish는 2 commit 앞섭니다. 과거 미추적 하네스 기록은 별도 보존하며 이번 소스 commit에 포함하지 않습니다.

## 실제 반영·검증

- 현재 연구 코드 정리를 `dev/eastha`의 `42999217fac1addae091b6bea1783b80d77dcf13`로 commit하고 push했습니다. main에 해당 branch를 병합한 commit은 `a6a5cfa`입니다.
- `dev/fish13031e`의 변경을 main에 통합했습니다. `src/vlade/data/episode_data.py`가 suite별 quota·선택 query 집합을 유지하며 같은 optimizer update 안의 반복 궤적 그룹을 줄입니다. FFT loader는 accumulation×batch 크기를 전달하고 checkpoint에는 새 순서와 update 크기를 기록합니다. Validation·Teacher cache·Student 순서는 유지합니다.
- 충돌은 하네스로 이동한 test_colab_pipeline.py 한 파일에서 발생했습니다. 현재 하네스 검사를 보존하고 협업자의 새 검사 2 case를 이번 작업 checks/test_suite_query_merge.py에 분리했습니다. tests/unit에 검증 파일을 되살리지 않았습니다.
- 전체 하네스 304 passed, 0 failed/0 skipped, SWIG warning 2개 (227.24초). 관련 Python compile과 정책 검사를 통과했습니다. HF 입력 경계·고정 revision·현재 Colab 경로·v0 동결·submodule pin을 보존했습니다.
- 기존 미추적 기록 25개는 stash 원본 Git blob과 비교해 동일함을 확인했고 이번 소스 commit에서 제외했습니다. 실제 데이터·가중치 재처리나 GPU 훈련, Drive 배포는 수행하지 않았습니다. 데이터 변경은 코드의 샘플링 순서 개선이며 기존 prepared payload를 다시 만들지 않습니다.
