# Qwen 오프라인 액션 증류 준비

2026-11-03 한국 시간. 사용자가 저장된 액션 청크로 나중에 증류를 실행하도록 코드 작성을 요청했습니다.
실제 모델 다운로드·GPU 학습·대규모 데이터 생성은 실행하지 않았습니다. 기존 미커밋 변경을 보존했습니다.

## 변경

- `src/vlade/training/cached_actions.py`: 02의 완료 cache/part 해시, Teacher provenance, train 분할,
  정규화, 원본 query/이미지 참조/GT/proprio/mask 대조. 한 episode의 액션만 보관.
- `src/vlade/distillation/action.py`, `training/student_task.py`: 기존 sampler/초기화/optimizer/저장·재개에
  GT L1 + Teacher L1의 가중합 연결. padding 제외, Teacher detach, 두 손실 로그, validation은 GT만 사용.
- `scripts/train/run_student_kd.py`, `configs/experiments/qwen_action_kd.json`: 별도 진입점/기본 loss 설정.
- `src/vlade/utils/kd_notebook.py`, `experiments/colab_05_qwen_action_distillation.ipynb`: 별도 05 생성,
  설치/다운로드/smoke/overfit/본 학습 스위치 모두 False, 03과 동일 조건의 학습·재로딩·pilot gate.
- generator, source manifest, 01~04 bootstrap 갱신. run_control의 KD pilot context에 cache/code 해시 추가.
- README/STRUCTURE 및 `docs/environment/qwen_action_distillation_colab.md`, 기존 Student 안내 갱신.
- `tests/unit/test_action_distillation.py`, source generation의 5-notebook 회귀 검사.

## 실행한 검증

기존 Anaconda Python, torch 2.12.0+cpu/transformers 5.9.0을 사용했습니다. 패키지는 설치하지 않았습니다.

```text
python -B -m pytest tests/unit/test_action_distillation.py tests/unit/test_training_loop.py tests/unit/test_colab_source_files.py tests/unit/test_colab_pipeline.py tests/unit/test_teacher_records.py tests/unit/test_student_contract.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-03-action-kd/scratch/pytest-01
82 passed in 31.16s
python -B -m pytest tests/unit/test_action_distillation.py tests/unit/test_colab_source_files.py tests/unit/test_final_run_review.py tests/unit/test_colab_pipeline.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-03-action-kd/scratch/pytest-02
56 passed in 31.02s
```

두 번째 검사는 GPU 없이 preflight 통과/변경 cache로 재개 거부 검사를 추가하고 cache pilot context를 보완한 후 실행했습니다.
합성 Teacher cache → 원본 causal sampler → KD target 연결, padding mask, 실제 CPU autograd,
optimizer/RNG 저장·재개 일치, 변조/다른 데이터/부분 cache 거부 및 기존 회귀가 통과했습니다.
5개 노트북 코드 구문, 로컬 manifest 전체 SHA256, `git diff --check --ignore-submodules=all`도 통과했습니다.
기본 git status는 외부 submodule metadata 접근 제한으로 실패하여 `--ignore-submodules=all`로 확인했습니다.
외부 submodule과 기존 데이터·가중치는 변경하지 않았습니다.

## Drive 반영

승인된 `내 드라이브/colab/VLADE`의 정식 파일 21개를 추가/갱신하고 모두 다시 읽어 SHA256 일치를 확인했습니다.
대상 폴더/동명 파일 유무를 먼저 확인했고 기존 파일 ID를 유지했습니다. 기존 01~04는 bootstrap 해시만
갱신했으며 사용자 셀/출력/metadata가 보존됐음을 원본 JSON과 대조했습니다.
05: https://drive.google.com/file/d/16HUYnZZfc21TmVZWWMY0RGXaEH9Bs-Mu/view
검증 hash는 `drive-verification.json`에 기록했습니다.
Drive manifest는 변경하지 않은 Drive 소스의 기존 hash/revision을 보존하므로 로컬 manifest와 bytes가 다릅니다.
각 환경의 01~05 bootstrap은 해당 환경의 manifest hash를 가리킵니다.
검증용 합성 데이터와 Drive 비교 payload는 작업 scratch 정리 대상입니다. Git commit/push는 수행하지 않았습니다.

## 제한

실제 Qwen GPU forward/backward/VRAM·학습 성능/rollout 성공률은 미실행입니다.
기본 loss 가중치 1.0/1.0은 시작값이며 튜닝 결과가 아닙니다. representation KD는 포함하지 않습니다.
04 LoRA checkpoint의 T1/cache 생성 연결은 별도 작업입니다.

## 후속 Git 반영

같은 날 사용자가 Git 푸시와 Drive 반영을 함께 완료하도록 명시했습니다.
05가 의존하는 기존 02/04 변경까지 포함하여 정식 파일 44개를 커밋했습니다.
`.harness/`의 이전/현재 작업 기록과 데이터·가중치는 커밋에 포함하지 않았습니다.

- 커밋: `a080bda4d2e0fc7b38a193b5d447bfc42bd5a176`
- 원격: `https://github.com/eastha10/VLADE.git`, `dev/eastha`
- `git push -u origin dev/eastha` 성공 후 `git ls-remote --heads origin dev/eastha`에서 같은 SHA 확인.
- `python -B -m pytest tests/unit -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-03-action-kd/scratch/pytest-push`:
  **161 passed in 45.89s**. GPU 학습은 실행하지 않았습니다.
- Git 자동 CRLF 정규화로 8개 소스의 index SHA가 기존 manifest와 달라지는 문제를 발견했습니다.
  `.gitattributes`로 원본 bytes를 보존하고 해당 파일을 다시 정규화하여 모든 parent-repo manifest 항목의
  staged bytes SHA 일치를 확인했습니다. 원래 CRLF를 가진 파일은 줄바꿈 자체를 보존한 커밋입니다.
- `git diff --cached --check` 통과. 푸시 후 tracked 변경/미반영 staged 변경은 없으며 기존 작업 기록만 untracked입니다.
- Drive의 05 노트북과 source manifest를 다시 읽어 앞선 SHA 검증본과 동일함을 확인했습니다.
  새 `.gitattributes`도 같은 Drive 프로젝트 루트에 업로드 후 본문 일치를 검증했습니다.
  Drive 파일 ID: `120L0NkeUs29iUiM1d4CluIpwj6BuARmF`.
