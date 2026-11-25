# S2 중간 표현 + Action Distillation

2026-11-25 KST. 시작 HEAD는 `caebd80c405766529b567e54c6b23a23a62de13a`입니다.
시작 시 정본 11개와 기존 T1 Plan 작업 기록·추론 셀이 수정/미추적 상태였으며 보존했습니다.
참조 대화와 그 첨부 `download.svg`를 읽어 고정 T1의 내부 표현·8×7 액션을 함께 전달하는
구조를 확인했습니다. 사용자 제공 T0/T1 Plan 실패·BF16 LM head 동일 관측은 새로 측정한 결과가 아닙니다.

## 실제 변경

- `src/vlade/distillation/representation.py`, `representation_config.py`: 액션 슬롯 56개 정렬,
  config 기반 Qwen→Llama2 Linear, 유효 슬롯만 LayerNorm+FP32 MSE, Teacher detach,
  Student RNG를 진행시키지 않는 projection 초기화, projection 저장/복구.
- `src/vlade/training/teacher_inference.py`: 기존 동일 T1 forward의 액션·temporal hidden을
  JSON 응답으로 함께 제공하고 observation·layer·layout·shape·finite를 확인합니다.
  기존 Plan 관련 사용자 변경을 보존했습니다.
- `src/vlade/training/qwen_vla_task.py`: S1 손실에 표현 손실을 더하고 projection을 optimizer,
  clipping, 메모리/저장 공간 계산과 checkpoint 봉인·재개에 포함합니다. 정책 model은 그대로
  저장하며 validation/best는 기존 GT L1입니다. 기존 Plan 관련 사용자 변경을 보존했습니다.
- `src/vlade/research/distillation.py`: opt-in 설정 검증, S2 출력/Hub 라우팅, 메모리 내 설정 계획 조회.
- `configs/experiments/qwen_vla_representation_kd.json`: S1 기본 설정에 representation만 추가.
- `src/vlade/utils/representation_kd_notebook.py`,
  `scripts/prepare/generate_representation_kd_notebook.py`,
  `experiments/colab_s2_representation_kd_existing_session.ipynb`: 기존 workspace의 AVA/Student
  환경과 HF cache를 재사용하는 두 셀입니다. 새 불변 Git 소스는 별도 checkout/자식 프로세스에
  적재합니다. 실제 S1 run-contract에서 입력 pin·예산·조건을 상속하며 설치/재시작을 하지 않습니다.
- `docs/environment/s2_representation_distillation.md` 및 README/STRUCTURE/Action KD 안내에
  새 S2와 보존한 자연어 Plan 경로를 구분하고 실행·제한을 명시했습니다.

## 실행·관측

기존 `C:/Users/kimdo/anaconda3/python.exe`의 CPU Torch 2.12.0+cpu/Transformers 5.9.0을
사용했습니다. 설치·다운로드는 하지 않았습니다. `-B`, PYTHONDONTWRITEBYTECODE,
PYTEST_DISABLE_PLUGIN_AUTOLOAD와 HF/Transformers offline을 사용하고 HF/Torch/TEMP/TMP를
이 작업 scratch에 지정했습니다. 모든 pytest는 `-p no:cacheprovider`와 고유 basetemp를 사용했습니다.

최종 기능·회귀 검사 명령:

```powershell
python -B -X utf8 -m pytest -q -x -p no:cacheprovider --basetemp=.harness/v1/2026-11-25-s2-representation-kd/scratch/pytest-contracts .harness/v1/2026-11-25-s2-representation-kd/checks/test_representation_kd.py .harness/v1/2026-11-23-distillation-runner/checks/test_distillation_runner.py .harness/v1/2026-11-25-t1-generated-plans/checks/test_generated_t1_plans.py .harness/v1/2026-11-22-s0-local-run/checks/test_local_inputs.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hf_assets.py
```

**70 passed, 2 SWIG deprecation warnings, 22.76초**였습니다. 축소 무작위 native Qwen의
Student/Projection gradient, 무효 슬롯 제외, 동결 Llama 표현·쿼리 replay, 실제 worker JSON 분기,
공통 loop의 저장/재개 후 다음 update bit-exact 일치, 실제 trainer adapter의 pilot→resume 및
optimizer moment·봉인된 projection·독립 정책 reload를 확인했습니다. adapter smoke의 GPU와
외부 HF 입력 경계는 CPU용 대체물을 사용했으며 실제 A100/HF 측정으로 취급하지 않습니다.
별도 기존 입력 검사에서 고정 HF/cache 수락·미등록/변조/revision 변경 거절을 확인했습니다.

초기 두 검사 실패는 합성 fixture의 조건 문제였습니다. 첫 검사는 AVA dropout이 다른 두
forward의 숫자를 비교해 실패했고 같은 forward의 보고 항으로 검증하도록 수정했습니다.
다음 검사는 legacy 4-layer Qwen fixture의 full attention이 최종층에만 있어 첫 query의 -2
action slots가 0이고 projection gradient도 0이었습니다. 실제 24-layer Qwen처럼 더 앞에도
full attention이 있는 축소 interleaved fixture로 수정해 유한한 비영 gradient를 확인했습니다.
운영 Student 구조/층 선택은 이 과정에서 바꾸지 않았습니다.

최종 run-group 경계 검사 추가 후 관련 Colab child·셀 검사 2개를
`scratch/pytest-session-guard`에서 재실행해 **2 passed, 3.14초**였습니다.
`python -B -X utf8 checks/verify_artifacts.py`는 정본 소스 7개 compile, config JSON,
nbformat 스키마·무출력 셀, 생성 notebook byte-exact 일치, 문서 상대 링크 4개를 확인했습니다.
생성 진입점도 실제 실행했습니다. CLI `--action plan`에서 `runs/s2-cpu-plan/S2` 라우팅을
관측했으며 지정한 plan workspace는 생성되지 않았습니다.

`git -c diff.ignoreSubmodules=all diff --check`를 통과했습니다. 기본 Git 상태/공백 조회는
고정 submodule의 로컬 접근 제한에 걸려 submodule 무시 옵션으로 상위 저장소를 조회했습니다.
submodule을 변경하지 않았으며 `.harness/v0`의 추적 변경은 없습니다.
scratch의 Git 제외를 `git check-ignore`로 확인했고 완료 시 이 작업 scratch를 정리했습니다.
보존 기록은 이 README와 checks 두 파일뿐이며 개별 1 MiB/합계 5 MiB보다 작습니다.

## 미실행·남은 조건

실제 T1 7B/Qwen 0.8B 가중치의 A100 forward·메모리·학습·rollout 성능 및 HF 게시는 미실행입니다.
GPU 실험·대용량 다운로드·환경 재설치·Git commit/push/브랜치 변경·Drive 배포는 하지 않았습니다.
새 소스를 포함하는 원격 Git commit이 없으므로 Colab 학습 실행에는 향후 소스 게시가 필요합니다.
이는 기존 source identity 검사를 우회하지 않기 위한 실행 조건이며 셀에 명시했습니다.
S1 기본값과의 설정 일치는 검증했으며 실제 이전 S1의 조건 비교에는 해당 run-contract가 필요합니다.
HF run·step·불변 revision·pending의 실제 원격 상태를 새로 생성하거나 확인한 작업은 아닙니다.

## Git 게시 후속 요청 · 커밋 전 확인

같은 날 사용자가 S2 구현의 Git push와 현재 A100에 붙일 셀을 요청했습니다.
원격 `origin/main`과 로컬 HEAD는 모두 `caebd80c405766529b567e54c6b23a23a62de13a`였습니다.
`checks/prepare_publication.py`로 S2만 포함하는 index patch를 준비했습니다. README/STRUCTURE/
Action KD 안내·공유 trainer/worker의 기존 Plan 변경은 작업 트리에 보존하고 index에서 제외했습니다.
`teacher_plan.py`, fixed_cache, 기존 Plan 노트북/진단 셀 등 별도 Plan 작업은 게시 대상이 아닙니다.

`checks/staged_source.py`는 변경된 vlade 모듈을 Git index의 실제 바이트로 로드합니다.
아래 명령으로 게시 대상 코드와 기존 입력 경계의 CPU 회귀를 확인했습니다.

```powershell
python -B -X utf8 -m pytest -q -x -p no:cacheprovider -p staged_source --basetemp=.harness/v1/2026-11-25-s2-representation-kd/scratch/pytest-publication .harness/v1/2026-11-25-s2-representation-kd/checks/test_representation_kd.py .harness/v1/2026-11-23-distillation-runner/checks/test_distillation_runner.py .harness/v1/2026-11-22-s0-local-run/checks/test_local_inputs.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hf_assets.py -k 'not test_native_preflight_verifies_teacher_and_student_without_cuda'
```

**63 passed, 1 deselected, 2 SWIG warnings, 19.31초**였습니다. 제외한 검사는 CoT 설정에서
미감독 T1을 허용하는 별도 Plan 변경에 의존하므로 이번 S2-only 게시 대상과 다릅니다.
S2의 무 GPU preflight는 새 `test_s2_preflight_verifies_s0_t1_dimensions_without_cuda`로
S0/T1 호출·1024→4096 차원·Plan 비활성·출력 미생성을 확인했습니다. 같은 index loader와
고유 `scratch/pytest-s2-preflight` 경로로 실행해 **1 passed, 3.03초**였습니다.

`git apply --cached --check`와 `git diff --cached --check`도 통과했습니다.
이 절은 커밋 전 관측이며 실제 commit/push 결과는 이후 관측 기록과 구분합니다.

## Git 게시 관측

후속 요청에 따라 `49a3ec6b07a18b761d32670c2c05cd165e08cdc7`
(`feat: add plan-free S2 representation and action distillation`)을 현재 main에 커밋했습니다.
`git push origin HEAD:refs/heads/main`이 성공했고 `git ls-remote origin refs/heads/main`의
SHA가 같은 값임을 확인했습니다. 별도 Plan 변경은 작업 트리에 보존했으며 Drive 배포와
실제 A100 실행·HF checkpoint 게시는 수행하지 않았습니다.
사용자에게 제공하는 셀은 이 불변 commit의 notebook 실행 셀을 읽어 기존 WORKSPACE로 실행합니다.
이 관측 추가는 로컬 하네스 기록이며 실행 소스를 바꾸지 않습니다.

## Colab 자식 로그 누락 후속 수정

사용자가 제공한 전체 출력은 시스템 Python 3.13에서 실행한 CLIENT의 CalledProcessError뿐이며
실제 자식 stderr는 없었습니다. 이 출력만으로 Python 버전·환경 누락·HF 입력 실패 등
구체적인 preflight 실패 원인은 확정할 수 없습니다.

`representation_kd_notebook.py`의 실행을 stdout PIPE/stderr 병합·비버퍼링·명시적 print로
바꿨습니다. 종료 실패는 긴 CLIENT 명령 대신 짧은 RuntimeError로 표시합니다.
사용자 중단은 이번 실행에서 새로 만든 process group에만 SIGINT를 전달하고 기다립니다.
모델·학습·HF 입력 검증·설치 환경 선택이나 기존 실행 소스는 변경하지 않았습니다.
생성 노트북과 실행 안내를 맞췄습니다.

`test_colab_stream.py`는 실제 작은 자식 프로세스의 stdout/stderr/traceback 출력과 짧은 오류,
성공 출력 및 자체 process group에 한정한 중단을 확인합니다. 기존 셀 compile·S1 상속 검사와
함께 `scratch/pytest-colab-stream`에서 실행해 **5 passed, 3.25초**였습니다.
artifact 검사는 소스 7개 compile·notebook 스키마/생성 일치·JSON·링크를 통과했습니다.
실제 Colab 실패 원인 확인과 A100 실행은 미완료이며, 사용자에게 기존 CLIENT를
preflight로 재실행해 실제 로그를 표시하는 셀을 제공합니다.
