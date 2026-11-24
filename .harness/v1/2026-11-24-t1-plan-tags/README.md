# T1 출력의 플랜 태그 처리

- `src/vlade/distillation/teacher_plan.py`에 응답 전체를 감싼 `<plan>…</plan>` 쌍을 벗기는 함수를 추가했습니다. `src/vlade/training/fixed_cache.py`에서 T1 디코딩 직후 이 함수를 호출합니다. `docs/environment/action_level_distillation.md`에 허용 형식을 기록했습니다.
- `python -B -m pytest -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-24-t1-plan-tags/scratch/pytest-plan-tags`로 이번 검사와 11월 22일 증류·11월 23일 플랜 평가 검사 파일을 실행했습니다. 23개가 통과했습니다. 평문과 태그 형식은 같은 Student 프롬프트로 전달됐으며 빈 태그, 미완성 태그, 중첩 태그는 거절됐습니다.
- 변경 Python 파일과 검사 파일 3개의 메모리 내 컴파일 및 `git diff --ignore-submodules=all --check`를 통과했습니다. 로컬 PyTorch가 CPU 빌드이고 T1 가중치가 로컬 캐시에 없어 실제 T1 생성과 LIBERO+ rollout은 미실행입니다. 보고된 `None`을 만든 실제 디코딩 문자열은 확인하지 못했습니다.

## 플랜 오류 진단 보완

- `normalize_generated_plan`은 비문자열을 `InvalidTeacherPlan`으로 거절하고, 형식 검증 실패에 디코딩 원문 앞 256자를 포함합니다. 태그 추출 후에도 빈 본문·중첩·미완성 태그의 거절을 유지합니다.
- `src/vlade/evaluation/plan_observations.py`는 플랜 거절 시 기존 `None` 응답을 유지하면서 task ID·timestep·오류 이유를 stderr JSON으로 출력합니다. `src/vlade/training/teacher_inference.py`는 `InvalidTeacherPlan` 상세를 JSON 응답으로 전달해 Student의 기존 플랜 오류 로그에 남깁니다. 일반 예외 메시지의 전달 범위는 확장하지 않았습니다.
- 다음 명령으로 40개 검사가 통과했습니다. SWIG 관련 DeprecationWarning 2개가 보고됐습니다.

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-24-t1-plan-tags/scratch/pytest-plan-diagnostics .harness/v1/2026-11-24-t1-plan-tags/checks/test_generated_plan.py .harness/v1/2026-11-22-teacher-plan-kd/checks/test_teacher_plan_kd.py .harness/v1/2026-11-23-plan-evaluation/checks/test_plan_evaluation.py .harness/v1/2026-11-16-online-action-kd/checks/test_online_action_kd.py
```

- 추가 검사에서 합성 T1 생성 거절 → worker JSON 응답 → Student 학습 오류 로그의 전달과, 평가 stderr의 관측 식별자·원문 발췌를 확인했습니다. 변경 Python·검사 파일 5개의 메모리 컴파일과 diff 공백 검사를 통과했습니다. 이번 검사의 임시 파일은 정리했습니다.
- 실제 T1 디코딩 문자열은 여전히 확보하지 못했으므로 기존 `None`의 원인을 태그로 단정할 수 없습니다. CPU 빌드 환경에서 실제 T1 생성·GPU 증류·LIBERO+ rollout은 미실행입니다. commit·push·Drive 배포·HF 업로드는 실행하지 않았습니다.

## `None` 원인 경로 재현

- 로컬 `artifacts/`, `experiments/runs/`, `runs/`, `data/`, `checkpoints/`에서 관련 로그 이름을 확인했지만 실제 플랜 생성 로그를 찾지 못했습니다. 플랜 관련 정식 노트북 3개의 저장된 출력 수는 모두 0입니다. Drive에서 `colab_t1_plan_alignment`와 지정 배포 폴더의 `plan`을 검색했으나 결과가 없었습니다. 이는 해당 검색 범위의 결과이며 외부 Colab 런타임에 로그가 없다는 뜻은 아닙니다.
- 직접 경로는 `SequentialTeacher.plan`의 디코딩 → `normalize_plan`(기존)/`normalize_generated_plan`(현재) 검증 → `live_response`의 `except InvalidTeacherPlan` → `plan=None`입니다. Student Qwen 프롬프트를 만들기 전에 발생합니다. 검증 실패로 `None`이 되는 동작은 실재하지만, 당시 T1의 실제 문자열은 확보하지 못했습니다.
- `checks/test_plan_failure_causes.py`에서 평문, 완성 태그, 빈 문자열, 숫자, 미완성 태그, 중첩 태그의 6조건을 실제 `live_response`로 재현했습니다. 완성 `<plan>…</plan>`은 기존 경로에서 `None`, 현재 경로에서 정상 본문이 됐습니다. 빈 출력·숫자·미완성/중첩 태그는 양쪽에서 `None`입니다. 이는 합성 문자열의 경계 검사이며 실제 T1 토큰 생성 실험이 아닙니다.
- 실제 `vlade.training.fft.compute_loss`를 축소 합성 모델에 적용해 역전파했습니다. 액션 head와 embedding에는 gradient가 생겼고, 출력에 differentiable LM loss가 있어도 LM head의 gradient는 `None`이었습니다. 현재 FFT는 hidden state의 연속 액션 L1만 사용하고 `output.loss`를 목표에 더하지 않습니다. 따라서 해당 학습이 자연어 플랜 생성을 지도한다는 근거는 없습니다. 사전학습 모델에 언어 능력이 전혀 없다는 주장이나 당시 빈 응답의 실측 근거로 확대하지 않습니다.
- 플랜은 `language_model.generate`로 별도 생성하고 액션은 `predict_action`으로 생성합니다. 액션 생성 경로에 생성 플랜을 입력하지 않으며, 플랜 생성에는 기존 AVA temporal context와 vision attention generator가 연결되지 않습니다. 현재 플랜을 액션의 내적 추론이라고 해석할 근거는 없습니다.
- `python -B -m pytest -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-24-t1-plan-tags/scratch/pytest-plan-causes .harness/v1/2026-11-24-t1-plan-tags/checks/test_plan_failure_causes.py`: **7 passed in 3.06s**. 실제 T1 가중치 생성·GPU 학습은 미실행이며 새로 정식 소스는 변경하지 않았습니다.

## 실제 생성 경로와 축소 Student 학습

- 로컬 환경은 RAM 31.6 GiB(조회 시 여유 15.9 GiB), PyTorch `2.12.0+cpu`, Transformers `5.9.0`이며 CUDA를 사용할 수 없습니다. `checkpoints/hf_assets/`에는 S0만 있고 T1은 없습니다. 기본 AVA T0를 T1 대신 사용하지 않았으며 대용량 다운로드·환경 설치는 실행하지 않았습니다.
- `checks/test_plan_generation_fit.py`에서 실제 `LlamaForCausalLM.generate`와 실제 fast tokenizer를 사용했습니다. 축소 무작위 Llama의 LM head logits를 제어해 EOS/UNK/BOS가 우선인 세 조건을 만들었습니다. 수정 전 세 검사 모두 빈 디코딩 문자열로 실패했습니다(`pytest-generation-before`, **3 failed in 16.19s**). 실제 T1 분포를 측정한 실험은 아닙니다.
- `src/vlade/distillation/teacher_plan.py`의 `plan_generation_kwargs`에 최소 1개 새 토큰, EOS 외 특수 토큰 억제, 기존 액션·추가 vocabulary 억제, 강제 BOS/EOS 해제를 적용했습니다. `fixed_cache.py`의 실제 생성 경로에 연결했고 거절 시 token ID·Episode·timestep을 추가했습니다. 세 조건은 수정 후 정상 텍스트를 생성합니다. 숫자만 생성하는 경우는 계속 거절하며 임의의 대체 문장은 넣지 않습니다.
- 같은 체크 파일에서 실제 축소 `Qwen3_5Model`의 vision/merger/language와 VLADE AVA/proprio/action head를 사용해 생성 플랜을 Student 입력으로 전달했습니다. GT+Teacher masked L1, backward, AdamW를 **12 step** 실행해 loss가 **0.7721566558 → 0.1672563404**로 감소했습니다. vision/language/AVA/proprio/action 모듈의 유한한 nonzero gradient와 플랜 로그를 확인했습니다. 합성 이미지·액션·processor와 제어된 Teacher logits를 사용한 CPU 통합 검사이며 실제 S0/T1/LIBERO+ 성능이 아닙니다.
- `src/vlade/training/teacher_inference.py`의 `probe_teacher_plans`와 `src/vlade/research/distillation.py`의 `--action probe`를 연결했습니다. 고정 HF 검증을 거친 입력으로 train·validation 첫 Episode의 각 최대 2쿼리를 확인합니다. Student/optimizer/Hub 게시 없이 플랜과 액션 계약을 확인하며 실패를 JSONL에 보존합니다. Action-only 설정은 workspace 생성 전에 거절합니다.
- `src/vlade/utils/action_kd_notebook.py`, `experiments/colab_teacher_plan_kd.ipynb`, `docs/environment/action_level_distillation.md`에 `preflight → probe → pilot → train`을 반영했습니다. `preflight`와 실제 생성 검사를 구분하고 플랜 의미 품질의 미검증을 명시했습니다. 입력 검증·고정 revision·기존 Hub 및 재개 계약은 변경하지 않았습니다.
- 생성·probe·기존 플랜·평가·실행기 검사 파일 6개를 `python -B -m pytest -q -s -p no:cacheprovider --basetemp=.harness/v1/2026-11-24-t1-plan-tags/scratch/pytest-generation-and-probe`로 실행해 **51 passed in 18.64s**였습니다. 마지막 probe 설정 거절 조건 추가 후 `checks/test_plan_probe.py`와 기존 `test_distillation_runner.py`를 `--basetemp=.harness/v1/2026-11-24-t1-plan-tags/scratch/pytest-probe-validation`로 실행해 **13 passed in 3.98s**였습니다. 13개에는 앞 실행과 중복된 검사가 있습니다.
- Python 9파일 메모리 컴파일, 정식 노트북 JSON/nbformat/셀 컴파일, 문서 링크와 diff 공백 검사를 통과했습니다. CPU 검사의 CUDA autocast 비활성 경고와 SWIG deprecation 경고는 관측했으며 실제 CUDA 검증으로 간주하지 않습니다. 이번 실행의 임시 폴더는 정리했습니다.
- 실제 T1 가중치·고정 AVA Transformers 런타임의 생성과 실데이터 S2 fit은 **미실행**입니다. 로컬 T1·CUDA·timm이 없고 실행 소스는 미커밋 상태이므로 기존 고정 실험 경계를 우회하지 않았습니다. 새 `probe`는 실제 Colab 실행용으로 연결했으며 여기서 실측 통과했다고 주장하지 않습니다. commit·push·Drive·HF 업로드는 실행하지 않았습니다.
