# Plan 생성 오류 수정 · 2026-11-24

## 확인한 원인과 변경

- 시작 루트는 `C:\Users\kimdo\Desktop\vlade`, 브랜치는 `main`, HEAD는 `0279cbd`였습니다.
  시작 `git status --short --ignore-submodules=all`은 깨끗했습니다. 기본 status의 submodule
  Git 메타데이터 접근은 Permission denied여서 submodule을 제외한 상태를 사용했습니다.
  외부 소스·gitlink·브랜치는 변경하지 않았습니다.
- 참조 대화와 로컬 `dev/eastha`/`origin/dev/eastha`의 `a8fa773`, 이전 `13f1904` 및 현재
  정본을 확인했습니다. `git ls-remote origin refs/heads/dev/eastha refs/heads/main`은
  `getaddrinfo() thread failed to start`로 실패하여 원격 최신 여부는 미확인입니다.
- 사용자 로그의 `ouch`·`imas`는 기존 평문 형식 검사에 통과합니다. 실제 로컬 AVA T0
  tokenizer에서 각각 `[2123,305]`, `[527,294]`로 복원됐습니다. Plan 태그는 일반 토큰이며
  special token이 아닙니다. T1의 실제 tokenizer 바이트·생성 ID·logits는 확보하지 못했습니다.
- 기존 FFT는 action hidden state의 L1만 최적화하고 LM CE를 사용하지 않습니다. 관련 기존
  회귀에서도 LM head gradient가 `None`임을 확인했습니다. 첫 EOS 강제 억제는 학습되지 않은
  언어 출력을 정상적인 텍스트처럼 보이게 할 수 있습니다. 실제 T1의 `ouch`·`imas`가 이
  logit 순서로 발생했다고 단정하지 않습니다.
- `src/vlade/distillation/teacher_plan.py`: 학습/추론 공통 PurePrompt, 감독 계약 검사,
  첫 EOS 허용, 생성-only ID 디코딩·EOS-only/예산 truncation 거절을 구현했습니다.
- `src/vlade/training/fixed_cache.py`, `teacher_inference.py`, `qwen_vla_task.py`,
  `student_task.py`, `src/vlade/evaluation/plan_rollout.py`: 액션 전용 체크포인트의 Plan 사용을
  차단합니다. Action-only 추론은 유지합니다. Validation probe는 Plan-only 응답임을
  `action_shape=null`, `action_status=not_requested_plan_only`로 명시합니다.
  Teacher 선택/추론은 현재 FFT의 canonical identity 두 필드도 대조하도록 맞췄습니다.
  기존 3필드 계약은 보존하며 새 감독 checkpoint의 5필드 계약과 실제 데이터가 일치해야 합니다.
- `src/vlade/training/teacher_plan_supervision.py`, `fft.py`, `prepared_fft.py`,
  `prepared_fft_suite_balanced.py`, `prepared_sequences.py`, `temporal.py`: 기존 FFT에 선택적
  Train Plan CE를 연결했습니다. 별도 LM forward에서 본문+EOS만 지도하며 기존 action
  forward/label masking과 val_l1·best 기준은 유지합니다. K1/K4·Suite 어댑터에서 주석을
  전달하고, 설정한 CE가 batch에서 빠지면 실패합니다. 기존 학습 루프·optimizer·Hub를 교체하지 않습니다.
- `src/vlade/storage/assets.py::resolve_input_record`: 기존 공통 HF 검증 후 pin을 노출합니다.
  주석은 URI·검증된 cache 모두 manifest와 본문이 실제 HF pin에 포함돼야 수락합니다.
  주석의 데이터 identity·Train 범위·coverage·중복·원문 지시를 확인하며 fallback은 없습니다.
- 새 checkpoint는 주석 manifest/본문 SHA256·HF 저장소/revision·prompt·CE weight·실제 step을
  봉인합니다. 감독 목표·주석 원문이 달라지면 resume을 거절합니다. 액션 전용 T1은 새 run의
  초기 가중치로 사용해야 합니다. 기본 `configs/experiments/libero_plus_fft.json`은 비활성입니다.
- 정식 설명은 `docs/environment/teacher_plan_generation.md`에 저장하고
  `docs/environment/action_level_distillation.md`, `README.md`, `STRUCTURE.md`에 연결했습니다.

## 실제 실행과 결과

환경은 PyTorch `2.12.0+cpu`, Transformers `5.9.0`, CUDA 사용 불가입니다. Python은 `-B`와
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=<프로젝트>/src`, `VLADE_PROJECT_ROOT=<프로젝트>`를 사용했습니다.
모든 pytest는 `-p no:cacheprovider` 및 이번 작업의 서로 다른 scratch basetemp를 사용했습니다.

```powershell
python -B -m pytest -q -s -p no:cacheprovider --basetemp=.harness/v1/2026-11-24-plan-generation-fix/scratch/pytest-plan-fix .harness/v1/2026-11-24-plan-generation-fix/checks/test_plan_generation_fix.py
```

첫 실행은 **20 passed, 3 failed**였습니다. 합성 tokenizer의 구두점 디코딩 기대값, CPU 합성
action head dtype, 합성 관측의 proprio 누락을 수정했습니다. 운영 출력을 보정하지 않았습니다.

이후 `scratch/pytest-plan-fix-r2`에서 이번 검사와 기존 태그·실패 원인·probe·KD·평가·FFT
검사를 실행해 **79 passed, 2 failed**였습니다. 두 실패는 감독 metadata 없이 Plan 생성을
허용하던 과거 fixture입니다. 과거 보존 파일은 변경하지 않았습니다. 이번 검사에 감독 계약을
갖춘 실제 JSON pipe·관측 일치 거절·실제 생성 경로 검사를 추가했습니다.

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-24-plan-generation-fix/scratch/pytest-regression .harness/v1/2026-11-24-plan-generation-fix/checks/test_plan_generation_fix.py .harness/v1/2026-11-24-t1-plan-tags/checks/test_generated_plan.py .harness/v1/2026-11-24-t1-plan-tags/checks/test_plan_failure_causes.py .harness/v1/2026-11-24-t1-plan-tags/checks/test_plan_probe.py .harness/v1/2026-11-22-teacher-plan-kd/checks/test_teacher_plan_kd.py .harness/v1/2026-11-23-plan-evaluation/checks/test_plan_evaluation.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_fft.py .harness/v1/2026-11-16-online-action-kd/checks/test_online_action_kd.py .harness/v1/2026-11-11-sequence-training/checks/test_temporal_training.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hf_assets.py -k 'not test_actual_json_pipe_returns_paired_signal_and_checks_timestep and not test_teacher_plan_uses_current_views_proprio_and_text_generation'
```

**127 passed, 2 deselected, 20.97초**였습니다. 제외한 과거 두 검사는 위의 감독 계약 없는
fixture입니다. 현재 회귀 검사로 대체했으며 운영 경계를 완화해 통과시키지 않았습니다.

주석 본문 SHA256·불변 revision의 재개 identity를 보강한 뒤 이번 파일만
`scratch/pytest-identity`에서 **27 passed, 15.50초**였습니다.

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-24-plan-generation-fix/scratch/pytest-checkpoint .harness/v1/2026-11-24-plan-generation-fix/checks/test_plan_generation_fix.py .harness/v1/2026-11-23-plan-evaluation/checks/test_plan_evaluation.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_fft.py
```

이 시점의 소스 상태에서 **52 passed, 16.87초**였습니다. 이전 127개와 중복된 검사입니다.
감독 CE의 LM/vision/proprio gradient·label masking, 기존 action-only gradient,
Validation 액션 L1 유지, URI/로컬 HF pin 수락·변조/미등록/미고정 파일 거절, train/val t=0/8
생성 경로, supervisor 없는 시작 거절, 새 checkpoint의 실제 저장·seal·Llama 가중치 재로딩,
동일 설정 재개 수락·변경 주석 재개 거절을 포함합니다. SWIG deprecation 경고 2개가 있었습니다.

현재 FFT의 canonical 식별자 2개와 Teacher 입력 계약의 연결을 보완하고
`scratch/pytest-canonical`에서 같은 세 검사 파일과
`.harness/v1/2026-11-16-online-action-kd/checks/test_online_action_kd.py`를 실행했습니다.
**63 passed, 19.25초**였습니다. 새 5필드 supervised identity 수락과 canonical 값 변경 거절,
기존 3필드 액션 Teacher의 수락·변조 거절을 함께 확인했습니다. 이전 실행과 중복된 검사입니다.

축소 무작위 Llama를 합성 Plan 하나로 CPU에서 40 optimizer step 학습한 CE는
**2.0423707962 → 0.0009270396**입니다. logits를 강제로 대체하지 않고 학습된 모델의 실제
`SequentialTeacher.plan`을 사용해 합성 train/validation t=0/8에서 `Move cup .`를 얻었습니다.
합성 WordLevel tokenizer의 구두점 간격이며 실제 T1·사용자 Episode의 의미 품질·일반화가 아닙니다.
별도 제어 logits 검사는 EOS-only와 `min_new_tokens=1`의 강제 텍스트 선택을 비교합니다.

변경·추가 Python **14개** 메모리 compile, 설정/manifest JSON **3개** 파싱,
`git diff --ignore-submodules=all --check`를 통과했습니다. 문서 링크의 작업 README는 이 기록
생성 전 미존재로 잡혔으며 기록 작성 후 다시 확인했습니다. scratch 제외 규칙은
`git check-ignore`로 확인했으며, 실행 임시 파일·checkpoint는 완료 전에 정리했습니다.

## 미실행·남은 조건

- 로컬 HF 캐시는 S0만 있고 T1은 없습니다. 실제 T1의 train/validation t=0/8 출력과 logits,
  checkpoint/tokenizer 일치, 고정 AVA Transformers fork의 실제 GPU 동작은 미검증입니다.
- 실제 검토된 Plan 주석 자산은 없고 GPU 학습·실제 T1 Plan 복구·S2 재학습·LIBERO+ rollout을
  수행하지 않았습니다. 합성 성공을 실측 Plan 정상화로 보고하지 않습니다. 올바른 다음 단계는
  검증된 HF 주석 준비와 새 supervised FFT 실행이며 별도 승인 없이 시작하지 않았습니다.
- 기존 T1·S1·S2 및 HF pin을 수정하거나 재게시하지 않았습니다. 최초 수정 단계에서는
  commit·push·Drive 배포·HF 업로드·대용량 다운로드·패키지 설치도 수행하지 않았습니다.

## 후속 Git 게시 요청

사용자의 별도 푸시 지시를 받은 뒤 원격 조회를 다시 실행했습니다. 원격 `main`은
`0279cbde66506885ef657b33b23ccb16908db35e`, `dev/eastha`는
`a8fa773e3088aaa8a670076cb29a2c9bf0f5371d`로 로컬 기준과 일치했습니다.
이번 변경 20개 파일과 하네스 보존 용량·scratch 부재를 확인했고 diff 공백 검사도 통과했습니다.
후속 작업은 이 변경의 Git commit·push이며 Drive/HF 배포는 포함하지 않습니다.
