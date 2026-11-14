# AVA K4 Notion 실행 안내 추가

- 대상: https://app.notion.com/p/414b5dd0687081d19a60ec338324a7ce
- 기존 본문을 보존하고 Colab 소스·인증 준비, 일반 FFT 4-step 파일럿, Bash/PowerShell 7.3 CLI, Suite pilot4/main, 출력·Hub 식별자와 일반 FFT 복구 예시를 추가했습니다. Notion insert_content 성공 응답을 확인했습니다.
- 근거: docs/environment/ava_sequence_training.md, research_runs.md, hugging_face_checkpoints.md와 src/vlade/research/{cli,client,configuration,execution,suite}.py. 확인 시 로컬 HEAD는 afb670a2c9bb3befe03a01c6ac2eaf3c99aca769입니다. 프로젝트 구현·설정은 변경하지 않았습니다.
- 실제 검사: PowerShell의 Standard native 인수 전달과 PYTHONDONTWRITEBYTECODE=1을 사용했습니다. `python -B scripts/research/run.py --action plan --workspace .harness/v1/2026-11-11-k4-notion-guide/scratch/runtime`에 일반 FFT(max_steps=4, warmup_steps=0, save_steps=4, eval_steps=4), Suite pilot4, Suite main(max_steps=1000, warmup_steps=100, save_steps=250) 설정을 각각 전달했습니다. 모두 exit 0, sequence_length=4, execution=not_started를 확인했습니다. plan은 workspace나 scratch 파일을 생성하지 않았습니다.
- 현재 CLI가 기존 settings 파일과 변경된 resume 설정을 비교하여 거부할 수 있는 제한, Suite 재개의 추가 경로·pilot 증거 조건을 문서에 명시했습니다.
- 미실행: setup, 패키지 설치, HF 다운로드·게시, GPU 학습, 실제 checkpoint 재개, Drive 배포. 명령 예시 검사는 학습 성공을 입증하지 않습니다.
- Git 상태는 --ignore-submodules로 확인했습니다. 기본 git status와 git diff --check는 외부 submodule 메타데이터 접근 거부를 관측하여 submodule을 제외한 검사로 범위를 제한했습니다. 기존 사용자 미추적 기록은 보존했습니다. commit/push는 수행하지 않았습니다.
- 후속 요청: `python -B -m scripts.research.run --config configs/research/ava_suite.json --workspace .harness/v1/2026-11-11-k4-notion-guide/scratch/runtime --action plan`을 실행해 exit 0, K4/pilot4/4-step, execution=not_started를 확인했습니다. Notion CLI 절을 python -m 직접 실행 3줄과 필요한 override만 쓰는 설명으로 간소화했습니다. 구현 변경이나 실제 setup/run은 없습니다.
