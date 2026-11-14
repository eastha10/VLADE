# 통합 Colab 실행본

- 정본: experiments/colab_vlade.ipynb. 실행용 개별 노트북 10개를 대체·삭제하고 configs/research의 18개 작업을 PROFILE로 선택하도록 구성했습니다.
- 모델·데이터 로더·학습·Hub 구현과 입력 설정은 변경하지 않았습니다. 소스·HF 인증·작업 선택·plan/setup/run 및 각 작업의 기존 파일럿·재개 조건을 연결합니다.
- 기본 run group을 실험 ID와 프로필별로 분리합니다. Suite pilot4/main과 논문 pilot/main은 같은 실험 ID를 유지합니다. 설정 변경 후 plan을 다시 확인하지 않은 setup/run은 거부합니다.
- 생성 정본: src/vlade/utils/colab_notebooks.py와 scripts/prepare/generate_colab_notebooks.py. 기존 hf_notebooks.py, kd_notebook.py, paper_notebook.py의 호환 함수도 통합본만 생성하도록 변경했습니다.
- README.md·STRUCTURE.md·관련 docs/environment 및 예산 결정 문서의 노트북 경로/선택 안내를 갱신했습니다. notebooks/references의 사용자 전달본 2개는 HEAD와 바이트 단위 일치를 확인했습니다.
- 삭제 전 기존 10개 노트북에 저장된 출력과 사용자 수정이 없음을 확인했습니다. 각 절대 경로가 프로젝트 experiments 바로 아래인 것을 검증한 후 명시한 파일만 삭제했습니다.
- 실행: python -B -m scripts.prepare.generate_colab_notebooks. 통합 노트북과 기존 Git-source manifest만 생성했습니다. manifest에는 내용 변경이 없습니다.
- 검증: python -B .harness/v1/2026-11-11-unified-colab/checks/check_unified_colab.py — 7 tests, OK (0.214초). 실제 CLI로 18개 profile의 side-effect-free plan, 출력/run group 분리, 기본 실행 차단, 변경된 선택/중첩 override 거부, Suite 단계 run group 유지, 재개 group 전달, 생성 멱등성과 호환 생성기의 구 노트북 비재생성을 검사했습니다. setup/run 전달은 mock으로 검사했습니다.
- 추가 검사: 변경 Python compile, 노트북 코드 셀 compile/JSON·빈 출력 검사, 로컬 Markdown 링크 76개, git diff --check --ignore-submodules=all 통과. nbformat 패키지는 없어 해당 라이브러리의 schema validator는 사용하지 않았습니다.
- 미실행: 실제 Colab 설치, GPU 훈련, HF 다운로드/게시·checkpoint 재개, Drive 배포. commit/push도 수행하지 않았습니다. 새 소스를 Colab에 배포하려면 검토 후 commit/push가 필요합니다.
- 임시 fixture는 검사 종료 시 제거됐으며 빈 scratch도 제거했습니다. 기존 사용자 미추적 기록과 참조 노트북은 보존했습니다.

## 4,000-step 실행 안내 추가

- 통합 노트북의 ava-4000-steps 설명 셀과 docs/environment/research_runs.md에 Suite pilot4 → main 4,000 step 및 일반 FFT 4,000-step 설정 예시를 추가했습니다. 생성 소스에도 반영하여 재생성 시 유지합니다.
- 예시의 warmup=400, save_steps=250, eval_steps=100과 기존 8시간 세션 상한·재개 조건을 명시했습니다. 기본 설정 파일이나 실제 학습 목표는 변경하지 않았습니다.
- 설명의 Python 예시 3개를 실제 CLI plan으로 실행하여 Suite pilot4=4, Suite main=4000, 일반 FFT=4000과 execution=not_started를 확인했습니다. 노트북 코드 셀 compile과 문서 본문 일치도 확인했습니다. 실제 setup/run은 미실행입니다.

## Git・Drive 게시 요청

- Drive VLADE 폴더에 통합 노트북 1개를 생성하고 코드·문서 19개를 업데이트했습니다. ID·부모·원격 크기를 읽어 20개 모두 로컬 크기 일치를 확인했습니다. 통합 노트북은 원격 원본을 다시 읽어 바이트 일치를 별도로 확인합니다. 기존 Drive 노트북의 출력·사용자 변형과 데이터·가중치·실험 결과는 보존했습니다. reports/drive-deployment.json에 파일 ID를 기록했습니다.
- Git은 미완료입니다. 현재 main/HEAD afb670a2c9bb3befe03a01c6ac2eaf3c99aca769에서 로컬 git add가 .git/index.lock Permission denied로 실패했습니다. 자동 검토가 저장소 신뢰·공개 업로드 범위 미확인으로 최초 GitHub tree 요청을 거부했습니다. origin 일치와 저장소 관리·push 권한을 확인한 뒤 작업 기록을 제외한 정식 변경 30개 경로만 재시도했으나 GitHub 연결이 Resource not accessible by integration(403)을 반환했습니다. 기존 Git 인증도 사용 가능한 자격증명이 없어 실패했습니다. 원격 tree/commit/ref를 생성·변경하지 않았으며 로컬 변경과 사용자 기록을 보존했습니다.
