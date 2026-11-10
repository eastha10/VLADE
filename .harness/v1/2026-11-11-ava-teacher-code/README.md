# 연구 실행 코드 독립화·모듈화 및 suite map 게시

작업일: 2026-11-11 KST. branch `main`, HEAD `ddc86c38632fcfafe48fc4317bda6334a47dffbd`를 유지했습니다. 기존 Qwen 패치 변경과 사용자 기록을 보존했습니다. commit/push·branch 변경·submodule 갱신·Drive 배포는 수행하지 않았습니다.

## 실제 변경

- 현재 AVA/OFT LIBERO 연속 L1 경로를 `src/vlade/models/teachers/`에 모듈화했습니다. `ava_vla.py`는 조합부이며 vision, 명시적 Llama 연결, HF 연결, projector, action head, AVA attention, multimodal forward, prediction, processor/config/output을 분리했습니다. 실제 Llama decoder는 고정 Transformers 패키지에서 제공합니다. 원본 출처·Apache-2.0/MIT 고지를 보존했습니다.
- FFT·논문 LoRA/TBPTT·Suite-balanced·cache의 모델/활성 데이터 transform·prompt/collator/action tokenizer/통계 저장을 프로젝트 코드로 연결했습니다. FFT의 원본 디렉터리 검사·외부 Prismatic import·submodule 초기화를 제거했습니다. OFT는 AVA decoder 인자를 전달하지 않습니다. native LoRA checkpoint의 평가 로더를 추가했습니다.
- `src/vlade/research/`, `src/vlade/runtime/`, `scripts/research/run.py`, `configs/research/` 및 `configs/runtime/`에 설정 기반 계획·설치·실행을 두었습니다. 기존 학습 루프·Hub 저장/재개를 사용하며 step·rollout·seed·학습률·저장/검증 주기·예산을 설정으로 전달합니다. Student reload/overfit gate와 Suite pilot4/main 분리 및 paper pilot gate를 유지합니다.
- 기존 Colab 및 paired rollout 참고본을 읽고 유지 관리 노트북 10개를 짧은 설정/CLI 호출로 바꾸었습니다. 원본 참고 노트북은 보존했습니다. 새 실행은 Git commit 고정이며 실행 스위치는 기본 False입니다. 원본 LIBERO와 LIBERO+ 환경·revision·task 수를 구분합니다. runtime benchmark namespace와 BDDL/초기 상태 자료는 고정 Git checkout으로 연결합니다.
- 검증 전용 Python 25개를 `checks/`로 옮기고 중복 smoke 래퍼를 제거했습니다. 이동 내역은 `reports/verification-relocation.json`입니다. 추가 검증도 checks에만 만들었습니다. AGENTS.md·.harness/README.md와 README/STRUCTURE 및 현재 실행 안내를 맞췄습니다. 실제 연구 실행 코드는 src/scripts/configs에 유지합니다.
- 기존 HF asset 12개의 내용을 보존하고 LIBERO+ assets.zip의 공식 revision·크기·SHA256 metadata와 아래 suite map을 lock에 추가했습니다. 약 6GB assets.zip 본문·대형 가중치·훈련 데이터는 다운로드하지 않았습니다.

## 승인된 suite map 게시

사용자가 제공한 Drive 루트 아래 원본 file ID `1Qaod3lYklGRd0vdzq42AoInOk9yoJFju`의 suite_map.json을 확인했습니다. by_instruction 40개이며 spatial/object/goal/libero_10이 각각 10개입니다. 사용자의 명시적 승인 후 비공개 model `VLADE-konkuk/VLADE-checkpoints`에 `suite_map_seed42`로 게시했습니다.

- asset revision: `ca06ce16238d46387b2b972bedc35a35259a4a4c`
- registry revision: `cafe64bc4267a3fa200a15743d7d8a24dc5436c4`
- 원본 크기: 2914 bytes
- SHA256: `06de171dd29b07bf5e4fb1283a3d1355a2e96a4169c07c850e9ae45253002efa`
- 원격 파일 확인 후 정식 resolve_input으로 다시 받아 URI/캐시 검증과 mapping parser를 확인했습니다. HF lock·Suite 설정에 연결했고 pending은 없습니다. Drive 원본은 변경하지 않았습니다. 증거는 `reports/suite-map-publication.json`입니다.

## 실제 검증

- 최종 전체 하네스: 299 passed, 3 skipped, SWIG deprecation warning 2개, 273.57초. 누락된 작은 google-crc32c를 scratch에만 설치한 후 건너뛴 CRC/전체 프레임 재시작 검사 3개도 모두 통과했습니다. 서로 다른 302개 검사에 대해 통과를 확인했습니다. 이동한 합성 학습 CLI도 직접 실행해 4 step 및 정확한 저장/재로딩/재개를 확인했습니다. 명령·결과는 `reports/validation-results.json`에 있습니다.
- 모델 원본과의 활성 정의·출력·gradient·상태 호환, 원본 디렉터리 없는 16개 파일의 별도 Teacher import/두 프레임 backward/L1/strict reload, 실제 축소 native Llama backward 및 Qwen vision/merger/언어 경로를 검사했습니다. 합성 vision/language 검사와 실제 축소 백본 검사를 구분합니다.
- HF 입력 거부/수락·무결성·checkpoint/RNG 재개·Hub 실패/재시도·전체 프레임 전송 재시작·Student gate·suite gate·설정 전달·paired rollout 계수를 검사했습니다. 시뮬레이터/대형 정책은 합성 fixture로 대체한 검증이며 실제 성공률이 아닙니다.
- Python 107개 compile, 연구 config 18개 검증, 노트북 10개 코드 셀 compile, 원본 Prismatic import를 막은 상태에서 런타임 모듈 23개 import가 통과했습니다. 상세 수치는 `reports/runtime-static-verification.json`입니다.
- `git diff --check` 통과. branch/HEAD, .gitmodules와 두 submodule gitlink가 시작 상태와 같았습니다. Git add/commit/push는 하지 않았습니다.

## 미실행 범위와 사용 전제

실제 Colab 패키지 설치, 7B DINOv2/SigLIP/Llama, pretrained 0.8B GPU 학습/KD, LIBERO(+) 실제 시뮬레이터 rollout과 CALVIN/Jetson/representation KD 후속 실험은 수행하지 않았습니다. 전체 GPU 연구가 검증되었다고 보고하지 않습니다. 기존 query-head Qwen trainer를 유지하며 별도 QwenVLA 모델을 해당 trainer로 학습했다고 주장하지 않습니다.

새 candidate checkpoint·전체 Teacher cache·실측 Teacher 비교 보고서는 실제 생성·검증·HF 게시가 선행되어야 합니다. suite map의 전체 prepared 데이터 coverage도 실제 입력 사전 검사에서 확인합니다. 현재 로컬 변경은 미커밋이므로 새 Colab에서 사용하려면 사용자가 검토 후 Git에 반영해야 합니다.

현재 실행 안내 정본은 `docs/environment/research_runs.md`입니다. 작업 scratch와 tests/unit의 기존 검증 전용 bytecode cache를 경계 확인 후 삭제했습니다. 기존 정상 구현의 과거 bytecode cache와 다른 작업 기록은 보존합니다.
