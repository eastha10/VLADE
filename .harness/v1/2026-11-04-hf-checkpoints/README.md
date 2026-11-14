# 비공개 HF 자동 체크포인트 연동

2026-11-04 한국 시간. 실제 Desktop/vlade 저장소 HEAD a080bda와 승인된 Drive VLADE를 확인하고 작업했습니다. 기존 미커밋 변경·외부 소스·데이터·원본 가중치를 보존했으며 commit/push는 하지 않았습니다.

- FFT/Qwen Task-only·KD/04 AVA의 기존 optimizer 루프를 유지하고 봉인된 checkpoint의 private Hub 업로드를 연결했습니다. loop.py는 수정하지 않았습니다.
- hub.py와 CLI를 추가했습니다. 완료 payload와 last/best/final pointer를 한 commit으로 게시하고, 실패 시 로컬 FIFO 재전송 목록을 유지합니다. optimizer/scheduler/RNG 재개, SHA 무결성, 다운로드 revision 고정, 비공개 확인, RNG 보존을 포함합니다.
- FFT/Qwen의 검증 개선 시 추가 저장, 04의 저장된 train MAE best를 구분합니다. 목표 step 완료만 final입니다. HF 비활성 시 기존 로컬 저장 동작입니다.
- 01~05 및 Drive 전용 Suite notebook에 Secrets/환경 변수 설정을 추가했습니다. 대상은 사용자 지정 eastha/VLADE-checkpoints이며 기본 ENABLE_HUB_UPLOAD=False입니다. Suite pilot4/main은 별도 group입니다. 사용자 셀·출력·metadata를 보존했고 작업 중 갱신된 Drive 01 출력도 다시 읽어 병합했습니다.
- 로컬 정식 파일 16개 및 Drive 18개를 반영했습니다. Drive 전용 suite adapter는 로컬 Git 소스로 새로 추가하지 않았습니다. 두 환경의 기존 manifest 차이를 보존하면서 각각 hash를 갱신했습니다.

## 검증

Anaconda Python, torch 2.12.0+cpu, pytest 8.3.4, huggingface_hub 1.16.1을 사용했습니다. 추가 패키지·모델을 설치하거나 GPU 학습을 실행하지 않았습니다.

- 실제 저장소 전체 tests/unit: 167 passed, 1 temporary Windows PermissionError at dataset-fixture rename (pytest-03).
- 해당 test_final_data_review.py 재실행: 14 passed in 37.61s (pytest-04).
- RNG를 소비하는 모의 업로드를 포함한 최종 Hub/Colab/paper 관련 재검사: 52 passed in 48.85s (pytest-05).
- 실제 CPU 작은 모델의 업로드/다운로드 후 AdamW·scheduler·RNG 복원과 다음 파라미터가 연속 학습과 정확히 일치했습니다. 이는 모의 Hub와 CPU 검증이며 실제 HF 서버나 GPU 검증이 아닙니다.
- git diff --check --ignore-submodules=all 통과, 실제 source manifest 전체 SHA 불일치 0, 로컬 전달본 16개 bytes 일치.
- Drive 18개를 재조회하여 SHA256 bytes 일치. notebook MIME도 확인했습니다. 6개 notebook의 bootstrap digest와 Suite embedded adapter SHA를 확인했습니다. Drive 검증 목록은 reports/drive-verification.json에 보존합니다.

HF credential 존재 여부만 확인했고 원문은 읽거나 출력하지 않았습니다. whoami는 서버 연결 ConnectError로 실패했습니다. 실제 대상의 private 여부·쓰기 권한·대용량 upload·Colab GPU 재개는 미검증입니다. 사용자 토큰은 Colab Secrets HF_TOKEN에 등록해야 합니다.

정식 사용법: docs/environment/hugging_face_checkpoints.md. 후속 조직 namespace 전환은 HF_REPO_ID 환경 설정으로 가능하며 조직 생성·데이터 이관은 수행하지 않았습니다.

합성 테스트 scratch는 완료 시 정리했습니다.


## 후속 조직 HF 자료 참조·이관 및 노트북 정리 (2026-11-04)

- 사용자 지정 VLADE-konkuk 조직을 사용합니다. 실제 로컬·Drive 파일 접근을 확인한 뒤 로컬 OFT 4-in-1 약 15.9GB를 private model 저장소에 게시하고 원격 크기/LFS SHA256/Git blob을 확인했습니다. 해당 asset pin은 f98f6a23f28e62aee95b9b6ed6b976fbe2f5db93입니다. 공개 Qwen·OpenVLA·raw RLDS 및 OFT 개별 suite는 원래 HF 고정 revision을 보존했습니다. OFT 개별 suite를 4-in-1로 대체하지 않았습니다.
- storage/assets.py, migrate.py, HF 논리 참조 설정과 Colab 입력 SSD 캐시를 추가했습니다. 전처리 TAR 해시·manifest/splits/normalization identity, archive 경로·symlink·중복 항목, 완료 디렉터리 hash를 검증합니다. 게시 완료 자료만 immutable registry에 등록합니다. Teacher full cache만 별도 private Dataset에 게시합니다.
- 기존 loop.py·손실·optimizer·schedule은 변경하지 않았습니다. 봉인된 checkpoint와 last/best/final pointer의 한 commit 게시 및 FIFO 실패 재전송, optimizer/scheduler/RNG 재개 기능은 이전 단계에서 연결했습니다. tokenizer.json을 비밀 파일로 오인하던 검사도 고쳤습니다. 확인되지 않은 과거 checkpoint를 best로 지정하지 않습니다.
- Drive 전용 episode sampler 확장·Suite adapter를 정식 소스로 보존했습니다. 기존 source manifest에서 누락되었던 episode_data.py hash mismatch를 수정했습니다. Drive의 기존 upstream·문서 파일별 hash 차이는 보존하므로 local/Drive manifest 자체 digest는 다릅니다. 각 환경의 여섯 notebook은 해당 환경 manifest와 일치합니다.
- 사용자 최종 지시대로 Colab 01의 기존 download/extract/prepare 셀을 원문·metadata·execution_count·outputs까지 동일하게 보존하고 purpose 구역만 접었습니다. 이관 셀은 맨 위 한 개로 유지하고 venv --without-pip + 호스트 pip --python 방식으로 교체했습니다. 기존 Secret HF_TOKEN을 읽으며 토큰 재등록·코드 저장을 요구하지 않습니다.
- 정식 변경 40개를 로컬과 Drive에 반영했습니다. Drive는 같은 파일 ID를 유지했고, 새 파일 3개만 해당 원래 폴더에 생성했습니다. 40개를 다시 읽어 bytes SHA 일치, 여섯 bootstrap digest·notebook MIME·접힘 상태·기존 출력 보존을 확인했습니다. reports/drive-assets-verification.json 참조.
- 실제 전체 CPU unit suite 177 passed in 133.99s (pytest-assets-01), 최종 storage/Hub/Colab/baseline 관련 55 passed in 22.05s (pytest-assets-02), 원래 준비 셀 보존 후 source generator 12 passed in 0.73s (pytest-assets-03). 이후 Suite 재생성의 다른 셀·출력·metadata 보존을 실제 notebook 복사 한 개로 확인했습니다. git diff --cached --check 통과, 검토한 파일 목록 40개 일치·HF 토큰 형태 문자열 없음·submodule/사용자 skeleton 제외를 확인했습니다.
- GitHub dev/eastha에 ceb8c9021ea5f300aee5f4b378f021b17eba1ee3을 commit/push하고 ls-remote가 같은 SHA임을 확인했습니다. 사용자 skeleton과 기존 다른 .harness 작업은 그대로 두었습니다.

마지막 실제 HF 확인: model과 prepared dataset 저장소 모두 private=True. 등록 asset 8개. prepared_d_seed42, ava_t0, ava_fft_step_1975는 아직 완료 registry에 없습니다. 원본은 삭제하지 않았고 GPU 학습은 실행하지 않았습니다. Drive 도구의 256MiB 파일 전송 제한으로 실제 11GB TAR·기존 약31GB FFT 상태 전송은 Colab에서 실행해야 합니다. Colab 실행 UI 조작이 불가하여 기존 맨 위 이관 셀 재실행을 사용자께 요청했으며 실행 완료 응답은 아직 없습니다. reports/hf-assets-status.json 참조.

과거 step-1975의 optimizer/scheduler/RNG는 이관 대상에 포함하지만 당시 코드와 이번 코드의 fingerprint가 달라 기존 엄격한 재개 검사는 거부할 수 있습니다. 이 검사를 완화하지 않았으며 GPU 재개 성공으로 보고하지 않습니다. Suite의 audited suite_map과 실제 생성 전의 full Teacher cache도 미확인/미생성입니다. 관련 자료가 없으면 추정하거나 다른 자료로 대체하지 않고 명시적으로 중단합니다.


## 이관 실시간 로그·재시도 보완 (2026-11-04)

사용자 실행 후 private HF의 prepared_d_seed42(11,371,123,200 bytes, revision 6d2542577647a9e291554524af4e1dd0928e9df6)와 ava_t0(23 files, 15,555,626,250 bytes, revision 4c12bf032fdccbede71b0527f27528a8be4f90f6)가 실제 게시되었습니다. 두 자료의 고정 commit·원격 LFS SHA256/작은 Git 파일 SHA256 및 정본 식별자를 독립 확인했습니다. reports/hf-migration-verification.json 참조.

이후 사용자가 CalledProcessError(exit 1)를 보고했습니다. 저장된 Colab 출력에도 상위 오류만 있고 자식 stdout이 없어 상세 원인은 아직 미확정입니다. ava_fft_step_1975는 미게시 상태입니다. 이를 완료로 처리하지 않았습니다.

- 기존 이관 셀 한 개를 subprocess.Popen(stdout=PIPE, stderr=STDOUT), Python -u/PYTHONUNBUFFERED, 줄별 flush 출력으로 교체했습니다. runtime-local /content/vlade_hf_transfer/migration.log에도 토큰을 가린 출력을 기록합니다. 실패 JSON에는 message와 where를 포함하고 HF 토큰 및 URL query/basic credentials는 가립니다.
- 이관 스크립트는 이미 등록된 불변 private 자산의 원격 파일 크기/해시를 확인한 뒤 재사용합니다. 전처리·AVA Drive 원본을 다시 읽거나 재업로드하지 않습니다. 기존 Python/HF 환경도 재사용합니다. 단계별 상태와 checkpoint 탐색 목록을 출력합니다.
- 완료된 두 자산을 로컬 lock에 반영하고 local/Drive별 manifest와 여섯 notebook digest를 갱신했습니다. 원래 다운로드·추출·전처리 셀·metadata·출력 및 이관 셀 ID는 보존했습니다. 새 셀은 추가하지 않았습니다.
- CPU HF 자산/Colab source 검사 26 passed in 1.74s. 실제 HF와 연결한 빈 Drive fixture에서 두 자산 reused_verified_remote 후 checkpoint_inventory=[]로 안전 중단하는 것을 확인했습니다. 실제 Drive 대용량 재읽기/재전송은 없었습니다.
- Drive 변경 11개를 재조회해 bytes 일치와 셀 compile을 확인했습니다. reports/drive-log-fix-verification.json 참조. 사용자 요청에 따라 GitHub dev/eastha에도 같은 변경을 commit/push했습니다.

FFT 이관의 구체적인 실패 원인과 실제 업로드 완료는 아직 확인 전입니다. 수정된 기존 셀의 재실행 결과가 필요합니다. 미완료 이관 확인을 위한 scratch의 작은 조회·검증 스크립트/HTTP metadata 캐시는 후속 확인까지 유지합니다. GPU 학습·과거 checkpoint 호환성 검사를 실행하거나 완화하지 않았습니다.

GitHub 로깅 수정 최종 확인: 첫 push는 서버 commit_refs 오류로 거부되었으나 일반 push 재시도가 성공했습니다. local HEAD와 원격 dev/eastha가 45feeb4707fbc234611343628965f99fe0fc8a55로 일치합니다.


## Drive 복사본 체크포인트 인식 보완 (2026-11-04)

사용자 실시간 로그에서 step-1975 마운트 폴더는 발견되었지만 checkpoint_sources가 COMPLETE/검증 목록을 인식하지 못한 것을 확인했습니다. Drive API 폴더 ID 1YF8pC6rg55KIQrLWAVE5lcvdXAvAhHCB의 원본에는 COMPLETE의 사본(3 bytes)과 checkpoint_manifest.json의 사본(4112 bytes), seal의 27개 payload 파일이 존재합니다. API 파일명은 NFC이며 실제 마운트 인식 실패가 Unicode 조합/속성 조회 중 어느 지점 때문인지는 미확정입니다. 완료 표시가 실제 없다고 단정하지 않았습니다.

- 새 정식 helper src/vlade/storage/drive_checkpoint.py로 복사본 이름 인식과 seal 검사를 분리했습니다. Unicode NFC·공백·복사 접미사를 처리하고 중복 canonical 이름, 누락, 잘못된 경로와 상태 없는 checkpoint를 거부합니다.
- 기존 이관 셀은 마운트에서 sealed step-1975를 정상 인식하면 그대로 사용합니다. 인식하지 못하면 표준 Colab Google 인증/읽기 전용 Drive API 요청으로 확인된 폴더 ID의 원본을 읽습니다. 폴더·파일명·원본 완료 표시는 변경하지 않습니다. HF_TOKEN을 Google 요청에 사용하지 않습니다.
- runtime-local seal-hash 디렉터리에만 임시 준비하고 원래 27개 payload의 크기/SHA256을 확인한 뒤 원본 COMPLETE를 읽습니다. 원격 다운로드의 Range 이어받기·완료 파일 재사용·진행 바이트 출력·실제 남은 용량 검사를 포함합니다. 깨진 복사본을 완료로 간주하지 않습니다. 이미 게시된 전처리와 AVA는 재사용합니다.
- 새 helper 호출만 기존 셀에 연결하고 추가 셀을 만들지 않았습니다. 원래 다운로드/추출/전처리 셀과 metadata·실행 결과를 보존했습니다. source guard 6개·local/Drive manifest·여섯 notebook digest를 함께 갱신했습니다.
- CPU 검사 32 passed in 5.75s. 모의 Drive API로 정상/분해형 한글 복사 이름·애매한 이름 거부·부분 다운로드 이어받기·완료 파일 재사용·누락 완료 표시/잘못된 hash 거부·정상 마운트의 Google 인증 불필요를 확인했습니다. 실제 Google 계정의 API 다운로드와 FFT 게시 완료는 아직 사용자 Colab 실행 전입니다.
- Drive 변경 12개를 재조회해 bytes 일치와 셀 compile을 확인했습니다(reports/drive-copy-fix-verification.json). GitHub dev/eastha 46b22f3bd0f838edf722845a77ddaa29bd30949c commit/push 후 원격 SHA 일치를 확인했습니다. 기존 사용자 skeleton·다른 작업 기록은 유지했습니다.


## 실제 이관 완료 및 최종 참조 고정 (2026-11-04)

사용자께서 셀 종료를 알린 후 HF를 직접 조회했습니다. 최종 registry commit은 8602a89f326eb683dc45b79229de03966a371cba이며 pending은 빈 목록입니다. model/dataset 저장소 모두 private=True입니다.

- prepared_d_seed42: 11,371,123,200 bytes, immutable revision 6d2542577647a9e291554524af4e1dd0928e9df6.
- ava_t0: 23 files, 15,555,626,250 bytes, immutable revision 4c12bf032fdccbede71b0527f27528a8be4f90f6.
- ava_fft_step_1975: 29 files, 30,691,562,124 bytes, immutable revision c685b31d7593128ac2f45ff91cf4629e57ba266a. COMPLETE, checkpoint_manifest.json, trainer_state.pt 존재 및 seal의 27개 payload row와 registry 일치, step=1975/microsteps=15800을 확인했습니다.

세 자산을 각각 pinned commit에서 조회해 remote size/LFS SHA256을 대조하고 작은 Git 파일은 실제 다운로드해 SHA256 및 Git blob SHA1을 확인했습니다. 원본 canonical D의 세 identity SHA도 일치합니다. 대용량 모델/optimizer 파일은 이 검증을 위해 다시 내려받거나 deserialize하지 않았습니다. reports/hf-migration-verification.json에 결과를 보존합니다. 앞서 로컬 OFT 4-in-1도 verified pin f98f6a23f28e62aee95b9b6ed6b976fbe2f5db93으로 게시되어 있습니다.

최종 HF registry를 로컬 lock으로 반영하고 local/Drive 각각 기존 source manifest와 여섯 bootstrap digest를 맞췄습니다. 원래 다운로드·전처리 셀, 이관 셀의 코드/실행 결과/metadata·사용자 선택 스위치는 보존했습니다. Drive 8개를 재조회해 bytes 일치 및 여섯 notebook source digest·compile, local/Drive lock 동일을 확인했습니다(reports/drive-final-pins-verification.json). 설정·pin 변경만 있어 JSON/hash/compile 검증을 수행했고 이미 통과한 CPU 학습 검사를 이유 없이 반복하지 않았습니다.

GitHub dev/eastha에 bf39b3e45cb43cd47d27febc125ad79f309ca913 commit/push를 완료하고 원격 SHA 일치를 확인했습니다. 원본 Drive/로컬 데이터·가중치를 삭제하거나 이름을 바꾸지 않았습니다. 이관 셀을 다시 실행할 필요는 없습니다. 원래 다운로드·전처리 구간은 접힌 상태로 보존했습니다.

완료 범위는 기존 자료의 HF 이관·원격 무결성 검증·정본 코드의 HF 참조 및 checkpoint 자동 업로드 연결입니다. 실제 GPU 학습/재개는 실행하지 않았습니다. 과거 step-1975 코드 fingerprint와 새 코드가 다르면 엄격한 기존 resume 검사는 그대로 거부할 수 있습니다. 아직 생성하지 않은 full Teacher cache와 미확인 audited suite_map을 생성·업로드 완료한 것으로 표현하지 않습니다. 이번 작업의 임시 조회 캐시·staging·CPU fixture scratch는 완료 후 정리합니다.
