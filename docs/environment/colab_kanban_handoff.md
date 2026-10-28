# 27일 마감 업무 · Colab 실행·검토 인계

작성일: 2026-10-28 한국 시간. 사용자 지정 검토 예정: **2026-10-28 아침**.
25일 연계 검토 3건과 cache/Task-only 실행 2건은
[통합 검토 페이지](https://app.notion.com/p/406b5dd068708197bcc9df8b8f79e175)에서 함께 확인합니다.

주 담당 김동하의 10월 27일 마감 카드는
[고정 Teacher cache](https://app.notion.com/p/403b5dd0687081bcbf1ef3b79d6acd73)와
[Qwen Task-only](https://app.notion.com/p/403b5dd068708126a4a5c5559d807981)입니다.
이번 산출물은 두 작업을 실행하기 위한 코드·노트북이며 실제 GPU 학습 결과가 아닙니다.

## 실행 파일의 위치와 역할

수정할 코드는 저장소의 일반 파일로 유지합니다. Colab에서 열 노트북과 설정·검토 문서도
아래 정본 경로를 사용합니다.

| 디렉터리·파일 | 사용 목적 |
|---|---|
| `src/vlade/data/` | 공식 자료 다운로드·무결성 검사, episode 분할·정규화·관측 참조 |
| `src/vlade/training/` | FFT·Student·공통 loop, 저장/재개·cache·실행 조건 검사 |
| `scripts/data/` | 데이터 준비·전처리·T1 선정·고정 cache를 실행하는 명령 |
| `scripts/train/` | 학습 실행·CPU smoke·실제 checkpoint 재로딩 검사 명령 |
| `scripts/prepare/generate_colab_notebooks.py` | 인자 없이 노트북과 source manifest를 갱신하는 명령 |
| `src/vlade/utils/colab_notebooks.py` | 위 노트북 생성 명령의 실제 구현 |
| `experiments/colab_01_download_data_weights.ipynb` | Drive 데이터·AVA 가중치 준비·전처리 |
| `experiments/colab_02_ava_fft_and_fixed_cache.ipynb` | AVA FFT·파일럿·T1 선정·cache |
| `experiments/colab_03_qwen_task_only.ipynb` | Qwen 가중치·파일럿·overfit·Task-only |
| `experiments/colab_04_openvla_oft_fft.ipynb` | OpenVLA-OFT 독립 환경 FFT·저장/재로딩 |
| `experiments/manifests/colab_source.json` | 사용할 파일 목록과 파일별 SHA256·source revision |
| `experiments/baselines/`, 원본 source manifests | 원본 비교 조건과 소스 이양 기록 |
| `configs/` | 모델·학습 조건·고정 revision |
| `docs/`, `tests/` | 연구/실행 검토와 회귀 검사 |

Drive의 `ROOT=/content/drive/MyDrive/VLADE`, `PROJECT_ROOT=ROOT`를 기본으로 사용합니다.
이 디렉터리 바로 아래에 `src/`, `scripts/`, `configs/`, `third_party/`가 있어야 합니다.
노트북은 `experiments/manifests/colab_source.json`과 일반 파일의 SHA를 검증한 뒤 필요한
파일을 `/content`로 복사해 실행합니다. AVA와 OpenVLA-OFT는 각각 고정된 upstream과
별도 환경입니다. baseline과 원본 manifest 기록은 함께 보관합니다.

## 추가 교차 검토·수정 결과

데이터/재개, checkpoint/cache, 노트북/다운로드를 나누어 독립 검토하고 아래 실제 결함을 수정했습니다.

| 발견한 문제 | 수정 내용 |
|---|---|
| 분할 정본과 실제 manifest 분할이 달라도 로딩 | splits와 episode ID/content/split/steps/queries·counts·normalization binding을 대조하고, 실제 trajectory/query 길이·순서·관측 참조도 검사 |
| 같은 run의 다른 optimizer 파일이 섞여 Student 재개 | 모델·optimizer·설정·processor 파일의 크기/SHA256과 step/microstep을 COMPLETE 전에 봉인하고, 재개/선정/파일럿에서 검증 |
| 타 episode의 정상 sealed part가 cache에 혼합 | 모든 part의 episode ID/시점/언어/proprio/mask/관측 참조/start-end/Teacher step과 hidden 모드·layer·dimension을 원본과 대조 |
| 중단 orphan이 완료 cache 용량으로 계산 | 완료 parts의 bytes만 집계하고 incomplete_bytes를 별도로 보고 |
| 다른 코드·데이터의 passed 파일럿 보고서 재사용 | 보고서를 현재 코드·데이터·파일럿 설정·봉인 checkpoint·overfit 판정 기준과 결합하고 본 학습 진입 시 대조 |
| 새 T1 입력을 기존 selection이 조용히 무시 | checkpoint·평가 보고서·선정 이유가 다르면 실패하고 새 SELECTION 경로를 요구. T0/T1의 빈/동일 경로도 거부 |
| 분할 ZIP 결합/해제 중단 후 이름 충돌 | attempt별 고유 임시 경로와 소유 표식을 사용. 실패한 자기 작업만 정리하고, 같은 POSIX host의 종료된 작업만 회수. 활동 중/다른 host/미인식 파일은 보존 |
| 과거 run을 최신 보고서로 표시 | UUID 사전순 대신 수정 시각으로 최신 run 선택 |
| prepared 학습/cache에서 TF GPU 점유 가능 | upstream import 전에 모든 AVA 학습·순차 추론 경로에서 TensorFlow GPU 비활성화 |
| Windows의 일시적인 staging 잠금 | 같은 parent·덮어쓰기 금지 조건으로 PermissionError만 최대 5회 짧게 재시도하고 지속 실패는 전파 |

배치 수정 전 교차 검토의 CPU 회귀 검사는 **115 passed / 35.26초**였습니다. 당시 명령과 실행 조건은
`.harness/v0/2026-10-28-colab-kanban/checkpoint-and-artifact-review.json`에 기록했습니다.
일반 파일 전환 후에는 manifest 생성·복사·누락/변조 거부·revision 복원 검사 12개를 추가했고,
전체 CPU 회귀 검사는 **127 passed / 40.24초**입니다. 현재 배치와 검증은
`.harness/v0/2026-10-28-colab-kanban/source-layout-verification.json`에서 확인합니다.
새 fingerprint·checkpoint 봉인 형식이므로 기존 구버전 pilot/보고서와 섞지 말고 새 run 이름으로
파일럿부터 실행하십시오. 이전 77개 검사는 보완 전 이력입니다. checkpoint 봉인/검증은 전체 파일을
읽으므로 실제 Drive I/O 시간이 추가되며 이는 GPU 환경에서 측정해야 합니다.

압축 해제는 중단된 attempt에서 새로 시작하며 byte-level 이어풀기는 아닙니다. 미인식 이전 임시 파일은
삭제하지 않으므로 공간은 별도 확인합니다. 전처리 query 준비 단계는 기존처럼 새 output 경로로 재실행합니다.

## 보완 완료: 실제 데이터 기반 학습량과 실행 조건

2026-10-28 사용자가 D/content-group 80/10/10, seed 42를 최종 확정했습니다.
Train/Validation/Test는 11,477/1,428/1,442 Episode이고 Train 내부 Calibration은 728개입니다.
후보 원본 SHA-256 `487d2be...6e80`을 그대로 canonical manifest로 보존했으며, 전체 원본
1,024 shard를 재판독해 group membership과 Train 1,791,269 transition 전용 normalization을
계산했습니다. 이전 85/5/5/5 통계는 사용하지 않습니다.

이전 노트북의 고정 1000-step을 제거했습니다. 새 Colab 본 학습은 실제 manifest의 train
query 수 N과 `MAIN_EPOCHS`로 `ceil(N × epochs / (batch × accumulation))`을 계산합니다.
기본 `MAIN_EPOCHS=1`은 전체 train query를 최소 한 번 순회하는 시작 설정이며, 최종 수렴에
충분하다고 확정한 값은 아닙니다. accumulation 경계 올림으로 다음 epoch에서 추가로 읽는
query 수까지 표시합니다. query 간격 8과 모든 timestep을 개별 관측으로 넣는 학습은 구분합니다.

Teacher/Student는 공통 seed 기반 episode shuffle을 사용하고 episode 내부 시간 순서를
유지합니다. prepared-data 재개는 epoch 순서와 query cursor를 복원합니다. 본 학습 플래그는
기본 False이며, 실제 1-step/reload 보고서와 Student overfit 감소 검사 통과 후 명시적으로
켜야 실행합니다. 원래 CLI/기본 JSON의 legacy 1000-step 값은 새 Colab 본 학습에서 사용하지 않습니다.

과거 임시값을 재사용하고 그 영향을 충분히 설명하지 않은 경위는 Notion 이력에 남겼습니다.
이전 코드와 source fingerprint가 바뀌었으므로 과거 checkpoint/보고서를 새 코드에 섞지 않고
새 run 이름으로 파일럿부터 확인합니다. 최종 epoch·LR·sampler/augmentation 정책은 사용자 검토 대상입니다.

## 검토 후 실행 순서

1. 저장소 루트에서 `python scripts/prepare/generate_colab_notebooks.py`를 실행해
   `experiments/colab_01~04` 노트북과 `experiments/manifests/colab_source.json`을 갱신합니다.
   현재 일반 소스·고정 AVA/OFT upstream·라이선스·설정·baseline·canonical manifest를 같은 파일 트리로
   준비합니다. GitHub HEAD만 clone한 사본에는 미커밋 구현이 빠질 수 있으므로 실제 작업 파일과
   그 파일에서 생성한 manifest를 함께 사용하십시오.
2. 이 파일 트리를 Drive의 `/content/drive/MyDrive/VLADE`에 배치하고 `experiments/`의
   `colab_01_download_data_weights.ipynb`를 Colab에서 엽니다. 네 노트북 모두 `ROOT`와
   `PROJECT_ROOT=ROOT`를 같은 실제 저장소 루트로 지정합니다. 시작 셀의 manifest·파일 SHA 검증이
   통과하면 필요한 일반 파일이 `/content`의 실행 디렉터리로 복사됩니다.
3. 공식 LIBERO+ `.z01/.z02/.zip` 세 part와 공개 AVA/OFT 전체 bundle을 Drive에 받고,
   공식 메타데이터와 파일 해시를 대조합니다. 세 part를 Info-ZIP으로 결합한 뒤 CRC를
   검사하며 추출합니다. 완료 표식 없는 추출/전처리는 학습에 사용하지 않습니다.
4. TFDS episode를 공통 prepared format으로 변환합니다. query 간격 8, 두 카메라,
   raw state의 첫 6+마지막 2차원 proprio, 상대 action 6차원과 gripper 규약을 사용합니다.
   exact action+state group을 canonical manifest와 대조해 split을 적용하고 train 전체 timestep만으로
   q01/q99를 재검산합니다. Query JPEG/PNG payload와 숫자 trajectory의 해시를 저장합니다.
5. A100에서 `colab_02_ava_fft_and_fixed_cache.ipynb`를 실행합니다. Python 3.11·AVA
   Transformers fork를 별도로 설치합니다. 실제 1-step backward/save/reload를 먼저
   확인한 뒤 실제 query 수·설정 epoch에 맞는 별도 실험을 실행합니다. 본 학습 플래그는
   기본 False입니다. dtype BF16, batch 1, 누적 8,
   paged 8-bit AdamW이며 LoRA는 사용하지 않습니다. 최종 budget은 검토 전 잠정값입니다.
6. T0/T1 validation·rollout 비교 후 선택 checkpoint·보고서·선정 이유를 지정합니다.
   마지막 저장을 자동으로 T1으로 선택하지 않습니다. 선택된 T1의 해시와 split·통계
   일치, frozen·비양자화 상태를 확인한 후 2-episode cache pilot을 실행합니다.
7. Pilot의 실제 용량·재실행 오차를 확인한 후 `RUN_FULL_CACHE=True`로 전체 train을
   생성합니다. 기본은 action-only이며 `SAVE_HIDDEN=True`는 layer -2 `[56,D]` 저장입니다.
   두 모드는 별도 경로를 사용합니다. 최대 3개 고르게 떨어진 episode의 전체 query를 재실행
   비교합니다. hidden JSON의 실제 용량을 먼저 측정하십시오.
8. 별도 환경에서 `colab_03_qwen_task_only.ipynb`를 실행합니다. 실제 Qwen 가중치는
   이 노트북에서 별도 다운로드합니다. 1-step·재로딩, 1-episode 50-step overfit를
   확인한 다음 동일 split·초기 revision·seed와 실제 query/epoch budget의 BC를 실행합니다.
   Overfit loss 초기/마지막 5개 중앙값 비율이 기본 0.8 이하여야 본 학습 검사를 통과합니다.
   이는 파일럿 감소 검사 기준이며 완전 수렴/성공률 기준은 아닙니다.
   Validation L1은 rollout 성공률과 구분합니다.

## 저장·재개

- 데이터/가중치/실험 설정/metrics/checkpoint/cache는 공통 ROOT의 Drive에 저장합니다.
  설치 환경과 소스 실행본, 결합 ZIP은 `/content`에 두므로 런타임 종료 후 재생성됩니다.
- `experiments/manifests/colab_source.json`과 그 파일에 기록된 일반 소스의 SHA256을 검사합니다.
  소스를 수정했으면 생성 명령을 다시 실행해 노트북·manifest를 갱신하고 Drive 파일도 맞춥니다.
  소스와 설정을 바꾼 실행은 별도 run 이름을 사용합니다.
- Teacher는 `COMPLETE`+`trainer_state.pt`가 있는 마지막 step에서 optimizer/scheduler/RNG를
  복원합니다. 새 prepared-data 경로는 공통 seeded episode shuffle·query cursor도 복원합니다.
  기존 raw RLDS 경로만 iterator 재시작 제한이 남습니다. GPU kernel bitwise 결정성은 미검증입니다.
- Student는 deterministic causal query cursor를 microstep 수로 건너뛰고 상태를 복원합니다.
  동일 소스·설정·Torch/Transformers·데이터/초기 manifest를 요구합니다. GPU kernel의
  bitwise 결정성은 별도 문제이며 보장하지 않습니다.
- Cache는 episode별 밀봉 후 재사용합니다. checkpoint/split/통계/hidden 여부/범위/코드 SHA·
  Torch/Transformers가 다른 출력을 거부하며 최대 3개 episode를 재실행 비교합니다.
  검증 실패 시 기존 COMPLETE를 남기지 않습니다.
- 1-step smoke는 본 실험과 경로와 budget이 다릅니다. 다른 목표량의 본 실험으로 이어붙이지 않습니다.
- 전처리는 완료 전 중단하면 전체 전처리부터 새 output 경로에서 재실행합니다. 원본
  archive/가중치 다운로드는 Hub cache로 이어받고 완료 추출은 재사용합니다.

## 검토 우선순위와 아직 검토하지 않은 사항

| 우선순위 | 검토·결정 사항 | 현재 확인한 사실 / 남은 일 |
|---|---|---|
| 완료 | 분할 정본 | D/content-group 80/10/10 seed 42와 후보 SHA-256을 확정했고, canonical manifest·Train 전용 normalization·prepared-data 검증 경계를 연결했습니다. 실제 Colab prepared-data 생성은 아직 실행하지 않았습니다. |
| 실행 전 | AVA temporal FFT | 기존/이번 FFT는 zero-context입니다. cache inference의 이전 hidden 전달과 시간 문맥 학습은 다릅니다. episode 학습·state reset·TBPTT 및 gradient 경로는 아직 구현/실측하지 않았습니다. 이 제한을 인정하는 파일럿인지 본 실험 전 확장이 필요한지 결정해야 합니다. |
| 실행 전 | Colab 설치·메모리 | 2026-10-04 A100 단일 추론 성공과 FFT는 다릅니다. 이번 환경 설치, 실제 TFDS schema, 첫 backward·checkpoint 저장·재로딩을 Colab에서 확인해야 합니다. A100 40 GB에서 FFT OOM 가능성이 있으며 80 GB도 실측 전 보장하지 않습니다. |
| 실행 전 | 저장공간·속도 | 원본 ZIP 75.5 GB 외에 추출본·query 이미지·가중치·optimizer 포함 checkpoint가 추가됩니다. Drive 계정 quota와 Colab scratch의 실제 여유 공간을 확인하십시오. 다수 query 파일의 Drive I/O 비용도 아직 측정하지 않았습니다. |
| 실행 전 | Student 가중치·budget | 실제 pretrained 0.8B 검토/로딩은 아직 하지 않았습니다. Python 3.13, Torch 2.12.0, Transformers 5.9.0 환경은 CPU 계약 검증과 구분합니다. 새 본 학습량은 실제 query×epoch로 계산하며 최종 epoch·LR·batch/누적과 KD 공정 비교 budget은 검토 대상입니다. |
| 학습 후 | T1 선정 | T0와 T1의 원본 LIBERO 유지·LIBERO+ validation rollout, checkpoint 간 비교, 선정 지표·tie-break·영상·학습 비용이 필요합니다. 평가 환경/rollout runner와 이 노트북의 자동 연결은 아직 없습니다. |
| cache 전 | Feature 저장 | 기본 action-only cache에는 representation KD용 hidden이 없습니다. `[56,D]` JSON의 크기·속도와 대용량 tensor shard 전환을 검토해야 합니다. canonical action/horizon feature의 projection·layer 선택도 후속 결정입니다. |
| Student 검증 | 실제 입력 영향·overfit | 축소 무작위 native Qwen CPU 입력 영향 검사는 통과했으나 real 0.8B·real episode에서의 입력 모달리티 영향, overfit 수렴·validation·rollout은 아직입니다. |
| 연구 검토 | Student 설계 | 2-query×2-camera, 첫 frame 반복, head 밖 proprio memory, VLM 전체 도입, 이미지/token budget은 구현 선택입니다. 사용자의 검토·승인 완료로 표시하지 않습니다. |
| 연구 검토 | 재현과 변형 | 현재 Student는 원본 CoTinyVLA의 8-frame·Plan/Think 생성 재현이 아닙니다. VLA-AD 의미 감독·Plan/Think label 생성, Action/representation KD trainer·loss 가중치·공식 코드 이식은 아직 하지 않았습니다. |
| 후속 | 양자화·최종 평가 | Teacher PTQ와 Student KD는 독립 분기입니다. QVLA/ActQuant/QuantVLA의 실제 모듈·packed kernel·Qwen hybrid attention 지원, calibration subset, S* 선정·Ablation·여러 seed·Jetson 비용은 아직 검토/구현/측정해야 합니다. |
| 집필 | 논문·주장 | Related Work 원문 버전·원본 구현과 변형의 경계, 조건표·결과 양식은 준비했습니다. 공식 코드 commit·라이선스와 실제 이식 가능성의 추가 검토는 남아 있습니다. 미실행 결과를 0이나 성능 개선으로 쓰지 않습니다. |

## 25일 연계 문서 통합 검토

원래 세 검토 문서의 예정일은 10월 26일이었으나, 이번 사용자 요청으로 10월 28일 아침에
통합 검토합니다. Notion 통합 페이지에는 원래 체크리스트·파일 목록·결과 기록 양식과
cache/Task-only 두 실행 카드의 원래 완료 조건까지 옮겼습니다. 원본 문서는 이력을
보존하고 통합 페이지로 연결했으며, 사용자 검토가 끝나기 전에 완료로 바꾸지 않았습니다.

### Teacher 저장 코드·cache 연결 조건

원본: https://app.notion.com/p/404b5dd0687081ffb383cdc487f52a01

- [ ] GT/predicted action·Teacher token·선택 hidden의 저장 필드가 연구에 충분한지 검토합니다.
- [ ] episode 시작/종료·query 간격·관측 참조를 실제 adapter가 제공하는지 확인합니다.
- [ ] checkpoint bundle·normalization·split의 manifest/해시 정본을 정합니다.
- [ ] debug sampling/용량 제한과 fixed KD의 순서·무결성 검사를 검토합니다.
- [ ] T1 고정·adapter 연결이 준비됐는지 판단하고 cache 실행 가능/선행 대기를 기록합니다.

볼 파일: `src/vlade/data/teacher_records.py`, `tests/unit/test_teacher_records.py`,
`docs/environment/teacher_records_and_student.md`, 새 `src/vlade/training/fixed_cache.py`.

### Qwen Student 구조·출력 계약

원본: https://app.notion.com/p/404b5dd0687081209307da07bda9d7e7

- [ ] Qwen3.5 VLM 전체 도입과 언어부만 교체하는 대조 실험을 구분합니다.
- [ ] 두 카메라×최근 2 query·간격8·첫 프레임 반복·이미지/token budget을 확인합니다.
- [ ] head memory의 proprio와 8-query continuous MLP head 선택을 검토합니다.
- [ ] 정규화 action 8×7·유효 mask·horizon feature·Teacher 좌표/통계를 대조합니다.
- [ ] 축소 무작위 CPU 검증과 pretrained 0.8B, 원본 CoTinyVLA 8-frame·Plan/Think 경로를 구분합니다.
- [ ] Hugging Face 로딩 방식·실제 모델 가중치 검토/다운로드·후속 실행 전제를 확인합니다.

볼 파일: `configs/models/student/qwen3_5_0_8b.json`, `src/vlade/models/students/qwen.py`,
`src/vlade/models/contracts.py`, `src/vlade/models/students/smoke.py`,
`scripts/train/smoke_student.py`, `tests/unit/test_student_contract.py`.

### Related Work·논문 및 실험 양식

원본: https://app.notion.com/p/404b5dd0687081fdaec2e3c2e5b32a2e

- [ ] VLA-AD·CoTinyVLA·QVLA·ActQuant·QuantVLA 원문 링크/버전과 요약을 검토합니다.
- [ ] 기존 Method Draft의 π0.5·flow KD·horizon50과 현재 AVA/OFT·8×7 계약 차이를 확인합니다.
- [ ] 연구 질문·기여 후보·outline과 실제 근거 없는 주장의 수정 대상을 정합니다.
- [ ] Teacher 적응/Student 증류/Teacher 양자화/선정 Student 양자화의 네 비교 축과 재현/변형을 구분합니다.
- [ ] 조건표의 split·seed·budget·선정 규칙과 담당자의 선행 확정본을 연결합니다.
- [ ] 결과 JSON의 NR/미지원/실패·측정값 구분과 성능 미측정 표기를 검토합니다.

볼 파일: `docs/paper/preparation.md`, `docs/paper/experiment-record.template.json`.

### 공통 검토 결과 기록

각 항목의 결정은 아직 **미검토**입니다. 승인 / 수정 필요 / 선행 대기 중 하나와 이유,
승인한 내용, 수정 파일·내용·후속 검증, 선행 작업·보류 이유, 다음 실행을 통합 페이지에
기록합니다. 과거 55/65/77개 검사와 추가 교차 검토 후 검증 기록을 구분하고, 이를 실제 GPU 성능으로 해석하지 않습니다.

## 재점검 결과: 보완한 코드와 남은 결정

원본 검토 3건 17항목과 실행 2건 7항목, 합계 **24/24**가 통합 페이지에 보존됐음을
문자열 대조했습니다. 추가로 연구 프로토콜·데이터·Teacher 선정·평가 budget·S*·Jetson
카드와 실제 소스/노트북을 확인하여 아래 누락을 보완했습니다.

| 항목 | 보완한 동작 | 검증 범위 |
|---|---|---|
| 학습량·순회 | 고정 1000 대신 실제 train query×epoch 예산, 전체 episode shuffle·내부 causal 순서, rounding 수 표시 | 전체 query coverage·epoch 변경·cursor 복원 CPU 검사 |
| 자동 본 학습 | FFT/Student 본 학습 기본 False, reload/overfit 보고서 실패·누락 거부 | 실제 노트북 셀을 stub 실행하여 미진입/실패 차단 검사 |
| Validation | Colab Teacher 100-step마다 및 마지막 step, Student 주기/마지막 val L1, validation 전체 구간에 고르게 펼친 deterministic subset | subset 위치·test split 차단 CPU 검사; 실제 val loss 미측정 |
| Gradient | 첫 update 전에 vision/language/projector/action/proprio/AVA attention의 유한·nonzero 경로 확인, 누락 시 optimizer update 전 실패 | 사용하지 않는 모듈을 둔 실제 CPU autograd 검사 |
| 학습 중 debug | 저장된 checkpoint의 실제 bundle hash로 post-save 재추론한 action/GT/Teacher token/관측 참조를 bounded training_debug로 기록 | 예산·모드/RNG 복원·KD 오인 거부 CPU 검사; 실제 AVA debug 미실행 |
| 저장공간·완료 표식 | 파일럿 크기로 예정 checkpoint 총량 및 저장 직전 공간 검사, JSON atomic 교체, 실패 cache COMPLETE 제거 | 공간 부족·원본 JSON 보존·cache replay 실패 검사 |
| Cache 출처·표본 | 코드 SHA·software 차이 혼합 거부, 범위에 고르게 떨어진 최대 3개 episode 전체 query replay | 합성 순차 Teacher의 저장/재개·코드 변경 거부 검사 |
| T1 보고서 | T0/T1 실제 rollout 수치·원본 LIBERO 유지 수치·protocol hash·validation split·선정 checkpoint 일치·test 미사용 검사 | null/최종 test 거부, Teacher 향상 없음도 허용하는 CPU 검사 |

표준 보고서 양식은 `docs/paper/teacher-selection-report.template.json`입니다. null은 미측정이며
그대로는 선정되지 않습니다. schema 검사는 보고서 수치가 실제 평가에서 나온 것인지 대신
입증하지 않으므로 원시 rollout 기록과 평가 담당자의 근거 확인이 필요합니다.

사용자가 확인할 결정/실측은 다음과 같습니다.

- **최종 학습량:** 기본 1 epoch를 최종값으로 채택할지, 여러 epoch·LR/schedule·정지 규칙과
  파일럿 overfit 감소 기준 0.8을 어떻게 정할지. 실제 GPU 처리시간·곡선·val을 근거로 정합니다.
- **입력·샘플링:** query 간격8·2-query×2-camera·증강 off·head proprio와 action mask 가중치,
  Teacher 단일 시점 zero-context와 Student history의 통제 비교. 파라미터 전체가 trainable해도
  미사용 LM 출력 head 등 모든 tensor가 매 step 갱신된다는 의미는 아닙니다.
- **Temporal 연구 범위:** episode/TBPTT temporal FFT는 아직 미구현입니다. 순차 cache를
  구현한 것으로 원본 AVA temporal 학습까지 재현한 것으로 승인하지 않습니다.
- **데이터 분리의 의미:** 80/10/10은 확정입니다. exact-byte content grouping은 유사한 다른
  episode/task/초기상태·교란 조건의 누수나 미관측 조건 일반화를 보장하지 않습니다.
  공개 학습 RLDS와 평가용 LIBERO+ assets/held-out 시나리오는 별개이며 평가 환경 commit/
  adapter·assets 준비는 아직입니다. 분할·통계 정본은 확정됐으며 실제 Colab 생성 결과의 해시를 대조해야 합니다.
- **Validation·선정:** val L1 subset 크기/대표성, rollout metric·tie-break·허용 원본 LIBERO 저하·
  지연/메모리 한계, seed/trial 수·time cap·실패 처리·7교란 macro 산식을 사전 고정합니다.
  최종 test로 T1/S*를 다시 고르지 않습니다. 수치가 악화되면 그대로 기록합니다.
- **최소 비교 범위:** T0/T1, Task-only/T1-KD, T1/T1-Quant, S*/S*-Quant의 네 축,
  원설계의 T0-KD(S-KD0) 비교를 필수/후속 중 어디에 둘지, action/representation/semantic/temporal
  ablation과 동일 입력·초기화·처리 샘플 budget을 확정합니다. Teacher 양자화로 기본 KD를 바꾸지 않습니다.
- **Cache·KD:** action-only 또는 layer -2 hidden·투영/정렬·loss λ·sample matching, 저장 용량,
  생성 비용 포함 비교를 정합니다. KD trainer·Plan/Think/semantic label 생성은 후속 기능입니다.
- **재현·실측:** 실제 pretrained 가중치·TFDS schema·AVA/BnB 첫 GPU step·pre/post-save 동등성 및
  optimizer를 복원한 다음 update·실모델 입력 영향·Drive quota/I/O·설치된 전체 의존성 freeze를 확인합니다.
  현재 real reload 검사는 load 후 finite/repeat 검사이며 real GPU resume 다음 step까지 통과한 증거는 아닙니다.
- **양자화·edge·집필:** 비양자화 S* 선정 범위/미완료 후보, packed kernel·Qwen hybrid attention,
  calibration·평균 bpw/scale bytes, Jetson 보드/runtime/power·batch1/warmup/동기화·p50/p95·전체 peak memory,
  chunk 처리량과 실제 제어 주기, 원문/공식 코드 라이선스·재현/변형·NR 표기를 검토합니다.

담당 선행: 데이터·평가/Teacher 선정·환경/양자화 담당 카드의 정본을 받아 연결해야 합니다.
이 문서 정리가 다른 담당자의 실험 완료나 사용자의 설계 승인을 뜻하지 않습니다.

## 코드와 검증 범위

- `src/vlade/data/colab_assets.py`: 공식 revision 다운로드·SHA/LFS/git-blob 검사·분할 ZIP 추출.
- `src/vlade/data/episode_data.py`: content-group 분할, train-only exact 통계, 공통 query·mask·관측 참조.
- `src/vlade/training/prepared_fft.py`: 기존 upstream transform/collator로 연결하는 어댑터.
- `src/vlade/training/fixed_cache.py`: T1 selection 검증·순차 context·episode reset·재개·표본 replay.
- `src/vlade/training/student_task.py`: native Qwen Task-only full-backbone/head 학습·저장·재개·bounded val L1.
- `src/vlade/training/run_control.py`, `debug_records.py`: data-based budget·실행 조건·gradient/공간 검사·bounded checkpoint debug.
- `src/vlade/utils/colab_notebooks.py`, `scripts/prepare/generate_colab_notebooks.py`: 일반 파일 정본에서 노트북·source manifest 생성. 핵심 학습 로직은 노트북에 두지 않습니다.

실제 GPU FFT, real TFDS 전처리, real T1 cache, pretrained Qwen 학습, LIBERO(+) rollout은
이 로컬 검증에 포함되지 않습니다. 합성 CPU 검증을 해당 업무의 실험 완료로 대체하지 않습니다.
