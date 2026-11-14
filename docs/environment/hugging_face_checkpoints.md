# VLADE · 비공개 Hugging Face 체크포인트

현재 실행본은 [통합 노트북](../../experiments/colab_vlade.ipynb)이며 `PROFILE`은 `각 학습 프로필`입니다. 설치·실행 셀은 [연구 실행 안내](research_runs.md)를 따릅니다. 아래의 과거 번호·셀 이름은 이전 실행 기록과 구분하십시오.

## 적용 범위

기존 학습 루프·손실·누적·LR schedule·optimizer·초기 모델을 유지하고 완료 체크포인트만
Hub로 전송합니다. `VLADE_HUB_REPO`가 없으면 기존 로컬 동작입니다. FFT, Qwen Task-only,
Action KD, 04 논문 AVA와 Suite-balanced adapter에 연결합니다. 별도 Trainer로
교체하지 않습니다. 학습·GPU·모델 다운로드는 이 기능 설정만으로 실행되지 않습니다.

| 실행 | 저장 주기 | best 기준 | 재개 파일 |
|---|---|---|---|
| FFT / Suite-balanced | 기존 save_steps, 예산 종료, 목표 step 및 검증 개선 시 | val_l1 최소값 | 전체 모델·연속 action 구성요소·trainer_state.pt |
| Qwen Task-only / KD | 기존 save_steps, 목표 step 및 검증 개선 시 | validation_l1 최소값 | model/·processor/·trainer_state.pt |
| 04 논문 AVA | 기존 저장 시점 | 저장된 checkpoint 중 train MAE 최소값 | 고정 원본 모델 참조·LoRA·head/proprio/generator·state.pt |

04는 기존 루프에 held-out validation이 없어 best가 일반화 성능이나 rollout 성공률을 뜻하지
않습니다. 검증을 끈 FFT에는 best.json을 만들지 않습니다. 마지막 목표 optimizer step을
마쳤을 때만 final.json을 만듭니다. 예산 종료·실패·강제 중단은 final로 표시하지 않습니다.
기존 학습에서는 저장하지 않았던 검증 개선 step에 추가 저장이 발생할 수 있습니다.

## 인증 연결

1. [HF 새 모델 저장소](https://huggingface.co/new)에서 원하는 이름으로 **Private** 저장소를 만듭니다.
2. [Access Tokens](https://huggingface.co/settings/tokens)에서 해당 저장소 쓰기 권한을 가진
   fine-grained 토큰을 만듭니다. 원본 gated 모델 다운로드 권한은 별도입니다.
3. Colab 왼쪽 Secrets에 이름 `HF_TOKEN`으로 등록하고 해당 노트북의 접근을 허용합니다.
4. 01~05와 Suite-balanced의 `private-hub` 셀은 자동 전송이 기본 켜짐입니다. 새 실행은 출력된
   `HF_RUN_GROUP`을 기록하고 재개는 기존 group을 지정한 뒤 실행합니다. HF_REPO_ID는 사용자가 지정한 `VLADE-konkuk/VLADE-checkpoints`이며,
   예: `HF_RUN_GROUP='qwen-kd-seed42-r1'`입니다.
   새 실험은 새 group, 재개는 같은 group과 output 폴더명을 사용합니다. 파일마다 run group이
   다르면 충돌하지 않습니다. Suite-balanced는 두 출력이 모두 checkpoints라는 이름이므로
   group에 `-pilot4` 또는 `-main`을 붙여 구분합니다. 같은 group/output에는 프로세스 하나만 쓰도록 합니다.

01에 등록한 동일한 `HF_TOKEN` Secret을 사용하되, 02~05/Suite에서도 각 노트북의 Secret 접근을
허용해야 합니다. 토큰 소유 계정은 VLADE-konkuk 조직의 해당 저장소를 만들거나 쓸 권한이
필요합니다. 01에서 `private-hub` 설정 → `install-data` → `private-hub-check` 셀 순서로 실행하면
계정명·대상 저장소·private 여부만 표시해 연결을 확인합니다. 저장소가 없으면 비공개로 만들며
모델·데이터 다운로드/업로드나 GPU 학습은 시작하지 않습니다. 실제 commit 쓰기 권한은 첫
체크포인트 업로드에서 최종 확인됩니다.

전처리 데이터와 Teacher cache는 아래 별도 private Dataset 저장소로 관리합니다.
기존 Drive 자료의 실제 이관은 Colab 01 맨 위의 이관 셀에서 수행하며 원본을 삭제하지 않습니다.
이전 개인 저장소의 hub_state.json이 있는 checkpoint를 조직 대상으로 재개하면 식별자 검사에서
거부합니다. 그런 기존 실행은 저장소와 체크포인트 metadata를 일관되게 이관하는 별도 작업이 필요합니다.

Secrets 값은 환경 변수로 자식 학습 프로세스에 전달합니다. 토큰을 Python/JSON/notebook
문자열·Drive 파일·실험 manifest·로그에 적지 않습니다. `login()`을 호출해 Drive의 HF_HOME에
토큰을 저장할 필요가 없습니다. 노트북은 HF_TOKEN_PATH와 새 모델 캐시를 `/content`에 둡니다. 토큰을 채팅으로 전달하지 마십시오.

로컬은 해당 학습 환경의 `hf auth login` 대화형 입력 또는 외부 secret manager가 주입한
`HF_TOKEN`을 사용합니다. Windows PowerShell에서는 아래와 같이 토큰이 아닌 실행 위치만
지정한 다음 원래 학습 명령을 실행합니다.

```powershell
$env:VLADE_HUB_REPO = 'VLADE-konkuk/VLADE-checkpoints'
$env:VLADE_HUB_RUN_ID = 'ava-seed42-r1'
python scripts/train/hub_checkpoints.py check --output artifacts/training/ava-main
python scripts/train/run_fft.py --model ava_vla --config configs/experiments/libero_plus_fft.json
```

`check`는 인증·접근·비공개 여부를 확인하고 저장소가 없으면 비공개로 생성합니다. 대상 저장소가
이미 공개라면 비공개로 자동 변경하지 않고 업로드를 거부합니다. 실제 첫 commit 쓰기까지의
권한 검사는 최초 업로드에서 이루어집니다. Hub 활성화 시 FFT는 재개 가능한 training state
저장을 자동으로 켭니다. 기존 HF 0.36.2 기반 AVA 환경과 1.x 기반 Student 환경의 공통
`HfApi.create_commit` 인터페이스를 사용하며 Transformers/torch 버전은 변경하지 않습니다.

## 원격 구조와 장애 처리

```text
runs/<HF_RUN_GROUP>/<output 폴더명>/
  checkpoints/step-0000500/  # 봉인된 모델·optimizer/scheduler/RNG·step/cursor
  checkpoints/step-0001000/
  last.json                 # 마지막 업로드 완료 checkpoint
  best.json                 # 위 표의 기준으로 선택한 checkpoint
  final.json                # 목표 학습 완료 checkpoint
```

가중치와 pointer를 **한 commit**에 올립니다. 불완전한 업로드를 가리키는 pointer를 게시하지
않으며 파일 해시·COMPLETE·학습 상태 파일 유무를 확인합니다. best 선택 값은 봉인 안의
hub_state.json에 저장하므로 재개 뒤 나쁜 checkpoint로 best를 덮지 않습니다.

학습 중 전송이 실패하면 3회 시도 후 학습을 계속하고 output/hub_pending.json에 재전송할
로컬 경로를 기록합니다. 다음 checkpoint와 정상 종료 시 다시 시도하며 FIFO 순서를 유지합니다.
실패 메시지는 오류 종류만 남깁니다. 전송 중 host/PyTorch RNG를 보존하므로 업로드가 난수
기반 학습 결과를 바꾸지 않습니다. 동기식 전송 시간은 전체 세션 예산에 포함해야 합니다.
Suite adapter의 강제 예산 종료 checkpoint도 전송 대상입니다.

```bash
python scripts/train/hub_checkpoints.py retry --output artifacts/training/ava-main
```

pending 목록의 checkpoint를 업로드 전에 삭제하지 마십시오. `/content`만 사용하던 런타임이
강제로 사라지면 아직 업로드하지 못한 파일도 사라집니다. 마지막 원격 commit까지만 복구할 수
있습니다. 현재 구현은 원격·로컬 checkpoint를 자동 삭제하지 않습니다. 특히 FFT와 optimizer는
용량이 크므로 save_steps·검증 주기·HF 저장 한도를 확인하십시오. best/final pointer는 파일을
복제하지 않습니다. 엄격한 마감이 있으면 파일럿에서 저장+전송 시간을 함께 측정하십시오.

## Hub에서 받아 재개

```bash
python scripts/train/hub_checkpoints.py download \
  --repo VLADE-konkuk/VLADE-checkpoints --run ava-seed42-r1/ava-main \
  --alias last --output artifacts/resume/ava-main
# 출력된 step 폴더를 기존 --resume에 전달합니다.
python scripts/train/run_fft.py --model ava_vla --config 실행에-사용한-동일-설정.json \
  --resume artifacts/resume/ava-main/step-0001000
```

`--alias best`/`final` 및 `--revision <commit SHA>`도 지원합니다. 다운로드는 pointer와 모든
파일을 같은 불변 commit SHA에서 가져와 별도 폴더에 복사·무결성 검사한 뒤 게시하며 기존
경로를 덮어쓰지 않습니다. Qwen/04도 원래 학습 진입점의 `--resume`에 출력 폴더를 넘깁니다.
04는 LoRA delta 저장이므로 기존에 고정한 원본 모델이 별도로 필요합니다.

optimizer·scheduler·RNG와 step/cursor는 복원하지만 기존 재개 검증도 그대로 적용합니다.
설정·코드·라이브러리·데이터/정규화·초기 가중치가 달라지면 재개를 거부할 수 있습니다.
Qwen identity에는 output/initial_checkpoint 등의 경로도 포함되므로 새 Colab 세션에서도 동일한
HF 캐시·실행 경로를 사용하십시오. 기존 코드로 만든 checkpoint와 이번 코드의 fingerprint는
다르므로 자동 호환된다고 간주하지 않습니다. prepared query는 cursor를 복원하고 raw RLDS는
stream을 다시 시작합니다. GPU 비트 단위 재현을 보장하지 않습니다.

## HF 자료 정본과 실행 구조

코드는 GitHub와 Drive에, 데이터·가중치는 Hugging Face에 둡니다. Colab과 로컬은 같은
`hfasset://키`를 사용하고 고정 commit SHA의 자료를 로컬 SSD 캐시에 내려받아 학습합니다.
Drive 원본은 이관 확인 후에도 자동 삭제하지 않습니다. Drive 자체의 자동 중계가 아니라
Colab Python 프로세스가 Drive와 HF 양쪽에 접근하여 업로드합니다.

| 자료 | HF 위치 / 논리 키 |
|---|---|
| 전처리 D seed42 TAR | private dataset `VLADE-konkuk/libero-plus-prepared` / `prepared_d_seed42` |
| 초기 AVA·기존 FFT·새 학습 checkpoint | private model `VLADE-konkuk/VLADE-checkpoints` / `ava_t0`, `ava_fft_step_1975`, `runs/...` |
| 로컬 OFT 4-in-1 초기 가중치 | 위 model 저장소 / `openvla_oft_t0` |
| 전체 T1 Teacher cache | 생성 완료 후 private dataset `VLADE-konkuk/vlade-teacher-cache` / `teacher_cache_t1`, `teacher_cache_t1_hidden` |
| 공개 raw RLDS·Qwen·OpenVLA·OFT 개별 suite | 원본 HF 저장소의 고정 commit 유지. 개별 suite 모델을 4-in-1로 바꾸지 않음 |

자료 레지스트리는 model 저장소의 `assets/registry.json`에 있습니다. 로컬 배포 설정은
`configs/storage/hf_assets.lock.json`입니다. 각 자료는 40자리 불변 commit과 SHA256을
포함하며 완료·검증된 게시물만 등록합니다. Colab의 첫 HF 입력 준비가 최신 레지스트리를
런타임 로컬로 가져옵니다. Drive 소스나 소스 해시 목록은 실행 중 변경하지 않습니다.

```bash
python scripts/data/hf_assets.py refresh-registry
python scripts/data/hf_assets.py fetch prepared_d_seed42
python scripts/data/hf_assets.py fetch ava_t0
```

기본 로컬 캐시는 `data/hf_assets/`와 `checkpoints/hf_assets/`입니다.
`VLADE_ASSET_CACHE`로 공통 캐시 루트를 지정할 수 있습니다. Colab 입력 캐시는
`/content/vlade_hf_assets`, 전처리·cache 생성 작업은 `/content/vlade_asset_work`, 새 실험
출력은 `/content/vlade_runs`입니다. 실행 결과는 HF checkpoint에 봉인 후 전송합니다.
런타임 SSD 자료는 세션 종료 시 사라지므로 업로드 실패 목록과 마지막 성공 commit을
확인하십시오. 기존 Drive 경로로 저장한 과거 실험과 새 캐시 경로의 재개 identity는 자동
호환하지 않으며 기존 엄격한 검사를 유지합니다.

Colab 01 맨 위의 이관 셀 한 개만 `RUN_HF_MIGRATION=True`로 실행하면 기존 Drive의
전처리 TAR·현재 AVA 초기 가중치·봉인된 FFT 상태를 순서대로 검증하고 전송합니다.
01의 원래 다운로드·추출·전처리 셀과 출력은 접힌 구역에 보존합니다. 이관 셀은
기존 `HF_TOKEN` Secret을 읽으며 GPU·학습 환경을 설치하지 않습니다. Python에 ensurepip가
없어도 `venv --without-pip`과 호스트 `pip --python`으로 최소 HF 환경을 구성합니다.
연결이 끊기면 같은 셀을 재실행할 수 있으며 업로드 내용 중복은 HF 저장소에서 처리합니다.

기존 파일명의 `의 사본`은 checkpoint seal에 따라 업로드 이름에서만 복원합니다.
Drive 파일은 이름을 바꾸거나 삭제하지 않습니다. 기존 step-1975는 optimizer/scheduler/RNG를
포함하여 보관하되 이번 코드와 fingerprint가 다르면 기존 재개 검사에서 거부됩니다.
확인되지 않은 과거 checkpoint를 최고 성능으로 지정하지 않습니다.

Drive 마운트에서 복사본 완료 표시를 인식하지 못하면 기존 이관 셀이 파일 ID로 원본을
읽는 경로로 전환합니다. 이때 표준 Colab Google 인증이 한 번 필요할 수 있으며 Drive에는
읽기 요청만 보냅니다. HF_TOKEN을 Google 인증에 사용하지 않습니다. 체크포인트는
`/content/vlade_hf_transfer/checkpoint-source/`에 준비하고 원래 seal의 크기·SHA256과 완료
표시를 모두 확인한 뒤 업로드합니다. 다운로드 진행 바이트가 출력되며 완료된 파일과
중단된 부분 파일을 재사용합니다. 원본 Drive 이름이나 완료 표시는 수정하지 않습니다.

Suite-balanced notebook은 원래 audited `suite_map.json`이 전처리 자료 안에 있거나
`suite_map_seed42`로 별도 게시되어 있어야 합니다. 없으면 준비 단계에서 중단합니다.
레이블을 instruction 문장으로 추측하여 생성하지 않습니다. 전체 Teacher cache는 실제
생성 전에는 존재하지 않습니다. 02는 full cache 완료 후 HF에 게시하고 05는 그 자료를
참조합니다. pilot cache는 full cache 이름으로 게시하지 않습니다.

인증은 Colab Secrets `HF_TOKEN`의 해당 노트북 접근 허용과 로컬 `hf auth login`을 사용합니다.
토큰을 코드·Git·Drive·명령 인수에 넣지 않습니다. 토큰 계정은 조직의 해당 private 저장소에
읽기/쓰기 권한이 있어야 합니다. 각 Colab의 자동 checkpoint 전송은 기본 켜짐이며
출력된 `HF_RUN_GROUP`을 기록하고 새 런타임 재개 시 동일한 값을 지정하십시오.
로컬 학습은 위 환경 변수 `VLADE_HUB_REPO`, `VLADE_HUB_RUN_ID`로 전송을 켭니다.
Windows 연결이 TLS 초기화에서 끊기는 환경은 `VLADE_HF_TLS12=1`로 인증서 검증을
유지한 TLS1.2 호환 모드를 사용할 수 있습니다.

공식 근거: [HF 업로드/commit](https://huggingface.co/docs/huggingface_hub/guides/upload),
[토큰과 권한](https://huggingface.co/docs/hub/security-tokens),
[인증 환경 변수](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables),
[pip 다른 Python 관리](https://pip.pypa.io/en/stable/topics/python-option/),
[Colab Drive 연결](https://colab.research.google.com/notebooks/io.ipynb),
[Colab I/O·런타임 제한](https://research.google.com/colaboratory/faq.html).
