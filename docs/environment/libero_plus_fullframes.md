# LIBERO+ 원본 전체 프레임 재처리

> 2026-11-04 이후 로컬/Git 노트북의 실행 기준은 [Git 소스·HF 입력 안내](hf_inputs_and_git_source.md)입니다. 아래 Drive 기반 절차·실행 기록은 기존 버전의 맥락입니다. 진행 중인 세션은 그대로 유지하며 Drive 배포는 별도 명령 후 수행합니다.

## 저장소와 보존 범위

사용자 승인에 따라 **새 전체 프레임 데이터만** 공개 dataset
[`VLADE-konkuk/libero-plus-fullframes`](https://huggingface.co/datasets/VLADE-konkuk/libero-plus-fullframes)에 게시합니다.
기존 `libero-plus-prepared`와 `VLADE-checkpoints`는 비공개를 유지합니다.
Drive의 원본 압축본·해제된 TFDS·이전 전처리 데이터는 삭제하거나 변경하지 않습니다.

원본은 MIT 라이선스의 `Sylvest/libero_plus_rlds` revision
`fb0c7029b076030d5d57227229e4f7460def1f7c`입니다. 해제된 TFDS는 metadata 기준
83,052,490,137 bytes, 1,024개 TFRecord shard, 14,347개 에피소드입니다.
새 출력은 **2,238,036 timestep 전부**를 행 단위 Parquet으로 저장합니다.
압축·컨테이너가 달라 출력 파일 크기는 원본과 같을 필요가 없습니다.

| 분할 | 에피소드 | 전체 프레임 |
|---|---:|---:|
| Train | 11,477 | 1,791,269 |
| Validation | 1,428 | 223,485 |
| Test | 1,442 | 223,282 |

기존 canonical D/seed42의 분할 JSON과 Train 전용 정규화 JSON을 바이트 그대로 보존합니다.
원본 내용 해시로 기존 episode ID에 연결하며 동일 내용의 중복 에피소드도 제거하지 않습니다.
중복 그룹의 split/calibration 라벨을 유지하고 Train의 calibration 728개도 유지합니다.

두 카메라의 원래 256×256 JPEG bytes, raw action 7차원, state 8차원, joint state 7차원,
매 시점 언어·reward·discount·first/last/terminal·원래 공개 원본의 episode file metadata를 보존합니다.
이미지를 resize/re-encode하지 않습니다. 저장 중 gripper 변환이나 정규화를 적용하지 않습니다.
비공개 가중치·토큰·새 로컬/Drive 경로·기존 내부 manifest는 공개 데이터에 넣지 않습니다.

## Colab 실행과 재시작

Colab 01의 맨 위 **원본 전체 프레임 → 공개 Hugging Face** 셀에서
`RUN_FULLFRAME_EXPORT = True`로 바꾸고 실행합니다. 기존 Secrets `HF_TOKEN`을 사용합니다.
토큰 소유 계정에는 새 조직 dataset 생성/쓰기와 기존 비공개 registry 쓰기 권한이 필요합니다.
기존 이관·설치·다운로드·전처리 셀은 실행할 필요가 없으며 출력과 함께 접어 보존합니다.

TensorFlow와 GPU를 사용하지 않는 독립 CPU 환경을 만듭니다.
Python의 기본 pip bootstrap에 의존하지 않도록 `venv --without-pip` 후 host pip의
`--python` 설치를 사용합니다. 의존성은 HF Hub 0.36.2, PyArrow 19.0.0,
protobuf 5.29.3, google-crc32c 1.7.1, NumPy 2.2.6으로 고정합니다.
진행 로그는 셀과 runtime SSD의 `/content/vlade_fullframes_export/export.log`에 즉시 표시합니다.

8개 원본 shard씩 128구간으로 처리합니다. 한 구간만 SSD에 준비하고 split별 Parquet과
진행 receipt를 한 HF commit으로 게시합니다. 원격 크기/SHA256 검증 후 자신이 만든
구간 임시 Parquet만 지웁니다. Drive 원본은 읽기만 합니다.
TFRecord의 length/payload CRC32C, 원본 파일 SHA256, 에피소드 내용 해시·길이·분할,
Parquet 행 수를 검사합니다. 중단하면 **같은 셀을 다시 실행**하십시오.
원격 검증이 끝난 구간은 원본을 다시 읽지 않고 이어갑니다.
`[Errno 107] Transport endpoint is not connected`는 Drive 마운트 연결 끊김입니다.
기존 셀은 시작 시 실제 파일 읽기로 연결을 확인하며, 전송 중 이 오류가 나면
표준 `drive.mount(..., force_remount=True)`로 최대 두 번 재연결하고 같은 명령을 재개합니다.
다른 오류는 자동 재실행하지 않습니다. HF에 검증된 완료 구간은 유지합니다.
같은 dataset에는 exporter 하나만 실행하십시오. 다른 설정·코드의 export를 섞지 않습니다.

초기 reference commit은 `21a81ee2c9cf120019ed18c5d1cd06bf4336c69e`이며 **완료 데이터 revision이 아닙니다**.
모든 에피소드·프레임·128구간 검사가 통과해야 `dataset_manifest.json`과 `COMPLETE`를 게시하고
비공개 중앙 registry에 `fullframes_d_seed42`를 등록합니다.
완료 로그의 불변 revision과 원격 `COMPLETE`를 확인하기 전에는 학습에 사용하지 않습니다.
중간 게시 상태의 카드에는 PREPARING을 표시합니다. 저장 제한/전송 오류 시 완료로 처리하지 않습니다.

## 로컬 실행과 사용

표준 `hf auth login` 또는 외부 주입 `HF_TOKEN`으로 인증합니다. 토큰을 명령 인수에 넣지 않습니다.
위 의존성이 있는 CPU 환경에서 다음을 실행합니다.

```bash
python scripts/data/prepare_fullframes.py \
  --tfds-root data/processed/libero_plus_tfds \
  --work-root data/processed/fullframes-export-work \
  --repo-id VLADE-konkuk/libero-plus-fullframes \
  --reference-revision 21a81ee2c9cf120019ed18c5d1cd06bf4336c69e --public
```

완료 후 registry를 runtime 경로에 갱신하고 `hf_assets.py fetch fullframes_d_seed42`로
완료 seal·분할·정규화가 있는 작은 descriptor만 받습니다. 이 명령은 전체 이미지 shard를 받지 않습니다.
실제 프레임은 HF `datasets`의 Parquet streaming으로 읽을 수 있습니다.

```python
from datasets import load_dataset, Image
from vlade.data.tfrecord_frames import iter_action_queries

frames = load_dataset("VLADE-konkuk/libero-plus-fullframes",
    revision="<완료 로그의 불변 commit SHA>", split="train", streaming=True)
# 바이트를 유지하려면 자동 이미지 decode를 끕니다.
frames = frames.cast_column("image", Image(decode=False))
frames = frames.cast_column("wrist_image", Image(decode=False))
queries = iter_action_queries(frames, query_stride=1, action_horizon=8)
```

chunk를 만든 뒤 query를 shuffle하십시오. 에피소드 중간부터 시작하거나 프레임 순서를 먼저
섞으면 연속성 검사가 거부합니다. tail action의 padding mask를 손실 계산에 반영하고,
raw action에 기존 정규화와 gripper 규약을 적용해야 합니다.

기존 `EpisodeData`/FFT/Student/KD의 입력·학습 루프·stride 8 기본값은 이번 작업에서
바꾸지 않습니다. 새 Parquet을 이전 희소 데이터 형식으로 취급하지 않습니다.
stride 1 대 8 비교는 같은 초기 가중치·D 분할·정규화·batch·optimizer step으로 별도 실행하고,
Validation 및 rollout으로 선택한 뒤 Student/KD와 temporal cache를 연결합니다.
전체 프레임 저장 자체가 성능 향상을 입증한 것은 아닙니다.

Google Drive는 코드와 원본 보존, GitHub는 코드 버전 관리, Colab CPU는 재처리,
HF는 불변 데이터 정본, 학습 runtime SSD는 캐시로 사용합니다.
[HF 저장 정책](https://huggingface.co/docs/hub/storage-limits)의 공개 저장 공간은 best-effort이며
유료 요금제 구매나 기존 비공개 저장소의 공개 전환은 수행하지 않습니다.
