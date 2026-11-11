# Git 소스 · Hugging Face 입력 실행 방식

현재 실행본은 [통합 노트북](../../experiments/colab_vlade.ipynb)이며 `PROFILE`은 `prepare_inputs`입니다. 설치·실행 셀은 [연구 실행 안내](research_runs.md)를 따릅니다. 아래의 과거 번호·셀 이름은 이전 실행 기록과 구분하십시오.

2026-11-04 로컬/Git 개편 기준입니다. 기존 Drive 노트북이나 실행 중인 Colab은 자동 갱신하지 않습니다. Drive 배포는 사용자의 별도 명령을 받은 뒤 수행합니다.

## 소스 배포

새 Colab 노트북은 `https://github.com/eastha10/VLADE.git`의 `SOURCE_REF`(기본 `main`)를 받아 정확한 `SOURCE_COMMIT`으로 분리된 실행 사본을 만듭니다. AVA 소스도 그 commit의 submodule gitlink로 고정합니다. 이미 실행 중인 사본을 갱신하지 않으며, 수정된 캐시 사본은 거부합니다.

같은 실험을 재개할 때는 시작 셀 실행 전에 이전에 기록한 40자리 commit을 지정하십시오.

```python
SOURCE_REF = '이전_실행에서_기록한_40자리_Git_commit'
```

소스 변경은 먼저 commit·push해야 새 런타임에 전달됩니다. 미커밋 로컬 파일은 배포 대상이 아닙니다. `experiments/manifests/colab_source.json`은 배포 저장소와 기본 ref를 설명하는 schema 2 설정이며 파일별 SHA256 목록을 담지 않습니다.

## 가중치와 데이터

외부 입력은 `hfasset://키`로 지정합니다. `configs/storage/hf_assets.lock.json`의 불변 HF revision과 SHA256을 사용합니다. 학습 진입점은 임의의 Drive/로컬 입력을 거부하며, 동일 레지스트리에서 받은 검증된 HF 캐시 경로는 허용합니다.

```bash
python scripts/data/hf_assets.py fetch ava_t0
python scripts/data/hf_assets.py fetch prepared_d_seed42
python scripts/data/hf_assets.py fetch qwen_initial
python scripts/data/prepare_episode_data.py \
  --tfds-root hfasset://raw_rlds --output data/processed/new-prepared \
  --canonical-manifest experiments/manifests/libero_plus_rlds/canonical_d_seed42.jsonl.gz \
  --canonical-normalization experiments/manifests/libero_plus_rlds/normalization_train_d_seed42.json \
  --canonical-normalization-sha256 eac95bcc81b6182b90a2bd030096df6e3e408cb88923d0a1d57dc6c81e297c67
```

전처리와 정규화 통계 생성도 같은 입력 검증을 사용합니다. 로컬 경로는 정확한 검증된
HF 캐시 루트일 때만 허용하며, 임의의 TFDS 폴더나 캐시 내부 하위 폴더를 지정해 검증을
생략할 수 없습니다. 출력은 새로운 로컬 경로를 사용합니다.

- `ava_t0`, `ava_fft_step_1975`, `openvla_oft_t0`: 등록된 모델/체크포인트.
- `qwen_initial`, `openvla_pretrained`: 고정된 공식 HF 모델 revision.
- `raw_rlds`: 공식 HF 원본 분할 ZIP을 받아 검증·해제한 RLDS.
- `prepared_d_seed42`: 등록된 D/seed42 전처리 데이터.
- `teacher_cache_t1`, `suite_map_seed42` 등은 실제 게시·검증 후 레지스트리에 등록해야 합니다. 없는 키를 다른 데이터나 Drive 경로로 대체하지 않습니다.
- `fullframes_d_seed42`는 128구간·14,347에피소드·2,238,036시점 및 `COMPLETE` 검증을 마친 별도 입력입니다. dataset revision `604e5e64f141d9fb78c33cf47d47540fcfddce49`를 고정하며 기존 희소 prepared 입력을 자동 교체하지 않습니다.

새 노트북은 코드에 포함된 lock을 사용합니다. 실행 중 최신 registry로 자동 전환하지 않습니다. 새 자료를 사용할 때는 registry를 별도 경로에 명시적으로 갱신하고 검토한 고정 버전을 Git에 반영하십시오.

```bash
python scripts/data/hf_assets.py refresh-registry --output <별도-registry.json>
```

## 캐시와 현재 실행 출력

Colab은 `/content/vlade_hf_assets`에 입력을, `/content/vlade_runs`와 `/content/vlade_asset_work`에 현재 실행 출력을 둡니다. Git 소스 사본에 결과를 쓰지 않습니다. 현재 실행에서 생성한 checkpoint·Teacher 선택 파일은 로컬에서 이어 사용할 수 있으며, 다른 런타임에 넘길 입력은 HF에 게시한 뒤 고정 키를 사용합니다. 기존 학습 checkpoint의 HF 저장·재개 절차는 [기존 안내](hugging_face_checkpoints.md)를 따릅니다.

같은 프로세스에서 파일 크기·수정/생성 시각과 레지스트리가 그대로인 입력은 검증 결과를 재사용합니다. 변경 시 다시 해시를 확인하고, 새 프로세스와 명시적 `fetch`는 전체 검증을 수행합니다. 데이터 분할·정규화·checkpoint·Teacher 출처 검증은 유지합니다.

새 Colab 01은 HF 자료 준비와 선택적 전체 프레임 내보내기를 제공합니다. 새 전체 프레임 내보내기는 Drive 추출본 대신 `raw_rlds`를 사용합니다. 다운로드·전처리·GPU 학습은 노트북의 실행 스위치를 사용자가 켤 때 시작합니다.

## 검증 범위

이 개편은 로컬 CPU 회귀 검사와 로컬 Git/HF 모의 입력으로 검증합니다. 실제 GPU 학습, 대용량 다운로드, 실행 중인 Colab 변경, Drive 전송, HF 자료 게시를 수행했다는 의미가 아닙니다. 과거 Drive 방식으로 시작한 세션은 종료할 때까지 기존 노트북/소스 사본을 그대로 사용하십시오.
