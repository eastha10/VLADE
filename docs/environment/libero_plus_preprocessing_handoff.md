# LIBERO+ 전처리 코드와 AVA 실행 노트북

현재 실행본은 [통합 노트북](../../experiments/colab_vlade.ipynb)이며 `PROFILE`은 `ava_suite / prepare_one_episode / ava_one_step`입니다. 설치·실행 셀은 [연구 실행 안내](research_runs.md)를 따릅니다. 아래의 과거 번호·셀 이름은 이전 실행 기록과 구분하십시오.

## 전처리 진입점

`scripts/data/prepare_episode_data.py`가 `vlade.data.episode_data.main()`을 호출합니다.
`src/vlade/data/canonical_split.py`는 기존 Canonical D/seed42 manifest를 검증하고
episode별 split을 배정합니다. `canonical_normalization.py`는 별도의 통계 생성
모듈이며, 전체 전처리 실행에서는 이미 생성된 정규화 JSON을 입력으로 받습니다.

2026-10-30 CPU 전처리는 `--resume --min-free-gib 20`과 기존 로컬 TFDS,
canonical manifest, 외부 정규화 JSON 및 각 SHA256을 지정해 실행했습니다.
결과는 Train 11,477 / Validation 1,428 / Test 1,442 episode였습니다.
위 내용은 당시 실행 기록입니다. 현재 `_preparation_source()`는 먼저 공통 HF 출처·무결성
검증을 수행합니다. 전처리와 정규화 생성 모두 `hfasset://raw_rlds` 또는 같은 고정
레지스트리에서 받은 정확한 HF 캐시 루트만 입력으로 받습니다. 미등록 로컬 TFDS나
Drive 경로는 TFDS 읽기·출력 생성 전에 거부합니다. 기존 단일 Train episode 준비
API와 loader 검증은 유지합니다.

외부 정규화 원본과 저장소에 있는 정규화 JSON은 바이트 SHA가 다를 수 있습니다.
실제로 선택한 파일의 SHA를 지정해야 하며 과거 입력의 SHA와 혼동하지 않습니다.
기존 데이터와 완료된 출력은 덮어쓰지 않습니다.

기본 FFT 설정은 Git에 보관된 정규화 JSON의 바이트 SHA를 검증합니다. 기존 고정 HF
prepared 자료가 원본 정규화 blob의 다른 SHA를 보존한 경우에는 action/proprio 통계값의
완전한 일치와 해당 HF 자료의 고정 revision·payload 무결성을 모두 확인합니다. 원본
prepared identity는 checkpoint와 재개 계약에 그대로 남으며 새 값으로 덮어쓰지 않습니다.

```bash
python -B scripts/data/prepare_episode_data.py \
  --tfds-root hfasset://raw_rlds \
  --output /path/to/new-prepared-output \
  --canonical-manifest experiments/manifests/libero_plus_rlds/canonical_d_seed42.jsonl.gz \
  --canonical-normalization experiments/manifests/libero_plus_rlds/normalization_train_d_seed42.json \
  --canonical-normalization-sha256 eac95bcc81b6182b90a2bd030096df6e3e408cb88923d0a1d57dc6c81e297c67 \
  --min-free-gib 20
```

정규화 통계를 재계산할 때도 동일한 검증된 HF 원본을 사용합니다. 아래 명령은 전체
원본을 읽는 작업이므로 실제 재계산이 필요한 승인된 실행에서만 사용합니다.

```bash
python -B scripts/data/compute_libero_plus_normalization.py \
  --data-dir hfasset://raw_rlds \
  --canonical-manifest experiments/manifests/libero_plus_rlds/canonical_d_seed42.jsonl.gz \
  --output /path/to/new-normalization.json
```

전체 action/proprio trajectory를 NPZ로 저장하고, 카메라 이미지와 지시문은
t=0,8,16,... query에 연결합니다. 이미지는 RLDS에 저장된 방향을 유지합니다.
Train 전체 transition의 정규화 통계를 사용하며, loader가 query별 8×7 action
chunk와 tail mask를 구성합니다. 재개할 때만 같은 입력·출력 계약으로 `--resume`을
추가합니다. 데이터·가중치·전처리 결과는 이번 Git 변경에 포함하지 않습니다.

## AVA FFT·Suite-balanced의 연속 시퀀스

전처리본은 에피소드 안의 시간 순서와 8개 행동 묶음, tail mask를 보존합니다.
2026-11-11부터 새 AVA 실행은 Episode와 시작점을 무작위로 선택해 연속 query
4개(t,t+8,t+16,t+24)를 묶습니다. 미니배치는 시퀀스 단위이며 내부 순서는 고정합니다.
Suite-balanced는 시퀀스 단위 suite 균형과 가능한 경우 update 내 다른 궤적 group 우선을
적용합니다. Validation은 별도 Episode의 결정적 구간만 사용합니다.
훈련 상태는 각 시퀀스 시작에 초기화하며 두 query마다 gradient를 끊습니다.
추론·Teacher cache 상태는 같은 Episode 안에서 전달하고 경계에서 초기화합니다.
4 query 미만 Episode는 제외하고 수를 기록합니다. [설정·차이·검증 범위](ava_sequence_training.md)를 확인하십시오.

재개 시 seed와 완료 step으로 동일한 시퀀스 위치를 재현합니다. 이전 개별 Query
순서의 체크포인트는 당시 고정 소스로 재개해야 하며 K4의 재개 입력으로 사용하지 않습니다.

## 첨부 노트북 보관

- `notebooks/references/colab_ava_fft_suite_balanced_supplied.ipynb`: 사용자가 전달한
  `colab_ava_fft_suite_balanced_cu_(1) (4).ipynb`의 코드·설명 셀입니다.
- `notebooks/references/colab_ava_libero_plus_paired_rollout_hf.ipynb`: 사용자가 전달한
  `colab_ava_libero_plus_paired_rollout_hf_ipynb_json.ipynb`의 코드·설명 셀입니다.

실행 출력·실행 횟수·출력 관련 metadata는 제거했습니다. 코드 셀의 경로, 실행 플래그,
HF 저장소, 내장 소스와 resume 식별자는 첨부본 그대로이며 이 Git 브랜치에 맞춰
자동 변경하지 않았습니다. FFT 첨부본은 `RUN_MAIN_FFT=True`, paired rollout은
`RUN_EVALUATION=False`입니다. 실행 전 사용할 입력과 실행 설정을 확인하십시오.
이 반영 작업에서 Colab 학습·GPU·rollout을 실행하지 않았습니다.

최신 실행 노트북은 `experiments/colab_vlade.ipynb`입니다. 전달본은 참조용으로 보존하며 최신 Git·HF 입력 계약을 사용하는 실행본을 덮어쓰지 않습니다. 전달본에는 과거 저장소·Drive 경로·실행 설정이 있으므로 그대로 새 런타임의 기본 입력으로 사용하지 마십시오. 새 OFT 및 단일 Episode AVA 점검 노트북은 Git 소스와 검증된 HF 입력을 사용합니다. 단일 Episode 출력은 HF 원본 출처와 canonical identity를 검사한 현재 실행 출력으로만 학습에 연결합니다.
