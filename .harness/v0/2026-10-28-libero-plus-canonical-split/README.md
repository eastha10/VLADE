# LIBERO+ canonical split 연결 기록

## 목적과 입력

- 기준 commit: `dev/fish13031e`의 `64030f3f093eeb6486eddef559a1d97ee8bbdf68`
- 후보 manifest: `experiments/reports/libero_plus_split_analysis/manifests/D_80_10_10_seed_42.jsonl.gz`
- 사용자 확정 정책: D, exact action+state content-group, seed 42, 80/10/10
- dataset: `Sylvest/libero_plus_rlds@fb0c7029b076030d5d57227229e4f7460def1f7c`

## 수행 결과

- 후보 SHA-256 `487d2be1039ba6a4facc3ba44785ba5159245dfe53462a1d5a41ba8230446e80`을 재검증했습니다.
- 14,347개 고유 Episode, 1,681개 content group, Train/Validation/Test
  11,477/1,428/1,442, Train 내부 Calibration 728개와 split 간 group 누출 0건을 확인했습니다.
- 원본 1,024 TFRecord shard의 14,347 Episode와 2,238,036 transition을 다시 읽어
  canonical group membership을 전수 대조했습니다.
- 확정 Train 11,477 Episode, 1,791,269 transition만으로 normalization을 계산했고 SHA-256은
  `74b5c78ddce7f5d5fb00ccf91e1c34381fc7cece060110752c75fe4e7ca1abb8`입니다.
- AVA/OFT 공통 prepared-data 경계와 동일 canonical identity 검사를 구현했습니다.

## 검증과 제한

- 이 폴더에는 데이터셋이나 가중치를 복제하지 않고 정식 산출물 경로만 기록합니다.
- `local-llm analyze`는 이 환경에서 helper 접근 실패로 결과를 반환하지 못해, 관련 소스는 직접
  검토했습니다.
- canonical/FFT/Colab source 집중 검사는 최종 배치에서 46개와 18개가 각각 모두 통과했습니다.
- 전체 `tests/unit`은 121개 통과, 2개 실패, 10개 setup error였습니다. 실패/error는 모두 현재
  Python 3.14 환경에 `transformers` package metadata가 없어 기존 fixed-cache fixture가 시작 전에
  중단된 동일 원인이며, canonical 집중 검사에는 실패가 없습니다.
- Colab prepared dataset 생성, 실제 A100 FFT, checkpoint 저장·재로딩, VRAM·checkpoint 크기
  측정은 실행하지 않았습니다.
