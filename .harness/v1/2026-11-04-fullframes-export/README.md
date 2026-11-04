# 전체 프레임 원본 재처리 · 2026-11-04

사용자께서 원본 재처리·HF 게시를 요청했고 새 전체 프레임 데이터의 공개를 허용했습니다.
기존 비공개 dataset/model과 Drive 원본은 보존합니다. 유료 요금제는 구매하지 않습니다.

## 확인 및 구현

- 실제 프로젝트는 Desktop/vlade, Drive 대상은 내 드라이브/colab/VLADE입니다.
- Drive의 원본 dataset_info/features와 기존 canonical D splits/normalization을 읽었습니다.
  원본 TFDS 1,024 shards / 14,347 episodes / metadata 83,052,490,137 bytes이며
  보존 대상은 총 2,238,036 timestep입니다. Train 11,477 / Validation 1,428 / Test 1,442,
  calibration 728개와 중복 multiplicity를 유지합니다.
- 첫 실제 TFRecord Example 277 timestep을 읽어 canonical content SHA
  7eef44890709aadcf0cf9f4aed27dee377945aa0c36561aaac31adaa234d7540 일치를 확인했습니다.
  원본 payload 14,200,739 bytes에서 Parquet 9,612,837 bytes를 만들고 모든 이미지 bytes,
  action/state/joint/language/reward/discount/flags를 다시 읽어 일치를 확인했습니다.
  이것은 첫 에피소드 검증이며 전체 데이터 검증으로 확대 해석하지 않습니다.
- src/vlade/data/tfrecord_frames.py: TensorFlow 없이 CRC32C/스키마/내용 해시를 검사하고
  query stride 1/4/8와 action horizon 8을 독립적으로 선택합니다.
- src/vlade/data/fullframes.py와 scripts/data/prepare_fullframes.py:
  원본 JPEG를 유지하는 Parquet, 8-shard 단위 원격 원자적 게시·SHA 검증·재시작,
  전체 완료 seal 및 비공개 registry 등록을 구현했습니다.
- storage/assets.py에 parquet_frames의 작은 metadata descriptor만 검증·fetch하는 분기를 추가했습니다.
  기존 학습 루프·EpisodeData·Student/KD·기본 stride 8은 변경하지 않았습니다.
- utils/fullframes_notebook.py는 독립 CPU 내보내기 셀 하나를 Colab 01 위에 배치하고
  기존 이관·다운로드·전처리를 접습니다. 기존 outputs/metadata/사용자 셀을 보존합니다.
  Secrets HF_TOKEN을 그대로 사용하고 venv --without-pip + host pip --python으로 준비합니다.

새 공개 dataset VLADE-konkuk/libero-plus-fullframes에 안전한 reference 네 파일과
PREPARING 카드·export_config만 게시했습니다. 초기 revision은
21a81ee2c9cf120019ed18c5d1cd06bf4336c69e입니다. 완료 revision이 아닙니다.
reports/public-dataset-initialization.json에 이 초기화 결과를 보존합니다.
원본 공개 dataset의 MIT 라이선스를 API/카드에서 확인했고 출처를 카드에 보존했습니다.
기존 VLADE-checkpoints/private=True, libero-plus-prepared/private=True를 재확인했습니다.

## 실행한 검사와 범위

- 실제 첫 에피소드 변환·Parquet round trip 성공, canonical reference 14,347개 strict 검증 성공.
- test_fullframes + test_hf_assets + test_colab_source_files: 43 passed in 3.41s.
  CRC 오류·unknown schema·NaN 거부, 모든 프레임/필드 보존, 중복 분할/순서 보존,
  stride별 action chunk와 tail mask, 전송 중단 시 COMPLETE 미게시, 원격 완료 구간 재사용,
  metadata seal/tamper 검사, 노트북 사용자 셀/출력/metadata 보존을 검사했습니다.
- test_fullframes + 기존 test_colab_pipeline: 32 passed in 19.94s.
  환경 수정 후 전체 프레임 검사와 기존 split/정규화/history/cache/budget 데이터 계약 검사가 통과했습니다.
- 로컬 manifest 244개 파일의 SHA256 및 여섯 노트북 digest/compile을 확인했습니다.
  Drive는 기존 문서/코드가 HEAD와 같음을 확인한 뒤 정식 변경 20개만 반영했고
  다시 내려받아 20개 모두 bytes 일치, 코드 셀 compile, 환경별 manifest SHA를 검증했습니다.
  기존 사용자 셀·outputs·execution_count·metadata 보존도 비교했습니다.
  reports/drive-verification.json에 readback 결과를 보존합니다.
- Git 반영 전 현재 checkout이 main / 41780a6임을 확인했습니다. 이전 bf39b3e HF 이관과
  5f5ce57 협업 지침이 이미 main에 병합되어 있으며 기존 변경은 유지합니다.
  브랜치를 임의로 바꾸지 않고 현재 main을 기준으로 source manifest의 부모 revision을 맞춥니다.
- 검사 명령은 python -B -m pytest ... -p no:cacheprovider와 본 작업 scratch의 고유 basetemp를 사용했습니다.
  google-crc32c 1.7.1은 scratch/deps에만 설치하고 PYTHONPATH에 추가했습니다.
  한 번의 확장 검사에서 이 PYTHONPATH를 빠뜨려 CRC 의존성 검사 둘이 실패했고 실행 환경을 바로잡았습니다.

전체 83GB 원본 재처리·공개 Parquet 업로드는 아직 미실행입니다.
연결된 Drive 도구의 대용량 제한과 Colab 실행 버튼 접근 불가 때문에 원본 마운트가 있는
Colab에서 사용자가 새 단일 셀을 실행해야 합니다. 완료는 원격 COMPLETE/manifest/revision과
전체 counts/hash 검증으로 별도 확인합니다. scratch는 이 진행 작업의 검증/동기화에 사용하며
대용량 raw manifest·sample·재생성 가능한 검사 산출물은 필요가 끝나면 정리합니다.
