# Drive 압축 해제 데이터 경로 정리

- 날짜: 2026-10-29 (한국 표준시).
- 사용자 요청에 따라 `colab/VLADE/data/raw/libero_plus_rlds/libero_plus_extracted`를 `colab/VLADE/data/processed/libero_plus_tfds`로 이동·이름 변경했습니다.
- 이동한 폴더 ID는 `1o92_i1SHyWOrRweRYfoMP6aCjISCgvG5`로 유지했습니다. 파일 복제·삭제·재다운로드 없이 부모 폴더 메타데이터와 폴더 이름만 변경했고, 압축 내부의 `inspire/.../libero_mix/1.0.0` 계층은 보존했습니다.
- `raw/libero_plus_rlds`의 `.z01`, `.z02`, `.zip`, `download_manifest.json`, 다운로드 캐시는 그대로 유지했습니다. `checkpoints`·`hf_cache`는 변경하지 않았습니다.
- dataset_info.json에 기록된 TFDS는 `libero_mix` 버전 `1.0.0`, train shard 수는 1024입니다. API에서 읽은 직접 자식 1000개(샤드 997개와 JSON 3개)의 ID·이름·크기가 이동 전후 동일함을 확인했습니다. 이 조회는 전체 파일 hash·CRC 검증이 아닙니다.
- Drive의 `experiments/colab_01_download_data_weights.ipynb`에서 사용자 추가 압축 해제 셀의 출력 경로를 새 위치로 바꾸고, 기존 폴더가 있으면 재해제·덮어쓰기하지 않는 가드를 추가했습니다. 셀 9개와 기존 실행 결과를 유지했으며 업로드 후 내용 일치를 확인했습니다.
- 수동 압축 해제본 루트에 `COMPLETE`와 `extraction_manifest.json`이 없음을 확인했습니다. 이 정리를 데이터 준비 완료 검증으로 취급하지 않으며 완료 표시를 만들지 않았습니다. 기존 prepare_episode_data 실행에는 extraction manifest 준비·검증이 추가로 필요합니다.
- 학습·전처리·전체 데이터 다운로드·Git commit/push는 실행하지 않았습니다. 기존 로컬 STRUCTURE.md와 관련 생성물 변경은 유지했습니다.
