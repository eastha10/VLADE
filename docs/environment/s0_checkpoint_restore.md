# S0-400 체크포인트 복원

[S0-400 manifest](../../experiments/manifests/s0_step400.json)에 optimizer step 400의 백업 분할 순서와 크기가 있습니다. 비공개 Google Drive 백업의 `step-0000400.tar.part-00`부터 `part-10`까지 총 11개에 해당합니다. 총 용량은 11,381,114,880바이트입니다.

백업 소유자가 Colab에서 원본과 Drive 복사본의 SHA-256 일치를 확인했습니다. 비교에 사용한 해시 값은 제공되지 않았습니다. 전체 tar의 신뢰할 수 있는 SHA-256을 별도로 확보했다면 복원 명령에 `--expected-sha256`으로 전달할 수 있습니다.

## 파일 복원

백업 소유자에게 접근 권한을 받은 계정으로 분할 파일을 준비합니다. 소유자는 Colab에서 Drive를 마운트한 뒤 `MyDrive/VLADE/S0-step400/`을 `PARTS_DIR`로 지정할 수 있습니다. `RESTORE_DIR`은 로컬 SSD의 출력 부모 디렉터리입니다. 분할 파일이 있는 경로 밖을 지정하고, 임시 tar와 추출본에 사용할 여유 공간을 확보합니다(이 백업은 약 22 GiB 이상).

저장소 루트에서 tar 추출 필터를 지원하는 Python 3.12 이상과 프로젝트의 NumPy·Pillow가 있는 환경으로 실행합니다.

```bash
python scripts/train/restore_s0_checkpoint.py \
  --parts-dir "$PARTS_DIR" \
  --output-dir "$RESTORE_DIR"
```

복원 스크립트는 분할 파일 11개의 바이트 수를 확인하고 번호순으로 연결한 뒤 tar의 SHA-256을 출력합니다. 추출 후에는 VLADE의 체크포인트 검사기로 `COMPLETE`, 내부 manifest의 파일 목록·크기·SHA-256, optimizer step 400, `trainer_state.pt`를 확인합니다. 성공하면 `$RESTORE_DIR/step-0000400`에 체크포인트가 생깁니다.

## 모델 로딩과 재개

복원한 디렉터리를 해당 실행을 생성한 Git revision과 설정의 체크포인트 경로로 넘깁니다. 전체 tar 해시나 분할 파일 경로를 모델 로더에 전달할 필요는 없습니다. 원래 실행 설정과 데이터 revision은 이 manifest에 포함되어 있지 않으므로 실행 기록에서 가져옵니다.

- FFT 체크포인트는 [FFT 로딩 검사](../../scripts/train/check_fft_checkpoint.py)의 `--checkpoint`에 지정합니다. 실제 query 검사에는 원래 데이터와 설정, 보고서 경로, CUDA 환경이 필요합니다. 학습 재개에는 [FFT 학습 진입점](../../scripts/train/run_fft.py)의 `--resume`을 사용합니다.
- Student task 체크포인트는 [Student 로딩 검사](../../scripts/train/check_student_checkpoint.py)의 `--checkpoint`에 지정합니다. 학습 재개에는 [Student 학습 진입점](../../scripts/train/run_student_task.py)의 `--resume`을 사용합니다.

어느 경로를 사용할지는 원래 실행 기록과 복원된 체크포인트의 메타데이터에서 확인합니다. 기존 로더가 적용하는 데이터·설정·코드 식별자 검사는 그대로 적용됩니다. 복원은 파일 무결성을 확인하며, 원본 모델의 추론과 학습 재개는 해당 실행 환경에서 검증합니다.

팀 공유가 필요하면 [비공개 Hub 체크포인트 절차](hugging_face_checkpoints.md)를 따릅니다.
