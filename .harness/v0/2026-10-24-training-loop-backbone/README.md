# Training loop 및 backbone 검토

기준: Desktop/vlade HEAD `1aa11fb`, 2026-10-24 한국 시간.
참조 대화 `6adc9137-5950-83ee-b02f-322423d0e7c3`에는 사용자 요청 한 건만
있어 실제 설계와 조건은 현재 저장소 README·설정·docs·고정 upstream에서 확인했습니다.

## 수행

- 기존 FFT 위에 모델과 독립적인 optimizer loop를 연결했습니다.
- accumulation, clipping·NaN 검사, scheduler, 선택적 validation, JSONL 로그,
  checkpoint 게시·재개와 실행 상태 기록을 보완했습니다.
- 모델·정규화·action chunk·upstream 소스를 유지했습니다.
- 공식 모델 카드/실제 config와 논문에 근거한 후보 비교를
  `docs/decisions/0003-backbone-review.md`에 기록했습니다.
- 실행·검증 안내는 `docs/environment/libero_plus_fft_colab.md`에 반영했습니다.

## 검증 결과

환경: Windows, Anaconda Python 3.13.5, Torch 2.12.0+cpu, pytest 8.3.4.

- 실제 저장소 `python -m pytest tests/unit -q -p no:cacheprovider`: 31 passed.
- `scripts/train/smoke_loop.py`: 합성 CPU 정책 4 optimizer steps, 8 microsteps.
  고정 데이터에서 step 2 저장·load·재개가 연속 학습과 정확히 일치했습니다.
- src/scripts/tests Python 11개 파일 compile 검사 통과.
- FFT CLI --help 확인.
- AVA/OFT 각각 --preflight: `data/processed/libero_plus_rlds` 부재로 실패.
  모델 학습 성공이나 전체 파일 준비 완료로 기록하지 않았습니다.
- GPU FFT, RLDS 실제 batch, bitsandbytes, pretrained save/load와 benchmark rollout,
  Student/KD, Jetson/TensorRT는 실행하지 않았습니다.

고정 데이터의 CPU exact resume은 shuffled RLDS iterator의 exact resume을 의미하지
않습니다. TensorFlow iterator/augmentation RNG 상태는 복원하지 않습니다.
기존 `.harness/v0/2026-10-03-basic-inference-smoke/README.md` 사용자 변경을 보존했으며
commit/push/브랜치·submodule 변경과 대용량 다운로드를 하지 않았습니다.
