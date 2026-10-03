# 0002: 주 실험 벤치로 LIBERO+와 Full Fine-Tuning을 사용

날짜: 2026-10-03
상태: 채택

VLADE의 주 비교 실험은 LIBERO+에서 수행하고 학습 방식은 Full Fine-Tuning(FFT)을
사용합니다. AVA-VLA와 OpenVLA-OFT의 원본 LIBERO 평가는 외부 baseline 재현과
인터페이스 검증을 위한 별도 트랙으로 유지합니다.

로컬 VLADE 폴더에는 공개 가중치와 필요한 LIBERO+ 자료를 내려받아 검증합니다.
추가 학습은 이 폴더를 사용자가 지정한 Google Drive 폴더에 업로드한 후
Google Colab A100에서 수행합니다. A100의 실제 VRAM 용량은 런타임 연결 후
확인해야 하며, 로컬에서 FFT를 수행한 것으로 기록하지 않습니다.

## 실험 구분

- `experiments/baselines/*/libero.json`: 저자가 공개한 원본 LIBERO checkpoint와 평가
  조건을 재현하기 위한 설정입니다.
- `configs/experiments/libero_plus_fft.json`: 동일한 LIBERO+ 데이터·평가 조건에서
  AVA-VLA, OpenVLA-OFT와 이후 VLADE Student를 비교하기 위한 주 실험 정의입니다.
- 원본 LIBERO 수치, LIBERO+ 논문 보고값, 자체 FFT 결과를 서로 섞거나 자체 결과로
  표기하지 않습니다.

LIBERO+ 공식 저장소가 제공하는 OpenVLA-OFT+는 mix-SFT 결과로 소개되어 있습니다.
VLADE에서 계획한 FFT 결과와 같은 실험으로 간주하지 않고 별도의 참고점으로만
취급합니다.

## 준비가 필요한 항목

현재 `benchmarks/libero_plus/` 어댑터와 공식 LIBERO+ 소스 revision은 아직 준비되지
않았습니다. 공개 학습 데이터는 `Sylvest/libero_plus_rlds`이며,
`Sylvest/LIBERO-plus`는 평가 환경의 assets 저장소입니다. 각 Hugging Face revision은
실험 설정에 고정했습니다. 실행 전에 다음을 준비해야 합니다.

1. 공식 LIBERO+ 저장소 commit과 라이선스
2. assets 및 RLDS 학습 데이터 파일 무결성과 압축 해제 후 디렉터리 구조
3. 두 baseline에 공통으로 적용할 optimizer, dtype, batch size, 학습 step과 seed
4. 일곱 perturbation 차원별 성공률과 macro average 계산 방식
5. GPU, VRAM, 학습 시간과 평가 지연시간 기록 방식

VLADE 측 FFT 학습 코드는 `src/vlade/training/fft.py`에 추가했습니다. 다만
의존성·데이터 형식과 실제 A100 학습이 검증되기 전에는 실행 가능한 실험이나
LIBERO+ 성능 결과로 표기하지 않습니다. AVA temporal zero-context 학습의
제한은 `docs/environment/libero_plus_fft_colab.md`에 명시합니다.

두 외부 원본의 `vla-scripts/finetune.py`는 현재 `use_lora=True`를 강제합니다.
VLADE의 별도 FFT 경로에서 공개 LIBERO 체크포인트를 초기 가중치로 사용합니다.
