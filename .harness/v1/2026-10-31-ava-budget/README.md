# AVA FFT · A100 8시간 예산 적용

2026-10-31, 한국 시간. 사용자가 AVA-VLA 논문을 기반으로 실험 비용 최적화를 요청하고
추가 질문에 A100 8시간 이내를 지정했습니다. 실제 GPU 학습/데이터 다운로드/가중치 변경은 실행하지 않았습니다.

## 변경

- `src/vlade/training/compute_budget.py`: 4-step 실측(첫 step 제외), 1 epoch/시간 상한의 최소값,
  저장·validation·startup 반영, 후속 1시간/종료 최소 5분/30% 여유, optimizer 경계 중단 판단.
- `src/vlade/training/fft.py`, `run_control.py`: 실제 step/저장/validation/startup 시간 기록,
  마감 전 저장 후 `budget_exhausted` 기록, 코드 fingerprint에 예산 모듈 포함.
- `configs/experiments/ava_fft_a100_8h.json`: FFT 유지, LR 2e-5, batch 1×accumulation 8,
  warm-up 10%, cosine, clipping 1.0, validation 16, 중간/최종 약 2회 저장.
- `src/vlade/utils/colab_notebooks.py`, `experiments/colab_01~03`, `experiments/manifests/colab_source.json`:
  세션 마감 보존, 파일럿/본 학습 분리, 측정 GPU 일치 검사, 재개 시 목표 고정,
  subprocess 마감 및 Colab runtime 반환 요청, cache 시간 추정/예산 거부.
- README, FFT 안내, 인계 문서 및 `docs/decisions/0004-ava-a100-budget.md`에 논문과 차이/미검증 범위를 기록.
- `tests/unit/test_compute_budget.py`: 예산 계산/부족 거부/파일럿 요구/실제 CPU accumulation 경계/
  FFT 예산 종료 상태/노트북 구문 검사. 기존 pipeline test에는 최신 환경 검증 콜백 mock을 추가.
- Drive의 최신 managed Transformers 설치 패치와 schema 2 데이터 분할 호환 코드를 먼저 비교/보존했습니다.
  로컬에 없던 `configs/models/teacher/transformers-avavla-a03eee5-checkpoint-extra-attn.patch`와
  `src/vlade/data/episode_data.py` 변경은 기존 Drive 정본을 반영한 것입니다. 외부 submodule은 수정하지 않았습니다.

## 실제 검증

Python 3.13 임시 CPU 테스트 환경에서 다음 검사를 실행했습니다.

```text
python -B -m pytest tests/unit/test_compute_budget.py tests/unit/test_training_loop.py tests/unit/test_fft.py tests/unit/test_colab_source_files.py tests/unit/test_colab_pipeline.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-10-31-ava-budget/scratch/pytest-budget-03
71 passed in 23.66s
python -B scripts/prepare/generate_colab_notebooks.py
git diff --check --ignore-submodules=all
```

초기 검사는 테스트 환경의 Transformers 메타데이터 누락과 기존 notebook 환경 callback mock 누락으로
3 failed/68 passed였습니다. 임시 환경에 해당 패키지를 설치하고 mock을 보완한 후 위 71개가 통과했습니다.
CPU 환경(torch 2.15.0+cpu)은 실제 Colab/AVA fork 환경 재현이 아니며 GPU VRAM·속도·성공률을 검증하지 않습니다.

## Drive 반영

승인된 `내 드라이브/colab/VLADE`의 15개 정식 파일만 추가/갱신했습니다. 기존 파일은 같은 ID로 갱신하고,
신규 4개는 대상 폴더에 동명이 없음 확인 후 생성했습니다. 읽기 회수로 15개 SHA256 일치와 노트북 manifest
연결을 검증했습니다. `drive-verification.json`에 경량 결과를 기록했습니다. 기존 데이터/가중치/결과는 건드리지 않았습니다.

Drive 01의 사용자 추가 셀 `f_DeAKUMVGV_`와 출력이 있는 5개 셀을 보존했습니다.
Drive 01/03은 기존 셀을 유지하고 bootstrap manifest 식별자만 갱신했습니다.
Drive manifest는 미변경 Drive 소스의 기존 hash/revision을 유지한 채 이번 정식 소스만 갱신했습니다.
따라서 로컬/Drive의 전체 manifest 및 노트북 파일 bytes는 사용자 추가 셀/기존 소스 bytes에 따라 다르며,
각 환경의 source identity를 따로 검증합니다. 동일한 변경 코드/설정의 내용은 일치합니다.

최초 신규 업로드 2회는 자동 승인 검토가 명시적 승인 및 중복 파일 여부를 이유로 거부했습니다.
AGENTS.md의 지속 승인/추가 파일 규칙과 폴더 부모 경로, 동명 파일 부재를 확인한 뒤 승인되어 반영 완료했습니다.
Git commit/push는 요청받지 않아 수행하지 않았습니다. 검증용 임시 패키지/비교 파일은 작업 scratch에서 정리합니다.

## 제한

논문 v4의 LoRA rank 32/K=4 TBPTT/attention regularizer/40k steps 재현이 아닙니다.
현재 FFT는 temporal zero-context입니다. max_steps·VRAM·수렴·T1 성공률·실제 CU 비용은 GPU 실측 전입니다.
Colab 서비스 장애 시 자동 runtime 반환과 정확한 비용 상한까지 보장하지 않습니다.
