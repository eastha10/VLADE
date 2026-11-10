# AVA 연속 시퀀스 학습

2026-11-11 KST. 시작 main `cac96fe1de3eee1bba47792f2c0ca1fdcc961bd0`.
사용자가 AVA 데이터 순서 변경과 main push, 황인호 칸반의 11월 11일 확인사항 기록을 요청했습니다.

## 변경

- `src/vlade/data/sequences.py`: Train Episode·시작점 균등 복원 추출, 연속 K4, 전체 구간 shuffle 대체 설정, suite 단위 균형, sequence cursor 재현.
- `src/vlade/training/prepared_sequences.py`: 기존 HF 검증 EpisodeData·변환·tail mask를 사용한 시퀀스 loader. Validation 분리와 짧은 Episode 제외 수 기록.
- `src/vlade/training/temporal.py`: 시퀀스 시작 상태 초기화, 이전 action hidden 전달, detach2·segment별 backward, query/sequence/누적 평균.
- `fft.py`, `loop.py`, `paper_ava.py`, prepared loader·Suite wrapper·research dispatcher 연결. 기존 optimizer·schedule·Hub·checkpoint 봉인 유지, K1→K4 및 다른 temporal 설정 재개 거부.
- `configs/research/ava_{fft,suite,paper}.json`과 실험 설정에서 K4 기본값 지정. `ava_one_step` 진단과 OFT는 K1 유지.
- `docs/environment/ava_sequence_training.md` 및 README/STRUCTURE·관련 안내 갱신. 기존 전처리 데이터·가중치·HF lock·submodule pin 변경 없음.

[AVA 논문 §3.4](https://arxiv.org/html/2511.18960v4#S3.SS4)는 훈련 시 **시퀀스 시작**, 추론 시 **Episode 시작** zero state를 명시합니다.
첨부 표의 상태 행을 훈련 전체 Episode에 그대로 적용하지 않았습니다. 기존 이미지 query 간격8을 유지하므로 K4는 t,t+8,t+16,t+24입니다.

## 실제 검증

Python 3.13.5 / Torch 2.12 CPU / Transformers 5.9, 실제 7B 가중치와 GPU 없이 실행했습니다.
첫 축소 AVA 검사는 FP32 fixture와 BF16 입력의 불일치로 실패했습니다. 운영 로더와 동일하게 fixture를 BF16으로 맞춘 뒤 통과했습니다.

다음 명령은 `PYTHONDONTWRITEBYTECODE=1`, HF offline, `PYTHONPATH=src`와 scratch의 임시 CRC 의존성을 사용했습니다. TEMP/TMP/HF_HOME/TORCH_HOME도 scratch로 지정했습니다.

```powershell
python -B -m pytest .harness/v1/2026-11-11-ava-teacher-code/checks .harness/v1/2026-11-11-main-branch-integration/checks .harness/v1/2026-11-11-sequence-training/checks -q --tb=short -p no:cacheprovider --basetemp=.harness/v1/2026-11-11-sequence-training/scratch/full1
```

결과: **313 passed**, 276.74초. SWIG 타입 관련 DeprecationWarning 2개. 기존 HF 입력 거부/수락·native Llama/Qwen·회귀 검사와 새 시퀀스 검사 9개를 포함합니다.

이후 추가한 `test_existing_sealed_k1_checkpoint_cannot_resume_as_k4`를 같은 pytest 옵션과 고유 basetemp `scratch/resume1`으로 실행: **1 passed**, 8.69초. 합계 314개 검사 통과.
실제 축소 AVA 모델의 recurrent 경로 gradient, 구간 내부 시간 순서, split/suite 균형, tail mask, loader cursor, 분할 backward 및 optimizer 저장/재개를 확인했습니다.

- 변경 Python 10개를 메모리 compile, 관련 모듈 6개 import, JSON 7개 parse: 통과. pyc 생성 없음.
- 관련 문서 로컬 링크 52개: 누락 없음. `git diff --check`: 통과.
- `git fetch origin main`: 작업 시작 HEAD와 원격 main 동일. submodule pin AVA `69ae5df...`, OFT `e4287e9...` 유지.
- 임시 CRC 패키지 1.9.0은 scratch에만 설치했습니다. 첫 제한 네트워크 시도는 DNS 실패, 허용된 설치는 성공했습니다. 전역 환경 변경 없음.

미실행: 실제 7B BF16 GPU 훈련·Colab 설치·성능/rollout 및 전체 HF 자료의 짧은 Episode 수 측정. 공식 데이터·학습 결과 생성과 Drive 배포를 수행하지 않았습니다.
기존 미추적 하네스 기록 25개는 보존하고 이번 변경 목록에 포함하지 않습니다.

## 외부 반영과 정리

- 코드 commit `26a58671fa6f6c258273e9386e8b83d3011c83c7`을 `origin/main`에 push했습니다. `git ls-remote origin refs/heads/main`과 로컬 HEAD 일치를 확인했습니다.
- [황인호 칸반 · 11월 11일 확인사항](https://app.notion.com/p/414b5dd0687081d19a60ec338324a7ce)을 생성했습니다. 담당자 황인호, 날짜 2026-11-11, Type Task로 저장되고 해당 개인 칸반 view에 나타나는 것을 조회했습니다. CPU 검증과 main 반영은 완료 사실로, GPU 파일럿·Hub 재개·rollout은 미실행 확인사항으로 기록하여 Status는 Not started입니다.
- 이번 scratch의 절대 경계를 확인하고 테스트 생성물·임시 CRC 의존성·캐시를 모두 삭제했습니다. 현재 작업 보존 파일은 3개이며 합계 20 KiB 미만입니다. 기존 미추적 기록과 submodule은 보존했습니다.
