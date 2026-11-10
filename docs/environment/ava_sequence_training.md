# AVA 연속 시퀀스 학습

2026-11-11부터 새 AVA FFT·Suite-balanced 실행은 같은 Episode의 연속 policy query
4개를 묶어 학습합니다. 기존 HF 전처리본, canonical D/content-group split과 Train 전용
정규화 통계를 그대로 사용합니다. 데이터 재분할·재게시가 필요한 변경은 아닙니다.

## 입력 순서와 상태

| 항목 | 현재 AVA 훈련 경로 |
|---|---|
| Episode 선택 | 길이가 충분한 Train Episode를 균등 무작위 선택, 복원 추출 |
| 구간 선택 | 해당 Episode의 가능한 시작점 중 균등 선택 |
| 시퀀스 내부 | 4개 연속 query, 기존 간격 8을 보존하여 t,t+8,t+16,t+24 |
| 미니배치 | 독립 시퀀스를 묶음; 시퀀스 내부 프레임 순서는 고정 |
| 상태 전달 | 시퀀스 시작 zero state, 내부에서 이전 action hidden layer -2 전달 |
| gradient | 두 query마다 backward 후 context 값은 유지하고 그래프만 detach |
| 검증 | Validation Episode의 결정적 연속 구간만 사용, Test 제외 |

[AVA 논문 §3.4](https://arxiv.org/html/2511.18960v4#S3.SS4)는 **훈련 시 각 시퀀스
시작**, **추론 시 각 Episode 시작**에 상태를 0으로 초기화합니다. 무작위로 섞인
학습 구간 사이에 이전 구간의 상태를 전달하지 않습니다. 추론·순차 Teacher cache는
같은 Episode 안에서 상태를 이어가고 Episode가 바뀌면 초기화합니다.

여기서 4-step은 원본의 인접한 raw timestep 4개가 아니라 기존 이미지가 준비된 연속
policy query 4개입니다. 각 query는 여전히 8×7 action chunk와 실제 tail mask를 갖습니다.
4 query 미만 Episode는 패딩 없이 제외하며 적격/제외 수는 run manifest와 checkpoint의
`sequence_sampling_statistics`에 기록합니다. 적격 Train 또는 필요한 Validation이 없으면
큰 모델을 읽기 전에 실패합니다. 전체 HF 데이터의 실제 제외 수는 이번 CPU 검사로 측정하지 않았습니다.

Suite-balanced는 각 suite에서 Episode·시작점을 선택하고 네 시퀀스마다 suite를 하나씩
배정합니다. 같은 optimizer update 안에서는 가능한 경우 다른 trajectory group을 우선합니다.
Validation에는 suite 재표집을 적용하지 않습니다. 원본 분할의 동일 trajectory group이
Train/Validation을 가로지르지 않도록 하는 기존 canonical/HF 검증은 유지합니다.

## 설정과 실행

`configs/research/ava_fft.json`, `ava_suite.json`, `ava_paper.json`을 사용합니다.
현재 Colab의 소스·인증 셀을 실행한 뒤 다음처럼 호출합니다.

```python
from vlade.research.client import job
overrides = {'run_group': RUN_GROUP, 'train.max_steps': 1000,
             'train.sequence_length': 4, 'train.sequence_sampling': 'episode_uniform'}
job(CODE / 'configs/research/ava_fft.json', action='plan', workspace=WORKSPACE, overrides=overrides)
job(CODE / 'configs/research/ava_fft.json', action='run', workspace=WORKSPACE, overrides=overrides)
```

최초 환경 준비에는 같은 설정으로 `action='setup'`을 사용합니다. 전체 절차와 평가 rollout
수 설정은 [연구 실행 안내](research_runs.md)에 있습니다. `train.detach_every=2`,
`train.temporal_feature_layer=-2`도 FFT 설정에서 제어합니다.
기본 batch 1 × 누적 8은 **8 sequences / 32 query 예측**을 한 optimizer step에 사용합니다.
loss는 query·sequence·누적 횟수로 평균합니다. 같은 step 수라도 이전 단일 query 학습과
노출량이 다르므로 GPU 시간·메모리·step 예산을 새 K4 파일럿에서 다시 측정해야 합니다.
복원 추출은 한 epoch의 모든 입력을 한 번씩 방문한다는 뜻이 아닙니다.

일반 FFT·논문 경로의 `all_windows`는 모든 유효 시작점을 섞는 대체 설정입니다.
Suite-balanced는 `episode_uniform`을 사용합니다. `ava_one_step`은 재로딩 진단을 위한
명시적인 K1이며, OpenVLA-OFT도 단일 query를 유지합니다. 기존 Qwen Task-only/KD의
causal history 입력을 AVA의 recurrent K4와 같은 구현이라고 해석하지 않습니다.

재개 cursor는 sequence 수를 세며 완료 step×batch×누적으로 복원합니다. K·샘플링·detach·
feature layer 변경 또는 기존 K1 checkpoint의 K4 재개는 거부합니다. 과거 실행은 당시
Git commit과 설정으로 재개하고 새 K4는 새 run group으로 시작하십시오. 기존 checkpoint,
Hub 게시·봉인·optimizer/scheduler/RNG 저장 계약은 유지합니다.

## 구현과 검증 경계

`data/sequences.py`는 구간 선택, `training/prepared_sequences.py`는 이미지·토큰 변환,
`training/temporal.py`는 상태 전달과 분할 backward를 담당합니다. FFT와 논문 LoRA 경로는
공통 시퀀스 처리를 사용하되 기존 FFT L1과 논문 attention regularizer는 각각 유지합니다.
추가로 `third_party/`에 의존하지 않습니다.

CPU에서는 연속성·분할·suite 균형·재개 순서, 실제 축소 AVA의 recurrent gradient,
분할 backward와 optimizer 저장/재개를 확인합니다. 실제 7B BF16 GPU 학습·수렴·
시뮬레이터 rollout 성능과 논문 수치 재현은 별도 실제 실행 검증 대상입니다.
