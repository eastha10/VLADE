# S2 · Intermediate Representation + Action Distillation

첨부 SVG의 고정 T1 → 내부 표현·Action Chunk → 학습 가능한 S2 구조를 구현합니다.
T1은 기존 AVA Llama2, Student는 기존 Qwen3.5-0.8B S0입니다. 별도 T1 재학습이나
자연어 Plan 생성·Plan CE·Plan 입력을 사용하지 않습니다. 기존 Plan 연구 경로는 보존하지만
이번 S2 실행과 구분합니다. 제공된 진단에서 T0/T1의 Plan은 모두 실패했고, BF16로 비교한
LM head는 동일했습니다. 이 관측을 자연어 생성의 모든 가능성에 대한 결론으로 확대하지 않습니다.

## 학습 목적과 정렬

`L = task_weight × GT L1 + action_kd_weight × T1 Action L1 + representation.weight × Representation MSE`

기본 세 가중치는 각각 1이며 representation의 값은 초기 실험값입니다. 두 L1과 정규화 공간,
`[B,8,7]` 액션·패딩 mask는 S1과 같습니다. 에피소드 끝의 무효 액션 차원은 표현 손실에서도 제외합니다.

- 텍스트·이미지 토큰은 tokenizer와 개수가 다르므로 ID나 전체 시퀀스 위치로 비교하지 않습니다.
- 두 정책의 **56개 액션 슬롯**을 `(chunk 시점 0..7, 액션 차원 0..6)`의 순서로 정렬합니다.
  각 표현은 `[B,56,D]`입니다. 같은 쿼리·같은 T1 forward에서 정규화 액션과 표현을 함께 받습니다.
- 기존 temporal hidden-state 선택을 그대로 사용합니다. 기본 T1과 Qwen 모두 HF의
  `hidden_states[-2]`에 해당하는 슬롯이며, 층 번호가 같은 의미 공간이라는 가정은 하지 않습니다.
  기존 AVA 에피소드 상태·쿼리 재생·Student TBPTT 경계를 유지합니다.
- Qwen `[B,56,1024]`에 학습 가능한 bias 없는 Linear를 적용해 Llama2 `[B,56,4096]`로 보냅니다.
  이 숫자는 기본 config의 값이며 실행 시 **검증된 T1 config와 Student config에서 차원을 읽어**
  provenance에 저장합니다. AVA Llama2가 아닌 Teacher config는 거절합니다.
- 각 유효 벡터에 affine 없는 LayerNorm을 적용한 뒤 FP32 MSE를 계산합니다.
  Teacher는 eval/frozen이며 목표를 detach합니다. Projection과 Student 모두 gradient를 받습니다.
  Projection 초기화는 Student RNG를 진행시키지 않습니다.

이 방식은 액션 위치에 대응하는 내부 표현 증류입니다. 자연어 CoT나 검증된 추론 과정의
전달을 의미하지 않습니다. Teacher와 Student의 중간층 표현이 완전히 같은 의미라는 보장도 없습니다.

## 설정과 S1 조건

정본은 [qwen_vla_representation_kd.json](../../configs/experiments/qwen_vla_representation_kd.json)입니다.
이 설정은 [S1 기본값](../../configs/experiments/qwen_vla_action_kd.json)에 `distillation.representation`만
추가합니다. S0/T1 HF 키·registry revision·checkpoint pins·전처리 데이터·seed 42·100 step·
누적 8·LR 2e-5·weight decay 0.01·full scope·save/validation 조건이 같습니다.
기본 모델은 `Qwen/Qwen3.5-0.8B`의 고정 revision과 기존 action head를 사용합니다.

실제 S1에서 override나 다른 S0/T1 pin을 사용했다면 **그 S1의 `run-contract.json`**을
노트북의 `S1_RUN_CONTRACT`에 지정합니다. workflow와 effective settings를 상속한 뒤
representation 설정만 추가합니다. 비교 기준 S1은 온라인 Action-only `gt_kd` 단일 run이어야 합니다.
경로가 공란이면 저장소 S1 기본값만 보장하며, 임의의 이전 S1 실행과 동일하다고 주장하지 않습니다.

입력은 기존 `resolve_input`·HF 불변 revision·cache 파일 무결성·S0/T1 봉인·데이터 identity
검사를 통과해야 합니다. 미등록 로컬 파일, Drive 가중치, Plan 실패 시 고정 문장으로 대체하지 않습니다.
표현 KD는 현재 온라인 T1만 지원합니다. 기존 action cache만으로 내부 표현을 추정하지 않습니다.

## 현재 A100 런타임에 붙일 셀

[기존 세션용 노트북](../../experiments/colab_s2_representation_kd_existing_session.ipynb)의 두 코드 셀을
현재 노트북에 복사합니다. 정본 생성기는
[generate_representation_kd_notebook.py](../../scripts/prepare/generate_representation_kd_notebook.py)입니다.

1. `S2_SOURCE_COMMIT`에 **이 구현이 포함되고 원격에서 읽을 수 있는 40자리 Git commit**을 지정합니다.
   현재 작업은 commit/push/Drive 배포를 수행하지 않습니다. 로컬 미커밋 변경은 Colab에서 받을 수
   없으므로, 소스 게시 전에는 셀을 실행해 학습할 수 없습니다. 기존 소스 identity 검사를 완화하지 않습니다.
2. `S2_WORKSPACE=Path(WORKSPACE)`로 현재 설치된 Student/AVA 환경과 HF cache를 재사용합니다.
   실제 S1의 `run-contract.json`이 있다면 지정하고 새 `S2_RUN_GROUP`을 기록합니다.
3. 새 소스를 `/content/vlade_sources/<commit>`에 별도 checkout하고, **별도 자식 프로세스**로
   실행합니다. 현재 커널의 `vlade` import·기존 CODE/SOURCE_COMMIT·학습 출력은 교체하지 않습니다.
   이미 import된 소스 때문에 커널을 재시작할 필요가 없습니다. `locate`로 기존 환경 규격을 확인하며
   설치/setup이나 CUDA·torch·Transformers 재설치는 하지 않습니다.
4. 다른 학습·진단의 GPU 프로세스가 종료된 상태에서 `plan → preflight → pilot → train` 순서로
   같은 실행 셀의 ACTION만 바꿉니다. 런타임을 끊거나 다시 시작하지 않습니다. 다른 프로세스를
   자동 강제 종료하지 않습니다. 기존 환경이 없거나 규격이 다르면 오류로 중단합니다.
5. `preflight`는 S0/T1·입력·표현 차원·Hub 비공개 접근을 확인합니다. GPU forward와 쓰기 성공은
   이후 파일럿에서 확인합니다. 자연어 Plan probe는 필요하지 않습니다.
6. `pilot`은 기존 목표 schedule·누적 8을 유지하는 **1 optimizer step**입니다.
   `representation_mse`, `task_l1`, `action_kd_l1`, `grad_norm`의 유한값과 저장·Hub 게시를 확인한 뒤
   `train`으로 목표 step까지 재개합니다. A100 실제 메모리·속도·학습 성공은 여기에서 확인해야 합니다.

저장소 기본 조건을 CLI로 조회할 때는 다음을 사용합니다. `plan`은 모델 다운로드나 GPU 사용을 하지 않습니다.

```bash
python -B scripts/train/run_distillation.py --config configs/experiments/qwen_vla_representation_kd.json --workspace /content/vlade_distillation --run-group s2-representation-seed42-r1 --action plan
```

## 검증·저장·재개·추론

훈련 metrics에 `representation_mse`를 추가합니다. Validation/best 선택 기준은 **S1과 같은
Plan 없는 held-out GT L1**입니다. validation에서 T1이나 projection을 추가로 사용하지 않습니다.
optimizer·누적·schedule·gradient clipping은 기존 공통 루프를 사용합니다.

Projection은 정책의 `model/` 내부에 넣지 않고 checkpoint의 `representation.pt`에 저장합니다.
이 파일과 `identity.json`의 표현 계약, Student, optimizer/scheduler/RNG/episode cursor를 함께
봉인합니다. 재개 시 projection 가중치와 optimizer moment를 복원하며, 파일·계약 불일치는 거절합니다.
기존 Plan S2 또는 S1 checkpoint를 표현 S2의 학습 재개로 사용하지 않습니다. 시작점은 동일 S0입니다.

HF 경로는 비공개 `VLADE-konkuk/VLADE-checkpoints`의
`runs/<새 group>/S2/checkpoints/step-.../`입니다. 기존 Hub 코드가 projection까지 같은 checkpoint
commit에 게시하며 `last`/`best`/목표 `final` 계약을 유지합니다. 파일럿 종료는 final이 아닙니다.
전송 실패 시 원본과 `hub_pending.json`을 보존하고 `ACTION='retry'`를 사용합니다.
원격 복구에는 `S2_RESUME_REVISIONS={'S2':'<불변 HF commit>'}`와 원래 source/group/settings를 유지합니다.

배포·rollout에서는 기존 `QwenVLA.load(checkpoint/'model')`만 필요합니다. Projection과 T1·Plan이
추론에 추가되지 않습니다. 과거 Plan 효과 평가 노트북은 이번 표현 S2의 평가 경로가 아닙니다.
S1/S2 성능 비교는 같은 조건의 plan-free 정책 rollout으로 별도로 수행해야 합니다.

로컬 CPU smoke는 축소 무작위 native Qwen·Llama 모델로 확인하며 실제 0.8B/7B 가중치,
A100 학습·GPU 메모리·성공률 또는 HF 게시 성공을 입증하지 않습니다.
