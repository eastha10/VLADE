# AVA Teacher Plan 생성과 지도학습

기본 흐름은 기존 T1을 고정한 Plan 직접 생성 → 실제 probe → S2입니다. 추가 Plan 학습은 필수가 아닙니다.
현재 실행 중인 A100에는 [기존 세션 진단 셀](../../experiments/colab_t1_plan_probe_existing_session.ipynb)을
복사하여 환경과 캐시를 재사용합니다. 별도의 감독 학습을 선택하는 경우에만
[학습 노트북](../../experiments/colab_teacher_plan_training.ipynb)을 사용합니다.

## `ouch` / `imas`의 원인 범위

2026-11-24 사용자 로그는 Train t=0/8의 `ouch`, Validation t=0/8의 `imas`입니다.
당시 T1의 원격 메타데이터·logits·생성 ID는 확보하지 못했습니다. 아래는 코드·로컬 tokenizer·
CPU 검사로 확인한 원인 경로이며 실제 T1 재추론 결과와 구분합니다.

- `dev/eastha`의 `13f1904`, `a8fa773`과 main 병합 `0279cbd`를 검토했습니다. 태그 처리는
  완성 `<plan>…</plan>`의 형식만 처리합니다. `normalize_plan`은 짧은 알파벳 평문을
  허용하므로 `ouch`·`imas`의 의미를 검증하지 않습니다.
- 기존 `training/fft.py::compute_loss`는 action hidden state → action head의 masked L1만
  최적화하며 `output.loss`를 사용하지 않습니다. `data/prismatic/transforms.py`의 label은
  프롬프트를 `-100`으로 가리고 액션 토큰·종료만 남깁니다. Plan 정답이 없습니다. 축소 모델의
  기존 경로에서 LM 출력 헤드 gradient가 `None`인 것을 재확인했습니다. 이 경로의 T1은
  **액션 지도 Teacher**이며 FFT 완료를 CoT 학습 완료로 간주할 수 없습니다. 사전학습 언어
  능력 전체가 없다는 뜻은 아닙니다.
- 이전 `min_new_tokens=1`은 EOS가 가장 높은 logit이어도 첫 EOS를 막습니다. 다른 특수·
  액션 토큰까지 억제하면 남은 일반 텍스트 조각이 선택될 수 있습니다. 제어된 축소 Llama
  logits에서 EOS-only가 강제 텍스트+EOS로 바뀌는 현상을 재현했습니다. 실제 T1의 분포는
  측정하지 못했으므로 `ouch`·`imas`의 정확한 생성 원인은 미확정입니다.
- 로컬 AVA T0 tokenizer는 BOS=1, EOS=2, PAD=32000, `add_eos_token=false`입니다.
  `ouch`→`[2123,305]`, `imas`→`[527,294]`로 복원됩니다. Plan 태그는 일반 토큰 조합이며
  special token이 아닙니다. 로컬 T0 tokenizer 검사이며 실제 T1 바이트 대조를 대체하지 않습니다.
- Plan은 PurePrompt `In: …\nOut:`와 두 카메라·proprio prefix를 사용합니다. Qwen chat
  template을 적용하지 않습니다. `inputs_embeds`만 전달하는 생성의 반환 ID는 새 토큰뿐이므로
  입력 길이로 자르면 안 됩니다. 실제 축소 Llama에서 확인했습니다. 기존 코드도 입력 길이로
  자르지 않으므로 slicing 버그를 원인으로 단정하지 않습니다.
- Plan과 action은 별도 forward입니다. Plan에는 action의 recurrent context·AVA attention
  generator를 전달하지 않습니다. 같은 t=0/8 출력만으로 시각 입력 무시를 확정할 수 없고,
  현재 Plan을 action의 내적 추론으로 해석할 근거도 없습니다.

## 수정된 생성 경계

`InferenceActions`, schema 1/2 Student 사전 점검과 `SequentialTeacher.plan`은 Plan 감독
이력 없이도 기존 VLA의 직접 생성을 허용합니다. `plan_supervision` 기록이 실제로 있으면
그 계약을 검증하며 없으면 미감독 생성으로 기록합니다. 기존 필수 감독 이력 검사는
원래 생성 실험과 맞지 않아 제거했습니다. HF 입력·봉인·데이터 identity는 계속 검증합니다.
메타데이터에 학습 이력을 추가하지 않습니다. 형식 통과나 감독 이력으로 의미 품질을
보장한다고 표시하지 않으며, 지시문·고정 문장 fallback은 사용하지 않습니다.

첫 EOS를 허용하고 BOS/UNK/PAD·액션 vocabulary 억제는 유지합니다. 생성 ID 전체에서 마지막
EOS만 분리하고 `skip_special_tokens=false`로 디코딩합니다. EOS-only는 빈 응답, EOS 없이
예산에 도달한 경우는 미완성으로 거절합니다. 완성 태그 쌍 처리 후 평문 검증은 유지합니다.
예산은 1~128이며 예산 증가가 없는 지도 신호를 만들지는 못합니다.

## 실제 Plan 학습 준비

기존 FFT 루프에 선택적인 `action L1 + plan_loss_weight × Plan CE`를 연결했습니다.
기본 `plan_annotations=null`, `plan_loss_weight=0.0`은 액션 전용입니다. AVA FFT의 prepared
K1/K4·Suite-balanced에 적용하며 04 논문 LoRA 경로에는 적용하지 않습니다.

1. 같은 시점의 카메라·proprio·지시를 근거로 검토한 실제 Plan 주석을 준비합니다. 미래 관측·
   Validation/Test 정답을 Train 주석 생성에 쓰지 않습니다. 지시문 복제나 실패한 T1 출력을
   정답으로 쓰지 않습니다. **현재 이 주석 자산은 등록되어 있지 않습니다.**
2. 비공개 HF dataset에 `manifest.json`, `plans.jsonl`을 게시하고 두 파일의 크기·SHA256·
   불변 revision을 runtime registry에 고정합니다. 공식 prepared 데이터와 별도 자산 키를
   사용합니다. `resolve_input_record`는 URI·검증된 로컬 경로 모두 `resolve_input`을 통과시킨
   뒤 두 파일 모두 HF pin에 포함됐는지 확인합니다. 원본 데이터는 변경하지 않습니다.
3. `manifest.json`은 아래 계약을 사용합니다. SHA256은 실제 prepared 데이터의 값입니다.
   `annotation_source`는 `human_reviewed` 또는 `externally_reviewed`이며 검토 이력 선언입니다.
   이 선언 자체가 자동 의미 평가 결과는 아닙니다.

```json
{
  "schema_version": 1,
  "prompt_version": "ava-approach-v1",
  "annotation_source": "human_reviewed",
  "prepared_data_identity": {
    "manifest_sha256": "<prepared manifest SHA256>",
    "split_sha256": "<split SHA256>",
    "normalization_sha256": "<normalization SHA256>"
  }
}
```

`plans.jsonl`은 아래 필드의 JSON을 줄마다 저장합니다. 실제 canonical ID·시점·원문 지시를
사용합니다. 선택한 prepared 입력의 모든 Train query에 정확히 하나의 주석이 필요하며
Validation/Test·중복·범위 밖 시점·누락은 거절합니다. 아래 문장은 스키마 설명용 예시입니다.

```json
{"split":"train","episode_id":"<canonical Train ID>","timestep":0,"instruction":"<원문 지시>","plan":"Move toward the handle with the gripper open."}
```

4. 별도 FFT 설정에서 `fft_runs.ava_vla.checkpoint`를 HF에 고정된 초기 가중치(T1 초기화
   가능)로, `fft_defaults.plan_annotations`를 게시한 자산으로 지정하고 `plan_loss_weight`를
   양수로 설정합니다. 기존 run group/output을 재사용하지 않습니다.
5. 승인된 학습에서는 기존 `scripts/train/run_fft.py --model ava_vla --config <설정>`의
   preflight → 최소 파일럿 → 본 학습과 기존 Hub 저장을 사용합니다. 이 문서는 새 GPU 학습·
   대용량 다운로드·HF 게시 승인이 아니며 이번 작업에서는 실행하지 않았습니다.

`teacher_plan_supervision.py`는 추론과 같은 `plan_prompt`로 별도 멀티모달 LM forward를
실행합니다. Plan을 action token 시퀀스에 끼우지 않습니다. 프롬프트·BOS·이미지·proprio는
`-100`, 본문과 EOS만 label로 설정하고 causal shift는 HF 모델이 한 번 수행합니다. 새 special
token은 추가하지 않습니다. truncation·UNK/특수/액션 target·128-token 초과를 거절합니다.
LM head·백본·vision/proprio에 CE gradient가 흐릅니다. 추가 forward의 실제 GPU 비용은
미측정이므로 기존 액션 FFT 예산을 그대로 적용하지 마십시오.

Validation의 `val_l1`·best 기준은 액션 L1로 유지하고 Plan CE는 Train에만 더합니다. train
loss는 복합 손실입니다. 학습 이력은 Plan 의미·관측 반영·action 일치의 증명이 아니므로
held-out 생성과 [Plan rollout 평가](action_level_distillation.md)를 별도로 확인합니다.
S2는 생성 Plan을 입력으로 액션 손실을 학습하며, 이 변경이 Student에 언어 CE를 추가하지는 않습니다.

## 저장과 재개

supervised FFT의 `vlade_fft.json.plan_supervision`에는 prompt 버전·`plan_ce` 목표·주석
manifest·본문 SHA256, HF 저장소·불변 revision, weight·실제 optimizer step을 기록하고 기존 seal·Hub commit에 포함합니다.
Action-only는 `null`입니다. 데이터·초기 가중치·Git commit·optimizer/scheduler/RNG·K4 계약은
기존 구현을 유지합니다.

Teacher 입력 검사는 현재 FFT가 저장하는 canonical manifest·normalization 식별자도 대조합니다.
기존 3개 식별자 형식은 보존하고, canonical 식별자가 저장된 checkpoint와 새 supervised
checkpoint에서는 해당 값까지 일치해야 합니다.

기존 액션 전용 `--resume`에 주석·CE weight를 추가하면 거절합니다. 목표 변경은 T1을 초기
가중치로 지정한 새 run입니다. 동일 supervised 실행만 같은 설정·주석 pin·weight·소스 commit·
run group/output과 검증된 `last`로 재개합니다. 주석·prompt·weight 변경은 새 run입니다.
HF download의 불변 revision·봉인 검증은 [Hub 안내](../../../../../../docs/environment/hugging_face_checkpoints.md)를 따릅니다.
수동 metadata 편집으로 학습 이력을 만들지 않습니다.

## Validation `N/A`와 실제 검증 범위

Train probe는 `get_signal`로 `[8,7]` Teacher action과 Plan을 함께 요청합니다. Validation은
`get_plan(split='validation')`만 요청합니다. Student 검증은 held-out GT L1이며 Teacher L1은
계산하지 않으므로 기존 `Action shape: N/A`는 액션 생성 실패가 아닙니다. JSONL은 Validation에
`action_shape=null`, `action_status=not_requested_plan_only`, Train에 `action_status=predicted`를
명시합니다. Validation Teacher 액션 측정에는 별도의 causal action 평가가 필요합니다.

이번 로컬은 PyTorch `2.12.0+cpu`, Transformers `5.9.0`, CUDA 사용 불가이고 HF 캐시에 T1이
없습니다. 실제 T1의 train/validation t=0/8 생성, 고정 AVA fork GPU 실행, 실제 Plan 주석 학습,
LIBERO+ 의미·성공률 평가는 **미실행**입니다. 로컬 T0·합성 모델을 실제 T1 대신 사용하지
않았습니다. 합성 회귀·학습 smoke의 명령·수치는
[작업 기록](../../../../../../.harness/v1/2026-11-24-plan-generation-fix/README.md)에 있습니다.
