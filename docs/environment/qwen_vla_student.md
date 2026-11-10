# Qwen3.5 자체 비전 + AVA Student

현재 Student는 Qwen3.5-0.8B의 **자체 비전 인코더·내장 merger·언어 모듈**을 함께 사용합니다.
DINOv2·SigLIP과 이들을 LLM에 연결하던 별도 3-layer Prismatic projector는 제거했습니다.
AVA 시간 상태, proprio projector, L1 action head는 유지합니다.

구현은 `src/vlade/models/students/qwen_vla.py`의 `QwenVLA`,
설정은 `configs/models/student/qwen_vla_0_8b.json`입니다.
기존 `qwen.py`의 native VLM + 독립 query head 초안 및 Colab 03/05 trainer는
이 변경과 별도로 보존합니다. 기존 학습 진입점은 아직 새 모델로 이관하지 않았습니다.

## 구성과 교체 근거

| 구성 | 현재 Student | 근거 |
| --- | --- | --- |
| Vision | Qwen3.5 자체 ViT: 12 layers, width 768, patch 16 | 고정 Qwen config의 vision_config |
| 비전→언어 연결 | Qwen 내장 merger, spatial merge 2, 출력 1024 | native get_image_features()의 pooler_output이 언어 입력 차원과 일치 |
| Language | Qwen3.5-0.8B 언어 모듈, width 1024 | native text_config |
| 외부 3-layer projector | 제거 | Qwen merger가 이미 비전 토큰을 1024차원으로 정렬하므로 추가 Prismatic projector가 필요하지 않음 |
| Proprio | 원본 AVA 8→1024→1024 GELU projector | 로봇 proprio는 Qwen의 이미지 merger가 처리하지 않으므로 유지 |
| AVA | 원본 AttentionWeightGenerator, 내부 폭 1024 | 명령 FiLM·시간 상태 cross-attention·self-attention·FFN 유지 |
| Action | 원본 L1RegressionActionHead, hidden 4096, MLPResNet 2 blocks | 로봇 연속 액션 chunk [B,8,7] 생성 |
| History | [B,56,1024] action hidden state, 기본 hidden_states[-2] | 이전 시점 상태를 다음 AVA·action placeholder로 전달 |

DINO는 projector가 아니라 비전 인코더입니다. 제거한 것은 DINOv2/SigLIP **인코더와
그 외부 연결 projector**입니다. Qwen의 학습된 내장 merger와 AVA 내부의 modality MLP,
proprio projector는 각자의 역할에 필요하므로 유지합니다.

이 구성은 비전과 언어 백본을 함께 Qwen으로 바꾼 Student입니다.
앞서 검토한 Llama→Qwen **LLM 단독 교체** 실험과 구분합니다.
성능 개선이나 원문 재현 성공을 의미하지 않습니다.

## Native Qwen 경로 보존

- 두 카메라는 현재 primary→wrist 순서로 native Qwen processor에 넣습니다.
  이미지 256×256마다 raw grid 16×16, merger 이후 grid 8×8, 시각 토큰 64개입니다.
  aspect ratio가 다른 입력은 현재의 고정 정사각형 grid 계약에서 거부합니다.
- native image placeholder 위치에 native vision/merger의 출력을 scatter합니다.
  두 이미지 사이의 카메라 표지·명령 토큰을 포함해 native Qwen 토큰 배치를 유지합니다.
- 이미지의 spatial M-RoPE는 Qwen get_rope_index()로 계산하며, proprio 1개와 action
  placeholder 56개는 그 뒤의 위치에서 계속합니다. 단순 1D 위치로 대체하지 않습니다.
- AVA 가중은 실제 image token 위치에만 적용합니다. 이미지 토큰이 한 덩어리로
  맨 앞에 있다고 가정하지 않으며, start/end marker·언어·proprio·action 열은 제외합니다.
- 원문의 visual softmax 가중은 logits에 log(omega)를 더하고 대각선은 가중치 1로
  유지합니다. Qwen full-attention 층 [3,7,11,15,19,23]에 적용하며 DeltaNet 18층은 유지합니다.
  native decoder 경계에서 mask를 분리하므로 Llama fork나 전역 monkey patch가 필요하지 않습니다.
- DINO/SigLIP의 512-token 배치에 맞춘 sink 좌표는 Qwen의 128-token 배치에 대응하지
  않으므로 사용하지 않습니다. 근거 없는 새 sink 좌표를 임의로 지정하지 않습니다.

## 가중치·시간 상태·입출력

초기 가중치와 processor는 `hfasset://qwen_initial` 하나만 필요합니다.
native 비전·merger·언어를 고정 revision에서 함께 로드하며, 내장 merger를 새로 초기화하지
않습니다. OpenVLA 가중치 및 DINO/SigLIP·timm 의존성은 새 Student 경로에 없습니다.
URI와 로컬 HF 캐시 모두 공통 resolve_input 검증을 통과합니다. 대용량 다운로드는 자동 시작하지 않습니다.

forward 입력은 native input_ids, attention_mask, pixel_values, image_grid_thw,
선택적 mm_token_type_ids와 정규화 proprio입니다. GT/Teacher action, 외부 position override,
KV cache와 video 입력은 받지 않습니다. 두 현재 카메라만 받고 과거 이미지는 반복 입력하지 않습니다.

episode/sequence 시작에서 state=None을 전달하고, 다음 timestep에는 반환 recurrent_state를
전달합니다. 학습의 detach는 호출 측 TBPTT 경계에서만 수행합니다.
GT 및 Teacher 생성 action chunk는 기존 PolicyOutput 기반 loss에서 타깃으로 사용할 수 있습니다.
두 종류의 action을 모델 입력이나 서로 같은 타깃으로 취급하지 않습니다.

checkpoint schema는 2입니다. 이전 DINO/SigLIP+별도 projector 버전의 schema 1 checkpoint와
자동 호환하지 않습니다. save/load는 native VLM 전체와 AVA/proprio/action 모듈을 저장하며
episode 상태는 저장하지 않습니다.

```python
from pathlib import Path
import torch
from vlade.models.students.configuration_qwen_vla import QwenVLAConfig
from vlade.models.students.qwen_vla import QwenVLA
from vlade.models.students.backbones.qwen import load_processor, prepare_observation

cfg = QwenVLAConfig.from_json(Path("configs/models/student/qwen_vla_0_8b.json"))
model = QwenVLA.from_pretrained(cfg).to("cuda").eval()
processor = load_processor(cfg)
inputs = prepare_observation(processor, cfg, instruction=instruction,
                             primary=primary_pil, wrist=wrist_pil)
inputs = {key: value.to("cuda") for key, value in inputs.items()}
state = None  # new episode
with torch.no_grad():
    output = model(inputs, normalized_proprio.to("cuda"), state)
state = output.recurrent_state
normalized_chunk = output.actions
```

## 검증

Torch 2.12.0+cpu / Transformers 5.9.0에서 축소 무작위 **native Qwen vision·merger·language**
모듈을 함께 사용해 검사합니다. 외부 합성 비전 인코더로 대체하지 않습니다.
원본 AVA/proprio/L1 head 연결, 2-frame backward, native patch embedding·merger까지의 gradient,
episode 초기화, BF16 저장·재로딩, 잘못된 이미지 grid와 schema 거부를 검사합니다.
AVA 가중이 1일 때 native multimodal Qwen forward와 동일한 hidden state/M-RoPE 경로를
유지하는지도 확인합니다. pretrained 0.8B·GPU 학습·LIBERO+ rollout·Jetson은 미검증입니다.

```bash
python .harness/v1/2026-11-11-ava-teacher-code/checks/qwen_vla_smoke.py --output-dir .harness/v1/2026-11-11-ava-teacher-code/scratch/smoke-native-vision
python -m pytest .harness/v1/2026-11-11-ava-teacher-code/checks/test_qwen_vla.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-11-ava-teacher-code/scratch/pytest-native-vision
```

## 1차 자료

- [고정 Qwen config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)
- [고정 Qwen image processor config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/preprocessor_config.json)
- [공식 Transformers 5.9.0 Qwen 구현](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py):
  get_image_features, get_placeholder_mask, get_rope_index, Qwen3_5TextModel
- [AVA 원문 §3.2–3.3](https://arxiv.org/html/2511.18960v4)
- [고정 AVA source](https://github.com/eastha10/AVA-VLA/tree/69ae5dfb90f4eb714259693bd2c222e0b3de6456):
  attn_weight_generator.py, projectors.py, action_heads.py. Apache-2.0 및 상속 MIT 고지는 원본에 보존합니다.

## 원본 AVA와 같은 역할 분리

| 파일 | 역할 |
| --- | --- |
| src/vlade/models/students/qwen_vla.py | 정책 조합·시간 상태·액션 forward·저장/재로딩 |
| src/vlade/models/students/configuration_qwen_vla.py | 모델 설정과 검증 |
| src/vlade/models/students/backbones/qwen.py | Qwen vision/merger/LLM 로딩·native 입력·M-RoPE |
| src/vlade/models/students/components/action_heads.py | 원본 AVA L1RegressionActionHead |
| src/vlade/models/students/components/projectors.py | proprio projector; 외부 vision projector 없음 |
| src/vlade/models/students/components/attn_weight_generator.py | 원본 AVA visual-weight generator |
| src/vlade/models/students/components/qwen_attention.py | Qwen hybrid attention에 AVA mask 연결 |

구현되지 않은 skeleton 진입점은 새 QwenVLA 정책으로 대체합니다. 원본 submodule은 수정하지 않습니다.

## Student 전용 코드 경계

정책·설정 클래스·Qwen 백본·AVA·action head·proprio projector·attention 연결은 모두 src/vlade/models/students/ 아래에 있습니다. AVA 구성요소는 고정 원본에서 Student 내부로 복사하여 Teacher와 독립시켰습니다. Teacher/prismatic 소스를 런타임에 불러오는 ava_source 로더는 제거했습니다. 원본 출처와 Apache-2.0/MIT 라이선스는 Student의 AVA_LICENSE.txt 및 각 파일의 헤더에 보존합니다. 공통 PolicyOutput·HF 입력 검증은 기존 프로젝트 공통 계약을 사용합니다. CLI·테스트·JSON 설정은 저장소의 scripts/tests/configs 역할을 유지합니다.
