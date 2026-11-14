# AVA-VLA source integration

AVA-VLA를 VLADE의 Teacher 및 baseline 후보로 연결했습니다. 원본 소스와 LIBERO
4-in-1 가중치를 로컬에 준비하고 가중치 파일의 SHA-256을 검증했습니다. LIBERO+
자료는 사용자 요청에 따라 로컬에서 삭제했습니다. 2026-10-04에 Colab A100에서
AVA-VLA 단일 추론이 유한한 8×7 액션을 생성해 통과했습니다. benchmark 평가와
증류 성능은 아직 검증하지 않았습니다.

## 원본 소스

- 사용자 포크: https://github.com/eastha10/AVA-VLA
- 공식 원본: https://github.com/LiAuto-DSR/AVA-VLA
- 경로: `third_party/ava_vla/`
- 고정 커밋: `69ae5dfb90f4eb714259693bd2c222e0b3de6456`

원본 submodule은 출처 확인용으로 보존합니다. 현재 연구 실행에는 초기화가 필요하지 않습니다.
실행은 [설정 기반 연구 안내](research_runs.md)를 따릅니다.

## 현재 사용하는 Teacher 코드

현재 AVA Teacher의 모델 코드는 `src/vlade/models/teachers/`에 있습니다. 위 고정 커밋에서
실제로 사용하는 LIBERO 연속 L1 경로를 가져왔으며, 원본 submodule과 Student 구현은
수정하지 않았습니다. 각 파일에 원본 저장소·커밋·변경 범위를 적고 `AVA_LICENSE.txt`에
Apache-2.0 및 상속 MIT 고지를 보존합니다.

| 파일 | 현재 사용하는 역할 |
| --- | --- |
| `ava_vla.py` | Teacher 정책 클래스의 조합과 초기화만 담당 |
| `backbones/vlm.py` | HF 모델 생성·Llama 연결·가중치/embedding 인터페이스 |
| `backbones/llama.py` | LlamaForCausalLM 생성·Llama 타입 검증·AVA decoder 설정 전달 |
| `backbones/prismatic.py` | DINOv2·SigLIP 비전 백본과 원본 LayerScale 호환 처리 |
| `components/multimodal.py` | 이미지·proprio·시간 문맥의 token 조합 및 멀티모달 forward |
| `components/action_prediction.py` | 연속 L1 추론·시간 feature 선택·Q99 역정규화 |
| `components/projectors.py` | 비전→언어 Prismatic projector와 proprio projector |
| `components/action_heads.py` | MLPResNet 기반 L1RegressionActionHead |
| `components/attn_weight_generator.py` | AVA 시간 상태·attention block·score head의 조합 |
| `components/attention_blocks.py` | 텍스트 FiLM·시간 cross-attention·시각 self-attention·FFN |
| `components/attention_heads.py`, `attention_utils.py` | softmaxscore head·위치 인코딩·MLP 유틸리티 |
| `outputs.py` | HF 출력 형식과 시각 가중치·hidden states 필드 |
| `configuration_ava_vla.py` | 현재 DINOv2/SigLIP·Llama 설정과 HF checkpoint 설정 호환 |
| `processing_ava_vla.py` | 기존 이미지 transform·tokenizer/processor, 로컬 클래스 로딩 |
| `constants.py`, `action_masks.py` | LIBERO 8×7 action·8차원 proprio·Q99 정규화·action token mask |

diffusion/noisy-action 구현, 이산 action 추론, STE/Gumbel/sigmoid attention head와
ALOHA·Bridge 플랫폼 선택 코드는 가져오지 않았습니다. 해당 비활성 모드는 명시적으로
거부합니다. HF의 차이 기반 설정 저장·재로딩에 필요한 원본 기본값
(`siglip-vit-so400m`, `vicuna-v15-7b`)은 유지합니다. 실제 Teacher는 기존 checkpoint의
`dinosiglip-vit-so-224px`·`llama2-7b-pure` 설정을 사용합니다.

`training/fft.py`의 AVA 분기, `training/paper_ava.py`, 이를 호출하는 순차 Teacher cache와
Suite-balanced FFT가 프로젝트 모델·구성요소를 사용합니다. OpenVLA-OFT도 공통 연속 L1
경로를 사용하되 AVA decoder 인자와 시간 attention을 전달하지 않습니다.
활성 RLDS transform·prompt/collator·action tokenizer·통계 저장 유틸리티는
`src/vlade/data/prismatic/`에 있으며 전체 Prismatic 패키지를 복제하지 않습니다.

Llama decoder의 실제 구현은 `configs/runtime/`에 고정한 Transformers 패키지입니다.
AVA는 기존 fork와 승인된 gradient-checkpoint attention patch를 사용하며 설치 결과도 검증합니다.
OFT와 Student는 각각의 독립 환경을 사용합니다. 프로젝트 `third_party/`를 Python 경로에 추가하지 않습니다.

파라미터 이름·state dict 구조, sink 좌표, attention/시간 문맥 규약, optimizer·schedule·
학습 루프·Hub 저장·재개 계약은 유지합니다. 초기 가중치/데이터는 기존 HF 검증 진입점을
통과하고, processor는 이미 검증된 입력 또는 현재 실행의 재개 출력에서 로컬 클래스와
tokenizer를 로드합니다. checkpoint의 Python 파일을 실행하지 않습니다.

기존 timm/torchvision·AVA Transformers fork 환경이 계속 필요합니다. 이 코드 정리는
기존 A100 추론 결과를 새 코드의 GPU 검증으로 바꾸거나 새 학습을 실행한 것이 아닙니다.

### third_party 없이 확인한 범위

작업별 하네스 `checks/test_ava_teacher.py`는 필요한 모델 소스 16개만 임시 공간에 준비하고,
`third_party` 폴더와 기존 저장소의 Python 경로가 없는 별도 인터프리터에서 실제
`OpenVLAForActionPrediction` 클래스를 import합니다. `prismatic` import는 명시적으로
차단합니다. 작은 합성 vision/language 모듈과 실제 Teacher projector·L1 head·AVA를
연결하여 두 프레임 backward, `[8,7]` 액션 추론, `[1,56,32]` 시간 상태와 strict
state dict 저장·재로딩을 확인합니다. 원본 디렉터리는 이동하거나 삭제하지 않습니다.

이 검사는 실제 7B Llama/DINOv2/SigLIP 또는 processor의 실행 검증은 아닙니다.
실제 백본·이미지 processor에는 기존 timm/torchvision 및 AVA Transformers fork가
필요합니다. 현재 Windows 환경에는 이 백본 환경이 없어 해당 실행은 미검증입니다.

데이터 준비·AVA/OFT FFT·논문 학습·Teacher cache·Student/KD·LIBERO(+) 평가의
현재 프로젝트 경로에서 외부 Prismatic import와 원본 디렉터리 사전 검사를 제거했습니다.
이 연결 변경을 실제 7B/GPU 전체 학습·시뮬레이터 검증 완료로 해석하지 않습니다.

## Hugging Face 가중치

| 벤치마크 | 공식 모델 | 로컬 경로·상태 |
| --- | --- | --- |
| LIBERO | [LiAuto-DSR/avavla-libero-4in1](https://huggingface.co/LiAuto-DSR/avavla-libero-4in1) | `checkpoints/teacher/avavla-libero-4in1/` 검증 완료 |
| CALVIN | [LiAuto-DSR/avavla-calvin-abc2d](https://huggingface.co/LiAuto-DSR/avavla-calvin-abc2d) | 아직 받지 않음 |

LIBERO 모델은 고정 revision에서 내려받아 전체 파일의 해시 검증을 끝냈습니다.
보조 `.pt`는 로컬에서 `weights_only=True`로 읽고 각 state dict의 키 수를 확인했습니다.
Colab A100에서 모델·보조 모듈 로드와 단일 액션 생성까지 확인했습니다. CALVIN 가중치는
이번 LIBERO+ 실험 범위에 포함하지 않습니다.

검토할 부속 모듈 패턴은 `action_head--*_checkpoint.pt`,
`proprio_projector--*_checkpoint.pt`, `vision_attn_weight_generator--*_checkpoint.pt`
입니다. 부속 `.pt`만으로 전체 VLA 모델이 구성되는 것으로 가정하지 않습니다.
모델 본체의 safetensors shard, config, tokenizer/processor와 dataset statistics도
함께 확인해야 합니다. 가중치와 데이터는 Git 추적 대상에서 제외합니다.

## 실행 환경을 준비할 때

`configs/runtime/ava.json`과 연구 CLI의 `setup`으로 독립 실행 환경을
준비합니다. 환경에는 PyTorch 2.2.0과 커스텀 Transformers가 포함됩니다.
커스텀 fork의 검토 시 HEAD는 `a03eee5da1794c45f5a911b60a9a4545e6e0ed1f`입니다.
이 커밋으로 A100 단일 추론을 통과했습니다. 실행 결과와 136개 패키지의
버전 기록은 `experiments/runs/inference_smoke/ava_vla_20261004T060317Z.json`에
보관했습니다. TensorFlow Metadata는 1.15.0으로 고정했고 TensorFlow 전처리는
CPU에서 수행했습니다. 후속 K4 변경에서는 축소 AVA의 이전 행동 문맥 전달과 recurrent
gradient도 검사합니다. 실제 7B GPU 학습·rollout은 미검증입니다.
[시퀀스 학습 안내](ava_sequence_training.md)에 상태 초기화와 검증 범위를 구분했습니다.

외부 checkpoint는 `hfasset://` 키 또는 정식 검증 캐시로 지정합니다.
프로젝트 로더는 검증된 가중치·processor 자료를 읽으며 checkpoint의 Python 소스를 실행하거나
원본 파일을 덮어쓰지 않습니다. 소스 identity는 Git commit, 가중치 identity는 HF revision과 파일 무결성입니다.

## 연결 계층

- `src/vlade/models/teachers/`: 현재 AVA 모델·구성요소·processor와 액션·hidden states 추출.
- `src/vlade/data/adapters/`: 이미지·언어·proprioception 전처리와 액션 정규화 정렬.
- `src/vlade/distillation/`: Teacher cache, action KD와 representation KD.

LIBERO 기준 액션은 8스텝 × 7차원, proprioception은 8차원입니다. 원본은 이전 모델
조회에서 얻은 hidden states를 다음 조회의 `temporal_context`로 넘깁니다. 에피소드
시작 시 상태를 초기화하고, Teacher cache를 생성할 때 순서와 액션 chunk 실행 주기를
보존해야 합니다. 이 설정을 CALVIN에도 그대로 적용하지 않고 해당 어댑터와
checkpoint의 규약을 별도로 검토합니다.

공식 LIBERO baseline의 모델 로딩과 짧은 평가를 먼저 확인하고, 그 다음 Student와
증류 실험을 구현합니다.
