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

```bash
git submodule update --init --recursive
```

위 명령은 VLADE에 기록된 커밋을 checkout합니다. `--remote`로 자동 갱신하지
않습니다. 원본 LICENSE와 OpenVLA-OFT에서 유래한 고지를 submodule 안에 보존합니다.

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

원본 설치 안내와 `third_party/ava_vla/pyproject.toml`을 기준으로 독립 실행 환경을
준비합니다. 원본 요구사항에는 PyTorch 2.2.0과 커스텀 Transformers가 포함됩니다.
커스텀 fork의 검토 시 HEAD는 `a03eee5da1794c45f5a911b60a9a4545e6e0ed1f`입니다.
이 커밋으로 A100 단일 추론을 통과했습니다. 실행 결과와 136개 패키지의
버전 기록은 `experiments/runs/inference_smoke/ava_vla_20261004T060317Z.json`에
보관했습니다. TensorFlow Metadata는 1.15.0으로 고정했고 TensorFlow 전처리는
CPU에서 수행했습니다. 이전 행동 문맥을 전달하는 다음 시점 동작은 미검증입니다.

실행 설정에는 AVA-VLA checkpoint의 Hub ID보다 완전한 로컬 다운로드 경로를 우선
사용합니다. 현재 원본의 Hub 부속 모듈 매핑에는 AVA-VLA ID가 없고, attention
generator는 로컬 파일을 찾습니다. 로컬 로더는 config와 모델 코드 파일을 동기화할
수 있으므로, 다운로드 당시 원본 파일 해시도 실행 전에 기록합니다.

## 이후 구현할 연결 계층

- `src/vlade/models/teachers/`: 모델 및 부속 모듈 로딩, 액션·hidden states 추출.
- `src/vlade/data/adapters/`: 이미지·언어·proprioception 전처리와 액션 정규화 정렬.
- `src/vlade/distillation/`: Teacher cache, action KD와 representation KD.

LIBERO 기준 액션은 8스텝 × 7차원, proprioception은 8차원입니다. 원본은 이전 모델
조회에서 얻은 hidden states를 다음 조회의 `temporal_context`로 넘깁니다. 에피소드
시작 시 상태를 초기화하고, Teacher cache를 생성할 때 순서와 액션 chunk 실행 주기를
보존해야 합니다. 이 설정을 CALVIN에도 그대로 적용하지 않고 해당 어댑터와
checkpoint의 규약을 별도로 검토합니다.

공식 LIBERO baseline의 모델 로딩과 짧은 평가를 먼저 확인하고, 그 다음 Student와
증류 실험을 구현합니다.
