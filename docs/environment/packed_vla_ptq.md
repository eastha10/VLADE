# Qwen Student의 ActQuant·QVLA·QuantVLA packed 배포

`src/vlade/quantization/packed/`는 세 방법의 보정 결과를 실제 bit-packed buffer로
저장하고 그 buffer에서 추론합니다. 기존 `evaluation/qwen_quantization.py`의
반올림·역양자화 파일럿과 별도입니다. 대상은 현재 schema2 native Qwen+AVA 연속
L1 action Student이며, 초기 S0와 GT-only/GT+KD 추가 학습본에 같은 API를 씁니다.

## 구현된 경로

| 방법 | 보정·배정 | 배포 표현 |
|---|---|---|
| ActQuant | RBF-HSIC의 행동 관련성−입력 중복성, transformer layer별 오차합 제곱의 greedy 예산 배정, action loss의 diagonal Fisher 및 크기 인자를 사용한 블록 scale 반복 적합 | 텐서별 2/3/4/8/16비트, 기본 블록32 |
| QVLA | Rademacher action-Jacobian proxy, 상위 채널의 실제 action 편차 재평가, 16→8→4→2→0 greedy 강등, 가지치기 수·민감도 제한 | 출력 채널별 0/2/4/8/16비트, 행별 scale |
| QuantVLA | 채널 smoothing, 두 직교 회전과 zigzag permutation, 같은 입력 변환의 실행 시 적용 | 비-DiT 언어부 W8A16, 행별 scale |

ActQuant는 [v4 §4, 식5–11](https://arxiv.org/html/2605.24011v4)을 참고합니다.
현재 연속 행동 Student에는 분류 LM head가 없으므로 AMF의 alpha=1, 즉 Action-Only
Fisher를 사용합니다. 음수 HSIC 점수는0으로 절단하고 raw 값도 보고합니다.
배정은 원소 수로 가중한 평균 weight bit 예산 이하입니다. 3비트와 서로 다른
레이어의 squared-error 균형을 지원합니다. scale은 고정 대칭 정수 코드북에 대한
가중 Lloyd 교대 최적화로 적합하며, 마지막 불완전 블록은 zero-padding합니다.

QVLA는 [v1 §3.3, 식4–8](https://arxiv.org/html/2602.03782v1)을 참고합니다.
기본 proxy는 query당2개 Jacobian probe이고, 상위32개 채널은 각 저비트 후보를
실제로 넣은 뒤 masked action squared-L2를 측정합니다. 원래 weight·bias와 동일한
recurrent context를 매 probe마다 복원합니다. `budget_metric="channels"`는 논문
식7의 채널 평균, 기본 `parameters`는 폭이 다른 행을 원소 수로 가중합니다.
0비트는 코드 payload가 없고 bias와 출력도0입니다. 기본 최대10%와 절대 action
sensitivity 1e-5의 두 제한을 적용합니다. 제한 때문에 예산이 불가능하면 실패합니다.

QuantVLA는 [v4 Appendix B/G](https://arxiv.org/html/2602.20309v4)를 참고합니다.
입력은 `x / input_scale @ R1` → permutation → `@ R2`, weight는 동일한 직교
변환을 folding하여 양자화 전 선형 함수를 보존합니다. SVD 직교 기저와 high/low
zigzag를 사용하며 수치 검증은 FP32로 수행합니다. 구현을 확인한 공식 소스는
[QuantVLA 고정 commit](https://github.com/AIoT-MLSys-Lab/QuantVLA/tree/9e435e56aac4e743526557e5d9ba6e4fddd6ccca)입니다.
현재 Qwen Student에는 DiT가 없으므로 논문의 DiT 전용 ATM/OHB는 적용하지 않습니다.
DiT 모델 경로를 구현했다고 주장하지 않으며 W4 DiT 설정으로 조용히 대체하지 않습니다.

세 방법 모두 activation은 BF16(A16), target은 선택한 language 및 vision block
Linear입니다. patch Conv3d·vision merger·AVA·proprio/action head는 원래 정밀도로
유지합니다. 원 논문의 OpenVLA/π0.5 모델, 전체 calibration·환경 성공률, GGUF 및
OmniModel.cpp 바이너리와의 일치 재현은 별도 범위입니다. 이 코드는 해당 논문
알고리즘을 현재 Student에 적용하는 독립 구현입니다.

## 실제 저장과 추론

`PackedLinear`는 dense floating weight Parameter를 갖지 않습니다. 행별 little-endian
packed uint8 code, byte offset, bit-width, FP32 block scale, bias를 저장합니다.
2/3/4/8비트는 signed symmetric code를 offset한 정수, 16비트 행은 원본 BF16
비트 패턴입니다. scale·bias·변환 행렬·offset·byte padding 비용은 별도로 보고합니다.

CUDA는 함께 배포하는 `kernel.cu`를 설치된 CUDA PyTorch의 NVRTC로 컴파일하고
현재 PyTorch stream에서 실행합니다. 코드북을 레지스터에서 풀어 누적하므로
전체 dense 역양자화 weight를 생성하지 않습니다. CPU는 출력16행 이하 tile만
임시 복원합니다. 두 경로 모두 full floating weight를 장기 캐시하지 않습니다.
CUDA/NVRTC가 없으면 CUDA 경로는 오류를 반환하며 fake quantization으로 fallback하지 않습니다.

이 CUDA kernel은 정확성 중심의 weight-only 참조 구현입니다. INT Tensor Core 최적화나
기존 BF16 대비 속도 향상을 보장하지 않습니다. 입력·누적은 FP32로 처리한 뒤 원래
activation dtype으로 반환합니다. 메모리 압축, 저장 크기, 지연시간은 각각 측정해야 합니다.
배포 모듈은 추론 전용이며 학습용 autograd를 지원하지 않습니다.

## 실행

기본 설정 `configs/quantization/packed_student.json`의 `checkpoint`는 명시적으로
`hfasset://S0`입니다. 새 GT-only/GT+KD는 HF 게시·검증 후 해당 immutable
`hub_checkpoint` record와 별도 키/설정을 사용하십시오. S0 결과를 새 모델 결과로
재사용하지 않습니다. 등록된 기존 입력은 `VLADE_LOCAL_INPUTS`로 연결할 수 있습니다.
세 기본 조건은 모두 같은 language Linear와 평균8비트 예산으로 맞췄습니다.
ActQuant의 sub-4bit 또는 QVLA의 저비트/vision 실험은 각 method의 target_bpw·scope를
설정에서 변경합니다. 서로 다른 범위·달성 비트의 결과를 단순 순위로 비교하지 않습니다.

```bash
# 입력 pin·세 조건만 표시. 다운로드·보정·GPU 계산 없음.
python scripts/quantization/run_packed_ptq.py --config configs/quantization/packed_student.json

# 각 방법은 원본 checkpoint를 별도 프로세스에서 로드.
python scripts/quantization/run_packed_ptq.py --config configs/quantization/packed_student.json \
  --method all --run --output runs/quantization/packed/student-comparison

# 한 방법만 새 출력에 실행.
python scripts/quantization/run_packed_ptq.py --config configs/quantization/packed_student.json \
  --method actquant --run --output runs/quantization/packed/student-actquant
```

기존 출력·외부 입력 root는 덮어쓰지 않습니다. 기본 calibration8 query는 Train의
is_calibration 안에서만 선택합니다. validation16 query의 BF16 원본과 packed 결과를
비교하고, 저장 후 다시 로드한 결과도 비교합니다. 이는 소규모 파일럿이며 전체
데이터 성능·로봇 성공률이 아닙니다. 이 실행기는 rollout을 수행하지 않습니다.
CUDA BF16, transformers5.9 계열, safetensors, numpy/Pillow 및 기존 HF 인증이 필요합니다.

## 배포 artifact

각 출력의 `artifact/`에는 `packed.json`, `model.safetensors`, `processor/`,
`COMPLETE`가 있습니다. safe-tensor에는 packed target buffer와 비양자화 모듈을
모두 포함하므로 원본 floating checkpoint를 다시 받아 재로딩하지 않습니다.
manifest에는 native 모델 설정, normalization, 입력 HF record, 소스·seed·query ID,
bit 배정, payload 크기·SHA256이 들어갑니다. 외부 checkpoint의 pickle이나
직렬화된 Python 실행 코드는 사용하지 않습니다.

외부 artifact도 전체 파일 fingerprint를 고정하여 HF asset으로 게시한 뒤 사용합니다.

```python
from vlade.quantization.packed import load
from vlade.storage.assets import resolve_input
from transformers import AutoProcessor

uri = "hfasset://my_packed_student"
model, manifest = load(uri, registry=registry_path, backend="cuda")
model = model.cuda().eval()
root = resolve_input(uri, registry=registry_path)
processor = AutoProcessor.from_pretrained(root / "processor", local_files_only=True,
                                         trust_remote_code=False)
normalization = manifest["normalization"]
```

외부 입력 API는 공통 `resolve_input`을 먼저 통과합니다. exporter 내부의 생성 직후
round-trip은 방금 쓴 manifest SHA를 대조하며, 미등록 local artifact를 외부 입력으로
받는 public 우회 옵션은 없습니다. 파일 집합·shape·dtype·scale·offset·직교성 및
strict state dict를 검증하고, 손상되면 실패합니다.
