# 04 · AVA-VLA 최고 보고 설정 준비

2026-10-31: 사용자가 기본 40K/batch64 대신 **최고 보고 성능의 100K/batch256** 설정을 선택했습니다.
노트북은 `experiments/colab_04_ava_paper_best.ipynb`, 설정은 `configs/experiments/ava_paper_best.json`,
실행 경로는 `scripts/train/run_paper_ava.py`/`src/vlade/training/paper_ava.py`입니다.
기존 02의 zero-context FFT, 실행 경로, 예산 설정과 분리합니다.

## 논문 근거와 해석

[AVA-VLA v4 Appendix D.1/Table 7](https://arxiv.org/html/2511.18960v4)은 사전학습 OpenVLA에서
100K gradient step·batch256으로 학습한 one-policy 모델의 LIBERO 평균 성공률 98.3%를 보고합니다.
이는 논문 보고값이며 우리 실험의 성능 보장이 아닙니다. 기본 Appendix A의 40K/batch64 또는
OFT 초기값을 최고 보고 설정과 혼합하지 않습니다. D.1이 나머지 조건을 재명시하지 않아,
rank32·K=4·2/3 경계 detach·LR5e-4·warm-up10%·cosine·clip1·lambda1/c.6/gamma[1.9,.1]은
Appendix A에서 이어 적용한 해석임을 설정에 명시했습니다. 모든 문제에서의 전역 최적값이라는 뜻은 아닙니다.

초기 가중치는 [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f)입니다.
고정 revision의 전체 payload hash를 확인하며, 기존 AVA나 OFT 체크포인트로 대체하지 않습니다.
다운로드는 기본 미실행이며 기존 가중치를 변경하지 않습니다. 새 사전학습 모델 경로는 `checkpoints/pretrained/openvla-7b`입니다.

## 실제 구현 및 차이

고정된 공개 AVA `finetune.py`는 시간 문맥·K=4 TBPTT를 제공하지 않습니다. 따라서 해당 스크립트에
숫자만 넘기지 않고 별도의 VLADE 어댑터를 작성했습니다. 이는 저자 학습 코드나 논문 결과의 재현 검증 완료를 뜻하지 않습니다.

- LLM/vision/projector/action head/proprio의 linear 층에 native LoRA(rank32/alpha16)를 적용하고 AVA generator는 전체 학습합니다.
  alpha16·Gaussian A/zero B 초기화는 고정 공개 source의 선택입니다. 원본 head의 `predict_action`을 보존합니다.
  원본 backbone은 동결하며 새 head/proprio의 frozen base까지 checkpoint에 보관합니다.
- 4개 연속 query를 같은 episode에서 샘플링하고 두 프레임 단위로 backward한 뒤 context를 detach합니다.
  두 segment의 gradient는 같은 optimizer update에 누적합니다. 다음 sequence의 context는 0에서 시작합니다.
  시간 문맥은 저자가 공개한 추론 계약에 맞춰 action hidden layer -2를 사용합니다. 논문 식의 layer 표기와 이 index의 관계는 별도 구현 선택입니다.
- Attention penalty는 generator가 실제 attention에 전달한 omega를 직접 포착하여 sample별 mean과 c=.6의 scalar L2 norm을 계산합니다.
  출력 필드의 enhance 확률을 실제 attention weight로 오인하지 않습니다. MAE와 penalty는 sequence 평균으로 정규화합니다.
- microbatch1×누적256은 **256 sequences**이며 K=4를 다시 batch로 세지 않습니다. optimizer step당 1,024 query 예측입니다.
  100K 목표와 warm-up10K/cosine은 재개 중 고정합니다. dropout0, AdamW/weight decay .01, seed42, 저장 간격5K는 구현 선택입니다.
- 현재 데이터는 01의 **LIBERO+ prepared train split**입니다. 논문의 LIBERO 학습/추가 학습 없는 LIBERO+ 평가와 다릅니다.
  이미지 query 간격8을 유지하므로 4개 관측은 t,t+8,t+16,t+24입니다. 4 query 미만 episode는 제외합니다.
  공통 이미지 계약 때문에 augmentation은 비활성입니다. 따라서 이 파일을 논문의 정확한 데이터/학습 재현으로 보고하지 않습니다.

## 실행과 보관

모든 대용량 동작 플래그는 False입니다. 파일만 준비한 상태이며 Colab A100을 할당하거나 가중치를 다운로드/학습하지 않았습니다.
사용자가 나중에 실행할 때 순서는 설치 → 사전학습 checkpoint 준비 → preflight → K=4 파일럿 → 본 학습입니다.
파일럿은 1 sequence의 backward/update/LoRA·generator gradient/저장·재로딩 검사이며 유효 batch256 실험이나 rollout 평가가 아닙니다.
본 학습을 새로 시작할 때는 현재 code/data/base checkpoint identity와 파일럿 보고서가 일치해야 합니다.

첫 update, 5K step 간격과 세션 예산 종료 전에 `runs/ava_paper_best/seed42/step-...`에 저장합니다.
각 저장은 adapters, 새 head/proprio base 및 LoRA state, AVA generator, optimizer/scheduler/RNG,
데이터 sequence cursor와 완료 봉인을 포함합니다. 기존 파일을 삭제하거나 덮어쓰지 않습니다.
이 형식은 merged FFT checkpoint가 아닙니다. 기존 `select_teacher.py`/`generate_fixed_cache.py`에 직접 입력하지 않습니다.
해당 추론·cache 연결과 환경 rollout 검증은 별도 작업입니다.

세션별 8시간 종료/Colab runtime 반환 요청은 유지하되, 100K 목표를 시간에 맞춰 줄이지 않습니다.
전체 실험 총 비용이 8시간 이내라는 뜻이 아니며, 최종 완료까지 여러 세션/추가 자원이 필요할 수 있습니다.
단일 A100의 실제 VRAM·속도와 full-batch optimizer step 시간은 실측 전입니다.
Colab 서비스/커널 오류 시 자동 runtime 반환까지 보장하지 않습니다.
