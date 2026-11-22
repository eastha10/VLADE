# S0 fake quantization 실험

세 방법을 공정하게 **독립 비교**하려면 [조건표·독립 실행기](s0_quantization_comparison.md)를
사용합니다. `scripts/quantization/run_s0_comparison.py`는 BF16·Uniform W4/W8과
공통 범위/예산별 파일럿을 분리하며 기본 동작은 다운로드 없는 dry-run입니다.
아래 기존 UI·스위치는 순차 조합 비교용으로 유지됩니다.

로컬 화면에서는 ActQuant, QVLA, QuantVLA 체크박스를 선택하고 평가를 실행합니다. 켠 기법을 QuantVLA → ActQuant → QVLA 순서로 같은 모델에 적용하고, 최종 모델로 MAE와 롤아웃 성공률을 측정합니다. 모두 끄면 BF16만 평가합니다. 양수인 BF16 대비 MAE 차이는 오차가 늘어났음을 뜻합니다. 저장된 평가 메뉴에서 이전 보고서를 다시 볼 수 있습니다.

CUDA BF16을 지원하는 PyTorch와 팀의 비공개 Hugging Face 저장소에 접근할 수 있는 인증이 필요합니다. 저장소 루트에서 다음 명령을 실행합니다. 경로를 직접 넘길 때는 같은 레지스트리로 검증한 Hugging Face 캐시 경로를 사용합니다.

```powershell
python -B scripts/data/hf_assets.py refresh-registry --revision 65799b213837eeb9b931a08d8fe6625a5fce43e4 --output $Registry
python -B scripts/evaluate/run_s0_quantization.py --registry $Registry --checkpoint hfasset://S0 --data hfasset://prepared_d_seed42 --output $Report --actquant on --qvla on --quant on
```

`$Registry`에는 레지스트리를 저장할 경로를, `$Report`에는 새 JSON 보고서 경로를 지정합니다. 그래픽 화면을 쓰려면 같은 Python으로 `scripts/evaluate/s0_quantization_ui.py`를 실행하고 `--registry`, `--checkpoint`, `--data`, `--results`, `--scratch`를 지정합니다.

스위치의 기본값은 모두 `off`입니다. 선택한 조합과 BF16에는 동일한 validation 쿼리를 사용합니다. 보고서에는 적용 순서, 단계별 할당, 보정·평가 query ID, 액션 MAE, CUDA 동기화 기준 forward p50/p95, PyTorch 최대 할당·예약 메모리가 기록됩니다. 복합 조합의 비트 수는 단계별 수치입니다.

## 롤아웃 성공률

화면은 기본으로 Spatial 변형 4개를 평가합니다. CLI에서는 `--rollouts 4`를 추가합니다. seed로 선택한 동일한 변형과 동일한 초기 상태에서 BF16과 선택 조합을 평가합니다. 각 변형은 1회 시행하며, 30스텝의 안정화 후 최대 600스텝 동안 모델 액션을 실행합니다. 환경의 성공 판정을 집계하며 보고서에 task ID와 초기 상태 SHA-256을 기록합니다. 이 수치는 선택한 Spatial 표본의 성공률입니다.

평가용 Python에는 [LIBERO+](https://github.com/sylvestf/LIBERO-plus)와 호환되는 robosuite·MuJoCo, 공식 assets가 필요합니다. 기존 VLADE paired rollout은 LIBERO+ commit `4976dc30028e805ff8094b55501d532c48fec182`와 assets revision `dd2bd61b7d9a6fef1abc52d606e983b41886a149`를 사용합니다. 시뮬레이터 로드가 실패하면 MAE는 저장하고 성공률 칸에 실행 불가와 원인을 표시합니다. 롤아웃 수 0은 MAE만 평가합니다.

보정은 seed 42로 섞은 train calibration 에피소드에서 8개 query를, 비교는 validation에서 같은 16개 query를 사용합니다. 에피소드 경계에서 recurrent state를 초기화합니다. `--calibration-queries`, `--eval-queries`, `--warmup`, `--seed`로 변경할 수 있습니다. ActQuant와 QVLA는 `--target-bpw`를 공유하고 QuantVLA 파일럿은 W8A16을 사용합니다.

| 스위치 | QwenVLA 파일럿 범위 |
| --- | --- |
| `--actquant` | 언어·비전 블록 Linear에 query 평균 토큰의 RBF HSIC, 음수 점수 절단, 텐서별 예산 할당, 연속 액션 손실 기반 Fisher와 블록 스케일 적합을 적용합니다. 부동소수점 복원에는 대칭 정수 코드북을 사용합니다. [ActQuant v4](https://arxiv.org/html/2605.24011v4) |
| `--qvla` | 같은 Linear에 액션 Jacobian의 확률적 근사와 가중치 오차로 채널별 정밀도 강등을 수행합니다. 2·4·8·16비트를 사용하고 활성값은 BF16입니다. [QVLA v1](https://arxiv.org/html/2602.03782v1) |
| `--quant` | 비 DiT W8A16 설정을 참고하여 언어 Linear에 보정 입력의 채널별 스무딩과 행별 8비트 가중치 양자화를 적용합니다. 이 파일럿은 DuQuant 회전을 생략합니다. [QuantVLA v4 부록 G](https://arxiv.org/html/2602.20309v4#A7) |

세 경로는 QwenVLA에서 부동소수점으로 복원한 양자화 오차를 평가하는 **fake quantization**입니다. 보고서의 비트 예산은 대상 가중치에서 scale과 메타데이터를 제외한 수치입니다. 지연시간과 메모리는 PyTorch BF16 실행의 관측값이므로 packed low-bit 배포 성능으로 사용하지 않습니다. 검증 query의 액션 오차는 실제 환경 rollout 성공률과 별도입니다.
