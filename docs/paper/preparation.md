# VLADE 집필 준비 · 2026-10-26

연결 카드: [Related Work·실험 조건표·논문 결과 양식 준비](https://app.notion.com/p/403b5dd0687081068f65d2aae66074b0).
현재 연구 기준은 [2026-10-24 확정 프로토콜](https://app.notion.com/p/3deb5dd0687081d782e9e8d418f5a0dc)입니다.
기존 Method Draft의 π0.5·flow KD·horizon 50 설명은 과거 초안이며 현재 AVA/OFT
Teacher와 연속 8×7 액션 계약의 근거로 사용하지 않습니다. 원래 초안은 보존합니다.

## 연구 질문과 주장 조건

- RQ1: LIBERO+ 적응 T1의 행동을 같은 입력의 경량 Qwen Student에 전달하면
  동일 구조·데이터·예산의 Task-only Student보다 교란별 성공률이 개선되는가?
- RQ2: 행동 KD에 표현·시간·의미 감독을 추가했을 때 얻는 이득과 추가 비용은 무엇인가?
- RQ3: Teacher 자체 양자화와 선정 Student 양자화는 각자의 비양자화 모델 대비
  어떤 성공률·메모리·지연 절충을 보이는가?
- RQ4: 최종 후보의 목표 Jetson 보드에서 전처리부터 액션 출력까지 측정한 지연과
  전체 공유 메모리는 실제 제어 요구를 충족하는가?

기여 후보는 고정 Teacher cache의 재현성, 증류 효과의 분리 분석,
동일 checkpoint의 양자화 비교, 실제 엣지 측정입니다. 각각 해당 검증이 완료된 경우에만
논문의 기여로 주장합니다. 성능 개선이나 Jetson 배포 가능성은 현재 주장하지 않습니다.

## Related Work 근거 표

다음은 원문에서 확인한 방법과 **VLADE 적용 판단**을 구분한 집필용 요약입니다.
다른 담당자의 논문 적용 매핑·실험 budget 확정 카드를 대체하지 않습니다.

| 연구·고정 원문 | 원문의 핵심 | VLADE에서의 위치·재현 경계 |
|---|---|---|
| [VLA-AD, v1](https://arxiv.org/html/2605.16241v1), Shi et al., 2026 | 성공한 Teacher rollout의 액션에 offline VLM의 phase·multi-frame direction 의미 감독을 결합 | AVA T1와 별도 Qwen Student에 적용하면 원본 Student 재현이 아닌 변형. 성공 rollout 필터·의미 label 생성 비용·추론 입력의 처리 방식을 별도 기록. 단순 action KD를 VLA-AD라고 부르지 않음 |
| [CoTinyVLA, v1](https://arxiv.org/html/2607.25487v1), Lee et al., 2026 | Qwen3.5-0.8B, 카메라당 8프레임, episode Plan·chunk Think 감독과 instruction paraphrase | 원본은 추론 중 Plan/Think 생성이 액션 경로에 포함됨. 이번 2-query 이력·연속 query head·생성 없는 경로는 계약 검증용 VLADE 변형이며 재현 완료가 아님 |
| [QVLA, v1](https://arxiv.org/html/2602.03782v1), Xu et al., 2026 | 최종 action sensitivity로 channel별 bit를 배정하고 pruning까지 budget 안에서 다룸 | 동일 T1와 동일 S*에서 각각 독립 적용. Qwen hybrid linear/full attention 지원과 실제 packed kernel·scale 비용은 별도 검증 |
| [ActQuant, v3](https://arxiv.org/html/2605.24011v3), Akbari et al., 2026 | action 기반 tensor별 mixed precision 및 block scale 조정; OmniModel.cpp 실행 경로 | 단순 fake quantization과 실제 low-bit runtime을 구분. AVA와 Qwen의 변환 지원·비양자화 제외 모듈·평균 bpw를 기록 |
| [QuantVLA, v4](https://arxiv.org/html/2602.20309v4), Zhang et al., 2026 | 선택적 양자화, attention temperature matching, output head balancing | 주 실험의 DiT 구조와 AVA/OFT·이번 Student의 연속 MLP head는 다름. 부록 G는 non-DiT OpenVLA에서 ATM/OHB 직접 적용이 불가하다고 설명하므로 적용 모듈을 재정의해야 함 |

원문 링크와 버전은 집필 근거입니다. 논문 라이선스가 코드 라이선스를 뜻하지 않습니다.
공식 코드 commit·코드 라이선스·실행 환경·원본 조건과의 변경 목록은 실제 이식 시 별도로
확정합니다. 공개 수치·재현 수치·VLADE 측정치를 한 행에 섞지 않습니다.

## 본문 outline

1. **Introduction**: LIBERO+ 강건성과 엣지 제어 제약, 연구 질문, 검증된 기여만 기술.
2. **Related Work**: 경량 VLA, action/semantic/temporal distillation, action-aware PTQ.
   VLA-AD 의미 감독과 CoTinyVLA의 실행 중 reasoning을 구분하고 재현/변형 명칭 명시.
3. **Method**: 원본 AVA T0 → LIBERO+ FFT T1 고정, 대응 입력·episode cache,
   Qwen Student 입력/출력·head·loss, 독립 Teacher/Student 양자화 분기.
4. **Experimental Setup**: 데이터·split·검증 선정·예산·normalization·시드·환경·실패 처리.
   미확정 항목은 아래 조건표에 남기고 실험 시작 전 담당자의 확정본을 연결.
5. **Results**: 적응/증류/Teacher PTQ/Student PTQ의 네 비교 축. 원본 LIBERO 능력 유지,
   LIBERO+ 교란별 결과, seed 편차, 실제 기기 비용.
6. **Analysis**: history·의미·표현 감독 제거 비교, label 생성 비용, 실패 사례와 오차 누적.
7. **Limitations / Conclusion**: 미지원 kernel, 가용 장비, seed 수, 실제 로봇 전이 한계.

초록은 결과 확정 후 작성합니다. 방법 문장에 구현 여부를, 결과 문장에 측정 근거를 붙입니다.

## 실험 조건 양식

| 항목 | 반드시 기록할 값 | 현재 상태 |
|---|---|---|
| 소스 | VLADE commit + 변경 파일 SHA256, AVA/OFT commit | 기존 소스 manifest 참조; 이번 변경 미커밋 |
| 모델 | 전체 backbone/head/proprio/projector, 초기/최종 checkpoint SHA256 | T1 미고정; Student 후보 revision은 설정에 고정 |
| 데이터 | 학습 repo/revision, episode split 해시, val/test 분리, 관측 참조 | dataset revision은 FFT 설정 참조; split 미확정 |
| 공통 좌표계 | 정규화 통계 해시·차원·gripper·좌표/단위·action mask | Teacher 어댑터 계약에 맞춰 추후 실제 통계 연결 |
| 입력 | 카메라 수·해상도·실제 token 수·history 길이·query 주기 | Student 계약 기본: 2 cameras × 2 queries, 간격 8; 원본 CoTinyVLA와 다름 |
| 학습 | optimizer·dtype·LR·스케줄·batch·누적·update 수·seed | 공정 비교 budget은 선행 카드 확정 대기 |
| KD | cache 해시·frozen T1·loss·가중치·의미 label 생성 모델/비용 | 저장 코드 구현; 실제 T1 cache/KD 학습 미실행 |
| 양자화 | 적용/제외 모듈·W/A bits·평균 bpw·scale/index bytes·calibration split | 세 방법 실제 이식 미실행 |
| 평가 | suites·7교란·난이도·task/episode 수·seed·동일 초기 상태·time cap | 선행 평가 budget 카드 확정 대기 |
| 효율 | 하드웨어/전력모드·동시성·batch·dtype·warmup·반복·동기화 | GPU/Jetson 실측 미실행 |
| 선정 | val 지표·tie-break·허용 지연·메모리, 최종 test 봉인 | 선행 담당자가 확정; 임의 값 기입하지 않음 |

## 결과표 양식

`NR`은 미실행, `UNSUPPORTED`는 구현/환경 미지원, `FAILED`는 실행 오류입니다.
이들을 성공률 0으로 기입하지 않습니다. 숫자는 원시 episode 판정과 run manifest에서만 채웁니다.
동일 초기 episode를 사용한 비교는 짝 비교로 보고하고, 여러 학습 seed가 없으면 단일 seed라고 명시합니다.

| 비교 축 | 기준 | 실험 | LIBERO 유지 | LIBERO+ 7교란 macro | 비용 | 상태 |
|---|---|---|---|---|---|---|
| Teacher 적응 | T0 | T1 | NR | NR | FFT GPU 시간·메모리 | NR |
| Student 증류 | S-BC | 동일 구조 S-KD | NR | NR | Teacher/label/cache 비용 포함 | NR |
| Teacher 양자화 | 고정 T1 | T1-QVLA / ActQuant / QuantVLA | NR | NR | 모델 bytes·peak 메모리·p95 | NR |
| Student 양자화 | val 선정 S* | S*-QVLA / ActQuant / QuantVLA | NR | NR | 모델 bytes·peak 메모리·p95 | NR |

| run ID / suite / seed | camera | robot | language | light | background | noise | layout | 7차원 macro |
|---|---|---|---|---|---|---|---|---|
| 미실행 → NR | NR | NR | NR | NR | NR | NR | NR | NR |

각 셀은 `successes / trials`와 성공률·구간을 함께 기록합니다. Macro는 먼저 각 차원의
성공률을 구한 뒤 7개를 동일 가중 평균합니다. task 수가 다른 차원을 전체 pooling한 값은
별도 micro로 명시합니다. seed 평균/표준편차와 binomial 신뢰구간은 서로 대체하지 않습니다.

| 기기 / run ID | 전체 param·model bytes | peak allocated / reserved / 전체 RAM | 전처리 p50/p95 | forward+head p50/p95 | end-to-end p50/p95 | 실제 control Hz |
|---|---|---|---|---|---|---|
| NR | NR | NR | NR | NR | NR | NR |

## 그림 양식과 작성 조건

- Fig.1: T1 고정 뒤 cache→Student / Teacher PTQ를 독립 분기하고 S* 선정 뒤 Student PTQ.
- Fig.2: LIBERO+ 교란별 성공률과 동일 seed·episode의 비교 구간. 원본 LIBERO는 별도 패널.
- Fig.3: 실제 같은 기기에서의 end-to-end p95 vs 성공률 Pareto, bubble=전체 peak memory.
- Fig.4: 입력 이력/표현/의미 감독 제거 비교. token budget과 학습량도 함께 기입.
- Fig.5: 실패 예시의 관측·액션·gripper 시간축. 미공개 데이터는 원본 저장 경로만 참조.

측정 전에는 성능 그래프를 생성하지 않습니다. 기본 기록 양식은
[experiment-record.template.json](experiment-record.template.json)을 복사해 사용합니다.

## 현재 증거와 인계점

CPU 합성 FFT 검증 및 2026-10-04 A100 단일 액션 추론은 benchmark 성공률 증거가 아닙니다.
이번 저장 round-trip과 축소 무작위 Qwen 모듈 검증 역시 사전학습 0.8B 성능 결과가 아닙니다.
주 담당이 다른 [논문 적용 매핑](https://app.notion.com/p/3e7b5dd0687081e5b2bfec7ab2759712),
[ablation·평가 budget](https://app.notion.com/p/403b5dd0687081baa23cce6c71d25857)은 수정하지 않았습니다.
해당 확정본을 받은 후 조건표의 미확정 항목을 채우면 됩니다.
