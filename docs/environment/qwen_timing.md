# T1 추론·S0 학습 시간 기록

QwenVLA Task-only/Action KD 학습기는 출력 폴더에 `timings-s0-<session>-<id>.csv`를 자동 저장합니다. 온라인 T1 worker도 같은 출력 폴더에 `timings-t1-<session>-<id>.csv`를 저장하며 `TIMING {...}` 로그를 stderr에 출력합니다. stdout의 Teacher JSON 통신은 유지합니다. 파일은 행마다 flush하고 재개할 때 새 session을 사용하므로 이전 CSV를 덮어쓰지 않습니다.

| event | 측정 범위 |
|---|---|
| `t1_model_forward` | 준비된 입력에 대한 Teacher의 `predict_action` 호출 |
| `t1_inference` | 쿼리 1개의 이미지/입력 준비, 모델 추론, CPU 결과 변환 |
| `t1_request` | Student가 요청을 보내고 응답을 받기까지. IPC와 필요한 과거 쿼리 재추론 포함 |
| `t1_worker_startup` | Teacher 프로세스 시작부터 모델 준비 응답까지 |
| `t1_model_load` | worker 내부 Teacher 모델 초기화 |
| `s0_forward` | 입력 텐서 준비 후 Student forward. `phase`로 train/validation 구분 |
| `s0_backward` | 누적 microbatch 1개의 backward |
| `s0_optimizer` | gradient clipping, optimizer/scheduler step, gradient 초기화 |
| `s0_training_step` | 누적 쿼리 로딩부터 optimizer 갱신까지. 온라인 KD에서는 T1 대기도 포함 |
| `s0_validation` | validation 전체. 학습 시간과 별도 |
| `s0_checkpoint_save` | 저장 공간 확인, checkpoint 직렬화·봉인 및 설정된 Hub 게시 |

시간 단위는 초이며 CUDA 작업 전후에 해당 device를 동기화합니다. 비동기 kernel launch 시간과 혼동하지 마십시오. 중첩 구간이므로 모든 event의 시간을 합산하지 않습니다. Student의 순수 연산 시간은 train `s0_forward` + `s0_backward` + `s0_optimizer`로, 온라인 KD 포함 학습 step wall time은 `s0_training_step`으로 봅니다. 동일 GPU에서 Teacher/Student를 실행하면 동기화가 측정 오버헤드와 프로세스 간 대기에 영향을 줄 수 있습니다.

CSV에는 session, event, 시작 UTC, seconds, status/error_type, optimizer step, episode/timestep, phase, replay, warm-up, batch/누적/chunk 크기, device/GPU/dtype와 동기화 방식을 남깁니다. `context_json`에는 seed·software·체크포인트/데이터 식별자를 기록합니다. 원래 JSONL의 각 optimizer step에도 `s0_train_step_seconds`가 추가됩니다.

추가 warm-up 쿼리를 실행하지 않습니다. T1의 첫 두 실제 추론과 S0의 각 loss closure의 첫 두 forward는 `is_warmup=True`이며 S0의 첫 optimizer window도 표시합니다. Teacher 모델 forward 행은 바깥 `t1_inference`의 동일 episode/timestep 및 순서로 대응시킵니다. `replay=True`인 T1 행은 에피소드 상태 복구에 사용한 과거 쿼리입니다. 벤치마크 평균을 낼 때 warm-up과 replay를 구분하고 hardware·dtype·batch·query 수를 함께 보고합니다. 검증 전후 closure 재생성에 따른 warm-up 표시는 하드웨어가 완전히 차갑다는 뜻이 아닙니다.

기존 학습 CLI와 설정을 그대로 사용하며, 계측 때문에 optimizer·손실·데이터 순서·checkpoint identity를 바꾸지 않습니다. 이 기능을 쓰려면 계측 소스가 포함된 Git commit으로 새 실행을 준비하십시오. 실행 중인 Colab 소스를 덮어쓰거나 기존 실행의 source identity 검사를 완화하지 않습니다. 과거 로그에 없던 T1 시간을 사후 추정값으로 채우지 않습니다.
