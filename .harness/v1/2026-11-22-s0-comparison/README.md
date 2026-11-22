# S0 독립 양자화 조건표·실행기 · 2026-11-22 KST

기존 main 작업 폴더의 사용자 변경을 보존하고 dev/fish13031e의 `1a5033bdf55678d84bd92500680438dd9d2b2d43`에서 격리 worktree로 구현했다. 원격 main은 `0cdf55c6f1dbaa390e630323a72fa806b3631b23`이며 기존 dev가 포함한다. 최신 dev/eastha AGENTS.md의 v1·checks·Git/HF 배포 경계를 확인했다. 이번 작업은 조건표와 기존 S0 평가 실행기의 후속 구현이며 실제 GPU 실험은 실행하지 않았다.

## 변경

- `configs/quantization/s0_comparison.json`: BF16 공통 기준선, 언어부 budget8의 Uniform/ActQuant/QVLA/QuantVLA, 언어·비전부 budget4의 Uniform/ActQuant/QVLA로 총 8개 독립 조건. 기본값은 기존 8 calibration/16 validation/seed42/rollout0의 파일럿이며 최종 선정 기준이 아니다.
- `configs/quantization/s0_assets.json`: 원격 registry `195d32e54c7f48960411c32963d91746183de488`의 S0·prepared_d_seed42만 별도 고정. S0 revision은 `380f8fb1bd828d42e7f396d6979d699b0b7d0cff`, 데이터는 `6d2542577647a9e291554524af4e1dd0928e9df6`. 현재 remote registry에 benchmark asset 키가 없어 임의로 만들지 않고, 기존 storage Lock의 `libero_plus_assets`/revision `dd2bd61b7d9a6fef1abc52d606e983b41886a149` 항목을 원문 그대로 사용했다. 기존 Lock은 변경하지 않았다.
- `src/vlade/evaluation/quantization_comparison.py`와 `scripts/quantization/run_s0_comparison.py`: 기본 동작은 입력 다운로드 없는 계획 출력. --run에서만 조건별 별도 프로세스로 평가. 원본 checkpoint에서 각 조건을 시작하고 BF16 action/query를 공유해 pairing 검증. 요청/실제 bit budget, 대상 모듈, 소스·HF identity와 calibration/validation ID를 대조한다. 실제 할당표의 평균 비트를 재계산한다.
- `src/vlade/evaluation/s0_quantization.py`: 기존 S0 평가 로직을 연구 모듈로 이동하고 단일 조건, scope, Uniform W4/W8, 공통 BF16 reference를 추가했다. 기존 `scripts/evaluate/run_s0_quantization.py`는 얇은 호환 진입점이다. 기존 GUI 스위치의 순차 조합 동작은 유지했다.
- 조건별 manifest/summary와 각 worker report를 저장한다. 실패 후에도 완료 BF16·다른 조건을 보존하며 resume는 동일 설정·Git SHA·HF revision과 report hash/pairing을 확인하고 완료 조건을 건너뛴다. 실패·중단 attempt는 덮어쓰지 않는다. 입력 cache 내부 출력과 dirty 연구 소스의 GPU 실행을 거부한다.
- NaN/Infinity 행동과 JSON, 잘못된 precision/scope·query·대상·예산을 거부한다. 선택적 rollout은 기존 고정 student_eval runtime/source 검증과 기존 HF asset 준비를 사용한다. rollout 미완료는 전체 성공으로 취급하지 않는다. 기본 rollout은 0이다.
- `docs/environment/s0_quantization_comparison.md`에 조건표·실행·재개·측정 해석을 작성하고 README/STRUCTURE/기존 S0 안내에 연결했다. Fake quantization·이론적 bpw를 실제 low-bit 가속/저장 크기로 주장하지 않으며 자동 승자를 선택하지 않는다.

## 실행·검증 결과

- 초기 새 CPU 검사 20개 통과. 이후 입력/예산/보고서 pairing·cache 보호·rollout 실패 거부를 보완했다.
- 새 비교 검사 23개 + 기존 Vision PTQ 26개 + HF assets 24개 = 결합 검사 73개 통과. 그 뒤 공유 reference 필드/calibration pairing을 강화한 최종 새 검사 23개 재실행도 통과했다. pytest_asyncio/Python3.14 deprecation 경고 1개다.
- 새 검사에서는 작은 CPU BF16 Linear 정책으로 실제 기존 ActQuant/QVLA/QuantVLA 함수와 독립 8조건의 실행 흐름을 검사했다. HF resolver·실제 Qwen/processor 로더·CUDA 경계만 명시적으로 mock했다. 모델 재로딩 초기 가중치 일치, 실패 후 BF16 보존, 완료 작업 재사용/실패만 재실행, 설정·결과 변조 거부, 독립/legacy 옵션 충돌, allocation/평균 비트 검증 등을 확인했다. 실제 pretrained Qwen의 GPU 검증을 뜻하지 않는다.
- 실행은 원래 프로젝트 루트에서 worktree의 검사 파일을 지정했고 `-B`, PYTHONDONTWRITEBYTECODE=1, `-p no:cacheprovider`를 사용했다. 고유 basetemp는 이번 task scratch의 p1/p2/p3/p4다. 결합 검사는 p2, 최종 검사는 p4였다.
- 새 CLI 전체 dry-run 및 L8-Uniform/L8-QuantVLA 선택 dry-run 성공. 선택 실행 계획은 BF16 기준선을 자동 포함한 3개 조건이었다. 기존 worker --help, 새 Python AST/설정 JSON 파싱, git diff --check, runtime 결과와 scratch의 Git 제외를 확인했다.
- 원격 HF 메타데이터 읽기만으로 S0 파일 13개의 존재/크기 및 LFS SHA256 4개, calibration archive LFS SHA256 1개를 확인했다. payload 다운로드·전체 파일 SHA256·실제 모델 로딩은 미실행이다.
- local-llm analyze와 최종 diff review는 로컬 Ollama 접근 실패로 즉시 일반 검토로 전환했다. 정본 변경과 중요 동작 판단은 직접 수행했다. 최종 staged 공백 검사에서 발견한 EOF 빈 줄을 제거했다.

실제 S0 GPU 파일럿, full calibration, action 성능 수치·rollout·Jetson 실행은 미실행이다. 새 의존성을 설치하거나 모델/데이터/원본 캐시를 수정하지 않았다. 사용자 지정 branch 외 commit/push, PR 생성·main 병합, Drive·HF 업로드는 수행 범위가 아니다. 검증 후 이번 scratch를 정리하고 compact 기록과 검사 소스만 보존한다.
