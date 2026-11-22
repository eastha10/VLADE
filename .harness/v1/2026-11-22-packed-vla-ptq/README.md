# Qwen Student용 실제 packed PTQ 구현

사용자는 QVLA·ActQuant·QuantVLA를 단순 fake quantization 외의 저장·추론 경로까지 작성하고 기존 dev/fish13031e에 게시하도록 요청했다. 브랜치명을 다시 확인해 같은 기존 브랜치만 사용했다. 원래 root main의 사용자 변경과 기존 자료는 보존했다. 기존 중단된 GT/KD 자료 복구·HF 게시·공식 rollout은 재개하지 않았다.

## 실제 구현

- src/vlade/quantization/packed/linear.py: 0/2/3/4/8비트 code와 BF16 행을 갖는 uint8 packed buffer, 행 offset·block scale·bias, 제한된16행 CPU tile 복원. dense floating weight Parameter나 전체 역양자화 weight cache가 없다.
- kernel.cu와 cuda.py: PyTorch에 포함된 NVRTC 및 CUDA driver API로 packaged kernel을 컴파일해 현재 PyTorch stream에서 실행한다. kernel이 packed bytes를 직접 읽어 register에서 복원·누적한다. 기존 설치·MSVC·CUDA toolkit을 변경하지 않았다. Tensor Core 최적화나 속도 향상을 주장하지 않는다.
- allocation.py/calibration.py: ActQuant의 layer-balanced squared-error 예산·3비트·Action-Only Fisher scale 적합, QVLA proxy 및 실제 action 편차 재평가·0비트/가지치기 제한, QuantVLA의 smoothing·dual orthogonal rotation·zigzag와 대응 입력 변환을 구현했다. 현재 continuous non-DiT Qwen Student에 해당하는 W8A16 QuantVLA 경로이며 DiT ATM/OHB를 구현했다고 주장하지 않는다.
- artifact.py: packed buffer와 비양자화 모듈 전체, native 설정·정규화·processor를 self-contained safe-tensor artifact에 저장한다. 생성 직후 manifest SHA로 round-trip하고 외부 load는 기존 공통 HF 입력 검증을 먼저 통과한다. 파일 집합·지문·shape·dtype·offset·scale·직교성·strict state를 검사한다. Windows 교체 저장은 기존 publish_directory의 안전한 재시도를 재사용한다.
- scripts/quantization/run_packed_ptq.py, configs/quantization/packed_student.json: 다운로드 없는 기본 계획과 명시적 --run, 세 조건별 독립 프로세스, packed export/reload 뒤 action 평가를 제공한다. 기본 비교는 동일한 language 평균8비트 예산이다. sub-4bit·vision 범위는 설정으로 선택한다. 기본 모델 키 S0를 새 GT/KD 결과처럼 표시하지 않는다.
- 앞서 준비한 Hub checkpoint/upstream_files 읽기 전용 등록을 완성해 quantization_comparison.py·storage/local_inputs.py에 연결했다. s0_quantization.py의 공유 benchmark workspace는 모델·데이터·고정 benchmark source·등록 root·HF cache 안에 쓰지 못하도록 검사한다. 기존 원격 지문과 실제 파일 검증은 유지했다.
- docs/environment/packed_vla_ptq.md, local_hf_inputs.md, s0_quantization_comparison.md, README.md/STRUCTURE.md에 사용법·범위·저장 표현·한계를 반영했다.

## 실제 검증 결과

최종 소스에서113개 검사를 실제 실행해 모두 통과했다. packed27개(실제 CUDA가 필요한9개 포함), 새 Student/HF binding·공유 workspace 및 기존 local input/S0 실행기/HF asset 회귀86개다. native integration은 작은 무작위 Qwen vision+language+AVA/action 모델이며 0.8B 학습 가중치의 성능 실험이 아니다.

- 실제 RTX5060/CUDA13.0, torch2.12.0+cu130, transformers5.9.0에서 mixed0/2/3/4/8/16비트 CUDA kernel과 CPU tile 참조의 일치를 확인했다. 비표준 입력 폭·3비트 byte 경계·별도 CUDA stream도 검사했다.
- 세 방법 모두 작은 native Qwen에서 실제 보정→packed 교체→safe-tensor 저장→원본 가중치 없이 재로딩→GPU action 출력 일치를 확인했다. 실행기 전체 경로도 세 방법에서 돌려 processor/정규화 저장, BF16 비교, 재로딩 검증 및 report.json 완료를 확인했다. 이 실행기 검사의 HF/processor/data 입력 경계만 합성으로 명시 mock했고 모델·autograd·CUDA 연산은 실제다.
- 정수 payload/metadata, BF16 원본 행, 0비트 bias·출력, AMF objective 비악화, DuQuant 실수 선형함수 보존, 예산·가지치기 불가능 조건, 실패 시 hook/grad/mode 복구, artifact 변조 거부, 미등록 외부 artifact 거부를 검사했다.
- 검사 명령은 기존 격리 eval-venv의 Python -B에서 torch/pytest를 먼저 import하고 기본 Python의 site-packages를 sys.path 끝에 읽기 전용으로 추가한 뒤 pytest.main을 호출했다. 이는 기존 회귀3개에 필요한 requests 누락을 해소하기 위한 것이며 패키지 설치·과거 환경 변경은 없었다. PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, -p no:cacheprovider, 최종 basetemp는 이번 작업 scratch/pytest/final04다. 최종 결과113 passed in28.40s다. CUDA cache도 이번 작업 scratch에 지정했다.
- Python AST9개, JSON 문법, 실제 CLI의 기본 계획 출력, git diff --check를 확인했다. 기본 계획은 다운로드/GPU 실행 false를 반환했다.
- pyproject.toml에 kernel.cu package-data를 선언했다. egg-info/build/bdist/dist를 모두 이번 scratch 아래로 지정해 실제 wheel을 빌드했고, wheel 안의 CUDA source가 정본과 byte 일치함과 TOML 문법을 확인했다. wheel·빌드 생성물은 Git 대상이 아니다.

초기 검사의 기본 환경에는 safetensors/requests가 없었으며 기존 CUDA 환경과 읽기 전용 의존성 참조로 실행했다. tiny native fixture의 deepstack 설정 누락, autocast의 회전 BF16 변환, Windows 일시 rename 오류를 발견해 최종 소스로 수정·검증했다. 실패 결과를 성능 수치나 완료 실험으로 사용하지 않았다.

방법 근거는 ActQuant arXiv2605.24011v4 §4, QVLA2602.03782v1 §3.3, QuantVLA2602.20309v4 Appendix B/G 및 공식 QuantVLA commit9e435e56aac4e743526557e5d9ba6e4fddd6ccca다. 외부 소스를 수정·vendor 복제하지 않았고 문서에 출처와 현재 Student 적용 범위를 명시했다. 원 논문 전체 환경 성공률, GGUF/OmniModel.cpp 바이너리 일치, 실제 GT/KD 품질·속도는 미측정이다.

큰 기존 source의 위치 확인에는 local-llm analyze를 먼저 시도했으나 Ollama 연결 실패로 중단되어 관련 위치를 직접 확인했다. 새 저장 형식·CUDA native 호출·HF 검증 경계·중요 동작·Git 최종 판단은 위임 금지 범위이므로 직접 검토했다. 실제 모델·데이터·실험 출력·캐시를 Git에 포함하지 않는다. 이전 rollout 작업의 worktree·eval 환경은 해당 중단 작업의 후속 재개와 동일 브랜치 작업 위치로 보존하며 이번 작업의 불필요한 scratch만 정리한다.
