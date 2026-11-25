# S2 Representation KD preflight 복구 · 2026-11-25 KST

시작 HEAD는 main의 `4d6145c431301143754b7b4afd8371fb8ab62948`이며 원격 main과 일치했다.
기존 legacy-plan-cot/S2 작업 README 수정 2건과 t1-generated-plans 미추적 폴더는 보존하고
이번 게시 대상에서 제외했다. 참조 대화의 실패 로그와 기존 CLIENT 인자 8개를 확인했다.

## 확인한 원인과 정본 변경

- T1 고정 revision `1fdc672416f2129624b22022e6977345a3b429e3`의
  `assets/T1/files/config.json`은 Llama2 ID와 text_config를 포함하지만 기본 hidden_size와
  num_hidden_layers가 생략된 HF diff 설정이다. 기존 코드는 두 필드의 명시적 존재를 요구했다.
- `src/vlade/distillation/representation_config.py`: 누락된 차원/층만 프로젝트의
  OpenVLAConfig.from_pretrained(local_files_only=True)로 복원한다. trainer가 전달한 기존
  AVA Python에서 실행하며 가중치나 checkpoint의 auto_map 코드를 로드하지 않는다.
  다른 backbone/model_type, 잘못 명시된 숫자와 범위 밖 층은 거절한다. 자식 stderr는 보인다.
- `src/vlade/training/qwen_vla_task.py`: 기존 teacher_python을 설정 복원에 전달한다.
- `src/vlade/research/representation_recovery.py`: 학습 전 실패 run만 소스 전환한다.
  saved config/variant/contract 일치, 기존 입력 registry 동일성, private Hub 및 해당 group의
  원격 결과 부재를 확인한다. 로컬 S2/checkpoint/metrics/pending/미지 파일이 있으면 거절한다.
  이전 run-contract 원문을 같은 실행 폴더에 보존하고 git_commit만 전환한다.
  일반 학습 재개의 정확한 source/identity/optimizer 검사는 변경하지 않았다.
- `src/vlade/utils/representation_kd_notebook.py`와 생성 notebook:
  saved effective config 재사용, 선택적 preflight 복구 인자, 완전한 단일 복구 셀을 추가했다.
  S2_CODE/S2_SOURCE_COMMIT/CLIENT를 새 소스로 함께 연결하고 기존 CODE/SOURCE_COMMIT,
  workspace, 환경, cache, group `s2-repr-20261125T081316Z-88f7798e`를 보존한다.
- README/STRUCTURE/S2 실행 안내에 원인·복구 조건·검증 범위를 반영했다.
  S1/S2 정식 JSON 설정·HF lock·runtime 환경 설정·loss·schedule·Hub 저장 계약은 변경하지 않았다.

## 실제 T1 검증

`python -B -X utf8 .harness/v1/2026-11-25-s2-preflight-fix/checks/verify_t1_config.py --contract`

기존 로컬 HF 인증을 사용해 고정 registry `195d32e54c7f48960411c32963d91746183de488`와
해당 T1 revision에서 config/checkpoint_manifest/vlade_fft 작은 파일만 읽었다.
config 61,603 bytes / SHA256 `ad44d42ca864f5322f001d192e5f14300d8c5674636d869a5b9ac3c3f5e1a47b`가
registry와 checkpoint seal 모두에 일치했다. 나머지 메타데이터도 registry 크기·해시에 일치했다.
T1 step1975, temporal_feature_layer=-2, 복원 차원4096/32층, S2 계약1024→4096/56슬롯을 확인했다.
요약 근거는 `reports/t1-config.json`에만 보존했다. 원본 전체 설정·가중치는 보존 기록에 넣지 않았다.

Python/urllib와 httpx 연결은 WinError10054로 끊겼다. Windows curl의 인증서 검증을 유지하는
TLS1.2 연결로 성공했다. 최초 조회의 원격 경로에서 files/를 빠뜨려 404였으며 기존 resolver의
directory 경로 규약대로 수정했다. 최종 검증도 간헐적 연결 reset 후 재시도로 성공했다.
인증 값은 stdin으로만 전달하며 명령 인수·기록·출력에 포함하지 않았다.

## CPU 검사와 실제 명령

기존 Anaconda Python3.13.5, Torch2.12.0+cpu, Transformers5.9.0을 사용했다. 설치하지 않았다.
PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, HF_HUB_OFFLINE=1,
TRANSFORMERS_OFFLINE=1을 사용하고 PYTHONPATH=프로젝트/src,
HF_HOME/TORCH_HOME/TEMP/TMP를 이 작업 scratch 아래로 지정했다.
pytest 실행 전 scratch 부모를 생성했고 고유 basetemp를 사용했다.

```powershell
python -B -X utf8 -m pytest -q -x -p no:cacheprovider --basetemp=.harness/v1/2026-11-25-s2-preflight-fix/scratch/pytest-verified .harness/v1/2026-11-25-s2-preflight-fix/checks/test_s2_preflight_fix.py .harness/v1/2026-11-25-s2-representation-kd/checks/test_representation_kd.py .harness/v1/2026-11-25-s2-representation-kd/checks/test_colab_stream.py .harness/v1/2026-11-25-s2-representation-kd/checks/test_existing_runtime.py .harness/v1/2026-11-22-s0-local-run/checks/test_local_inputs.py .harness/v1/2026-11-11-ava-teacher-code/checks/test_hf_assets.py
```

최종 **100 passed, 2 SWIG deprecation warnings, 73.04초**였다.
실제 T1의 diff 구조를 반영한 설정 복원과 trainer preflight, 잘못된 명시값/타 모델 거부,
기존 AVA interpreter 전달, saved effective config/세 pin/커널 source 보존, 새 CLIENT 연결,
학습 결과가 있는 run 전환 거부와 중단 후 재시도를 검사했다.
기존 CPU 축소 native Qwen/Llama 표현 gradient·mask·동결·동일 query replay,
실제 trainer의 pilot→resume·projection/optimizer 저장·봉인·정책 독립 reload도 통과했다.
기존 HF 입력 검사는 고정 입력/검증된 cache 수락 및 미등록·변조·revision 변경 거절을 포함한다.
GPU와 외부 S0/T1 전체 입력은 trainer smoke에서 CPU 대체물을 사용했다.

첫 추가 검사29개는 통과했다. 이후 기존 tiny trainer fixture가 16차원에 attention heads를
생략한 상태라 전체 설정을 불필요하게 재로딩하면 실패했다. 명시된 두 유효 차원은 그대로 읽고
실제로 생략된 경우만 복원하도록 수정했다. 기존 검사 파일을 수정하지 않았다.
새 CLIENT 통합 검사 한 번은 합성 config fixture의 마지막 개행 누락으로 실패해 실제
notebook 생성 형식과 동일하게 fixture를 수정했다. 위 최종 검사는 이 수정 상태에서 실행했다.

`python -B -X utf8 .harness/v1/2026-11-25-s2-preflight-fix/checks/verify_delivery.py`는
변경 Python4개 compile·문서 상대 링크4개·S1/S2 설정 차이가 representation뿐임·CLI plan의
읽기 전용 S2 라우팅을 통과했다. pytest에서 nbformat 스키마·코드 셀 compile·출력 없음과
생성 notebook의 byte-exact 일치를 확인했다. notebook 생성 진입점도 실제 실행했다.
git diff 공백 검사와 scratch 제외를 확인했다. v0와 submodule은 변경하지 않았다.
이 작업 scratch는 검증 후 정리하며 보존 기록은 1MiB/파일·5MiB/작업보다 작다.

## 미검증 범위

사용자의 실제 Colab AVA/Student 환경·HF 전체 cache에 접근해 full preflight나 A100 forward를
실행한 것은 아니다. 실제 7B/0.8B 가중치의 pilot/train·메모리·속도·rollout·HF checkpoint
게시는 미실행이다. 제공 셀의 preflight/pilot에서 확인해야 한다.
HF checkpoint는 새로 생성하지 않았으며 기존 run의 pending 상태를 측정했다고 주장하지 않는다.
환경 재설치·A100 재연결·대용량 다운로드·Drive 배포는 수행하지 않았다.
코드 commit/push는 사용자 요청에 포함되며 검토한 이번 파일만 대상으로 진행한다.
