# 기존 HF 입력 읽기 전용 연결 구현 · 2026-11-22 KST

사용자는 기존 파일의 HF 정본 일치 전수 확인 후 연결을 요청했다. 기존 main 사용자 변경을 보존하고 dev/fish13031e의 5cacd196dd1b78254fbe4579cd3a4814ddc9f8dc에서 격리 worktree로 구현했다. 모델·데이터 원본을 변경·복사·이동하거나 새로 다운로드하지 않는 입력 연결이다.

- `src/vlade/storage/local_inputs.py`와 `scripts/data/bind_local_input.py`를 추가했다. 등록 전에 기존 immutable asset_record, 원격 HF 파일 크기/LFS·작은 metadata SHA256, 실제 모델 전체 파일 또는 prepared manifest가 인증하는 전체 payload를 검증한다. receipt는 기존 입력 밖의 별도 runtime JSON에 원자적으로 저장하고 concurrent 등록 lock을 사용한다.
- 공통 `assets.resolve_input`과 HF URI의 `resolve_path`가 명시적 VLADE_LOCAL_INPUTS 등록을 사용하도록 연결했다. exact root만 허용하며 등록되지 않은 임의 로컬 경로/하위 폴더는 기존처럼 거부한다. 고정 record·출처·경로 및 모델 파일/데이터 metadata를 매 사용 시 재검증한다. 기존 dataset reader의 실제 trajectory/query/image hash 검사도 유지한다. 변조된 bound 입력에서 다운로드로 조용히 fallback하지 않는다. 환경 변수를 사용하지 않을 때는 기존 캐시 동작을 유지한다.
- 독립 비교 실행기의 출력 보호에 등록된 root를 추가하고, parent manifest/worker 입력 identity에 receipt SHA256을 기록한다. 기존 입력 안에 실행 결과를 쓰거나 등록 정보를 바꾸어 이전 reference/resume와 섞는 것을 막는다.
- `docs/environment/local_hf_inputs.md`에 등록·offline 실행·검증 경계를 작성하고 README/STRUCTURE/HF 입력 안내/S0 비교 안내에 연결했다. 정식 설정에 개인 절대 경로를 추가하지 않았다. 실제 경로는 Git 제외 runtime receipt에만 둔다.

## 실제 검증

- 초기 새 등록 검사와 기존 HF/S0 실행기 회귀 검사 60개 통과. identity fingerprint 강화를 포함한 결합 검사 62개 통과. 이후 이전 등록 root 내부 lock 쓰기를 막는 사전 검사를 보완하고 최종 등록 전용 검사 16개를 재실행해 모두 통과했다. 기존 HF 24개/S0 23개 경로의 동작은 유지된다.
- 검사는 실제 작은 파일·hash로 URI/exact local root 재사용, 모든 payload 검증, 데이터 reader의 이미지 변조 거부, source revision/receipt 변조, 누락된 HF 지문, remote 실패, 기존 receipt 보존, 등록 파일 위치, lock 충돌 및 출력 보호를 검사했다. 등록 테스트의 remote API만 명시적으로 mock했으며 운영 코드에 검증 해제 옵션은 없다.
- pytest는 원래 프로젝트 루트에서 worktree 검사 파일을 지정하고 -B/PYTHONDONTWRITEBYTECODE=1, -p no:cacheprovider 및 이번 task scratch의 고유 bind-p1/p2/p3 경로를 사용했다. pytest_asyncio/Python3.14 deprecation 경고 1개다.
- 현재 PC의 기존 CUDA 환경(torch 2.12.0+cu130, transformers 5.9.0)에서 Qwen3_5Model import와 작은 CUDA BF16 matmul forward/backward의 유한 출력을 실제 확인했다. pretrained 모델의 8개 조건 결과를 뜻하지 않는다.
- 이미 확인한 실제 S0 13개·prepared payload 600,688개의 HF 동일성을 운영 등록 절차로 다시 확인하고 있다. 기존 모델 등록은 완료했고 데이터 등록은 이 구현 기록 작성 시 진행 중이다. 실제 runtime 등록/평가 결과는 로컬 task README와 Git 제외 공식 실행 결과에 별도로 기록한다.

입력 무결성·인증·등록 persistence·동시성 결정은 위임 금지 범위여서 로컬 LLM에 위임하지 않고 직접 검토했다. 기존 dataset·checkpoint·학습 형식이나 HF lock을 변경하지 않았다. 원격 AGENTS는 blob11063626811456465a013e56a4b06162c287f29f로 이전 확인과 같았다.
