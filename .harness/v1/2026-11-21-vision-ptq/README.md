# Vision-only PTQ 구현 · 2026-11-21 KST

- 원래 작업 폴더의 main(600f304292)과 기존 수정·미추적 파일을 보존하고 scratch/worktree에서 dev/fish13031e를 작업했다.
- `git fetch origin main dev/fish13031e` 후 원격 dev/fish13031e(ef4b1058ef)로 fast-forward했다. origin/main(0cdf55c6f1)을 병합했다. fixed_cache.py와 teacher_inference.py의 충돌은 main 대비 차이를 직접 검토하여 기존 작업 브랜치의 timing·오류 보고 확장을 보존했다. 병합 commit은 00ad45b이다. v0·submodule gitlink의 병합 변경은 없다.
- `src/vlade/quantization/vision/{quantizer,sensitivity,allocation,runner}.py`, `scripts/quantization/run_vision_ptq.py`, `configs/quantization/{vision_ptq,hf_assets}.json`, `docs/environment/vision_ptq.md`를 추가했다. README·STRUCTURE·gitignore에 실행 위치와 결과 제외 규칙을 반영했다.
- FP/W8/W4/Mixed(Combined 별칭), A0~A6, 가역적 per-output-channel fake quantization, 고정 FP 시간 문맥, 단일 그룹 민감도와 예산 제한 greedy/random 배정을 구현했다. 결과는 다섯 JSON과 원본 HF checkpoint 식별자로 재구성하며 모델 복제본을 저장하지 않는다.

## 관측한 검증

- 현재 HF registry `195d32e54c7f48960411c32963d91746183de488`의 T1과 prepared_d_seed42만 PTQ 전용 registry에 고정했다. 기존 storage Lock은 수정하지 않았다. T1 revision `1fdc672416f2129624b22022e6977345a3b429e3`, 데이터 revision `6d2542577647a9e291554524af4e1dd0928e9df6`이다.
- 기본 CLI `python -B scripts/quantization/run_vision_ptq.py --config configs/quantization/vision_ptq.json --dry-run` 성공. T1 파일 29개 존재/크기 및 LFS SHA256 13개, 데이터 archive LFS SHA256 1개를 확인했다. 작은 config/index/vlade_fft/봉인 JSON의 SHA256과 T1 step1975·calibration identity를 확인했다. 실제 모듈 순서는 DINOv2→SigLIP이며 기본 대상은 204개다. 가중치·데이터 payload 전체 검증은 미실행이다. 실행 시 기존 resolver·teacher_identity가 담당한다.
- 초기 메타데이터 검사에서 `files/`를 누락하여 이전 경로가 없다고 잘못 판단했다. 검사 경로를 수정하고 원격을 재조회하여 기존 asset도 revision `c685b31d7593128ac2f45ff91cf4629e57ba266a`에서 파일 29개와 LFS SHA256 13개가 유효함을 확인했으며 사용자에게 정정했다. 현재 T1 키와 이전 asset의 경로·revision을 혼동하지 않는다.
- 기존 asset과 현재 T1의 파일 29개 크기·SHA256 manifest는 전부 동일하다. PTQ 전용 설정은 현재 T1 키와 revision을 명시한다.
- 새 CPU 검사 26개 통과. FP 보존, W4/W8 grid/오차, 비전 밖 상태 보존, 복원/실패 복구, 단일 그룹 민감도, loss 안정성, bit budget/동일 seed, 모든 방식의 JSON 출력, 설정 거부, offline CLI, Train calibration 제한, 실제 AVA 모듈과 고정 시간 문맥을 검사했다.
- 기존 HF assets, temporal training, online action KD 검사와 합쳐 66개 통과, 4개 제외. 4개는 현재 Python 환경의 transformers 미설치로 실행하지 못했다: actual_teacher_fft_sequence_backward_updates_recurrence, prepared_loader_transforms_complete_windows_and_resumes, full_s0_restores_trained_head_and_native_backbone, online_s0_preflight_needs_no_cache_or_teacher_gpu. 큰 환경 설치로 우회하지 않았다.
- 최초 결합 실행은 59 통과/11 실패였다. 새 합성 fixture의 AVA embed_dim을 실제 head 계약에 맞게 128로 수정했다. Windows 긴 경로 실패는 원래 프로젝트 루트의 작업 scratch를 pytest basetemp로 사용해 해결했다. 나머지는 위 transformers 의존성 제한이다. 수정 후 결합 실행 66 통과, 최종 PTQ 단독 재검사 26 통과다.
- pytest는 `-p no:cacheprovider`와 고유 basetemp를 사용했다. 최종 PTQ 단독 검사의 basetemp는 `.harness/v1/2026-11-21-vision-ptq/scratch/ptq-final`, 결합 검사는 `scratch/r2`였다. Python bytecode 생성을 비활성화했다. 경고 1개는 설치된 pytest_asyncio의 Python3.14 deprecation이다.
- 새 Python AST와 설정 JSON 파싱, git diff 공백 검사, runs/quantization 및 scratch의 Git 제외를 확인했다.
- local-llm analyze는 입력 context 한도 초과, review는 로컬 Ollama 접근 실패로 즉시 일반 검토로 전환했다. 충돌 결정과 중요 동작은 원문을 직접 확인했다. 최초 회귀 실패 출력은 예상보다 커졌으며 후속 출력은 scratch 로그로 제한했다.

GPU 실행·전체 Calibration·rollout·Action MAE·성공률·속도·최종 성능 비교는 미실행이다. HF 가중치 업로드는 기본 비활성화이며 이번에 실행하지 않았다. 원격 AGENTS의 현재 배포 규약에 따라 별도 요청 없는 Drive 배포도 실행하지 않았다.
