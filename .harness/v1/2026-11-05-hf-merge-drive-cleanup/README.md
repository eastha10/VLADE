# HF 완료 pin · PR 병합 · Drive 배포와 중복 정리

2026-11-05 사용자가 dev/eastha push, 현재 열린 PR 포함 main 병합·pull,
최종 Drive 반영 및 불필요한 Drive 데이터 삭제를 명시적으로 지시했습니다.
기존 일반적인 원본 보존 지침의 이번 정리 예외이며 HF 대체본과 경로를 확인한 대상만 다룹니다.

- 시작 checkout dev/eastha 1fdd1a1, origin/main ef01ed1. GitHub CLI는 없어 저장된
  Git credential을 메모리에서만 사용해 GitHub REST API를 조회했습니다. 토큰은 출력·저장하지 않았습니다.
- 열린 PR #1: codex/ava-transformers-patch-a / deb8bec447e8963159c357c362bad22070b9db7e.
  attention 패치 내용은 현재 main에도 동일하게 존재합니다. 세 충돌 파일은 최신 Git 배포 방식으로
  유지하고 PR의 설치/실패 차단/실제 import source 검사를 추가해 새 schema 2 실행 계약에 맞췄습니다.
- 첫 검사 59 passed / 2 skipped였으며 로컬 Git clone 네 검사는 기본 실행 제한으로 실패했습니다.
  로컬 fixture만 사용하는 source 검사 재실행은 10 passed in 8.88s입니다. 실제 GPU/모델 설치는 하지 않았습니다.
- HF fullframes 128구간/384파일/14,347에피소드/2,238,036시점/60,862,121,397 bytes,
  COMPLETE→manifest SHA256와 원격 payload size/LFS SHA256을 재검증했습니다.
  prepared_d_seed42·ava_t0·ava_fft_step_1975·openvla_oft_t0도 pinned 파일들을 재검증했습니다.
  reports/hf-verification.json에 결과를 보존합니다.
- 현재 HF libero-plus-prepared는 public, VLADE-checkpoints는 private입니다. 공개 설정은 변경하지 않았습니다.
- fullframes_d_seed42 최종 pin을 기존 11개 asset을 바꾸지 않고 lock에 추가하고 환경 문서를 갱신했습니다.
- Drive 실제 프로젝트는 1jbXOZooUbe-SM9oUrKmNac_4Dfthkgs0입니다. 원본 ZIP/TFDS,
  prepared 복사본 3개와 TAR, 모델·step1975·HF 캐시를 조회했습니다.
  고유 suite_map.json·변경된 AVA 설정/코드 백업·실행 로그·Qwen processor provenance는 보존 대상입니다.
