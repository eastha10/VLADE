# 27일 업무·Colab 실행 준비 및 칸반 정리

## 현재 배치: 용도별 디렉터리의 일반 파일

사용자 요청에 따라 `C:/Users/kimdo/Desktop/vlade/`의 일반 파일을 정본으로 사용합니다.
구현은 `src/vlade/`, 실행 명령은 `scripts/`, 실행 노트북은 `experiments/`, 설정은
`configs/`, 설명서는 `docs/`, 검사는 `tests/`에 둡니다. 노트북 생성 명령은
`scripts/prepare/generate_colab_notebooks.py`, 구현은 `src/vlade/utils/colab_notebooks.py`입니다.
`experiments/manifests/colab_source.json`에는 소스 파일별 SHA와 원본 revision을 기록합니다.
노트북은 Drive의 같은 일반 폴더 구조를 검증하고 필요한 파일만 `/content`에 복사합니다.

제가 생성했던 바탕화면 루트 ZIP, `artifacts/colab/2026-10-28/` 전달 복사본 및
Documents/output의 동일 전달본은 양쪽 파일 SHA 일치를 확인한 후 총 18개 파일을 제거했습니다.
기존 사용자 파일이나 다른 output은 정리 대상에 포함하지 않았습니다. 노션 통합 검토 페이지와
연결 카드 5개의 위치·실행 안내를 이 구조로 수정했습니다.

이번 배치·일반 파일 복사 검증은 `source-layout-verification.json`에 기록합니다.
`tests/unit/test_colab_source_files.py`의 새 검사 12개와 기존 검사를 포함해 전체 **127 passed / 40.24초**를
확인했습니다. 일반 파일 전환 독립 검토에서 실행 차단 오류나 필수 경로 누락은 발견하지 못했습니다.
아래 소스 ZIP·Documents 전달 경로·패키지 검사는 배치 수정 전 작업 이력입니다.

## 추가: 마지막 교차 코드 검토

사용자 요청으로 데이터/재개, cache/checkpoint, 노트북/assets를 독립된 3개 검토로 나누고 루트가
발견 사항을 수정했습니다. 원본 split 정본·실제 episode 메타데이터 불일치, 다른 단계의 optimizer 혼합,
다른 episode/hidden 모드 cache part 혼합, orphan 용량 포함, stale 파일럿 보고서·T1 입력 무시,
추출 중단 후 충돌, UUID 보고서 순서, TF GPU 비활성화 누락을 보완했습니다.

새 회귀 검사는 `tests/unit/test_final_data_review.py`, `test_final_cache_review.py`,
`test_final_run_review.py`입니다. WinError 5의 일시적인 directory 게시 잠금은 덮어쓰기 없는
bounded 재시도로 보완하고, 추출 임시 디렉터리 정리는 절대 경계·소유 표식·같은 host 종료 PID를
확인합니다. Windows에서 os.kill(pid,0)은 호출하지 않습니다. 실제 source/submodule은 수정하지 않았습니다.

완료 checkpoint는 파일별 SHA·byte 수 및 step/microstep으로 봉인하고 재개/보고서에서 확인합니다.
파일럿 context는 code/data/config/checkpoint 및 overfit policy까지 묶습니다. 이 형식이 없는 이전
파일럿/보고서는 새 코드에서 거부하므로 새 run 이름을 사용해야 합니다.

검증 수치·source SHA·패키지 검사는 `checkpoint-and-artifact-review.json`에 남깁니다.
실제 pretrained 가중치 다운로드·TFDS 전체 처리·GPU 학습·real rollout은 실행하지 않았습니다.

최종 전체 회귀 검사: `python -m pytest tests/unit -q -p no:cacheprovider
--basetemp=artifacts/testing/final-review-run3 --tb=short` → **115 passed / 35.26초**.
데이터 전용 14건, cache 전용 14건과 checkpoint/report/extraction 회귀 검사를 추가했습니다.
첫 전체 실행의 1건은 노트북 셀 stub 테스트의 json global 누락이었으며 fixture를 수정한 뒤
전체 재실행에서 통과했습니다. 신규 helper/변경 Python AST 및 노트북 셀 19개를 검사했습니다.

## 최신: 재점검 후 코드 보완·검증

사용자가 확인된 코드 문제는 먼저 보완한 뒤 Notion에서 결과 검토를 요청하도록 지시했습니다.
고정 1000-step 대신 실제 query×epoch budget(기본 1 epoch), 공통 episode shuffle/causal query cursor,
주기·균등 위치 validation, 본 학습 기본 미실행·reload/overfit gate, 첫 gradient 진단,
파일럿 기반 저장 총량/저장 직전 공간 검사·atomic JSON, 실제 checkpoint hash의 bounded debug,
cache 코드/software fingerprint·최대 3개 episode replay·실패 COMPLETE 제거,
T0/T1 실측 보고서 schema·최종 test 선정 거부와 소스 교체 검사를 연결했습니다.

최종 검사: `python -m pytest tests/unit -q -p no:cacheprovider
--basetemp=artifacts/testing/colab-code-fixes-run4 --tb=short` → **77 passed / 15.37초**.
노트북 3개·코드 셀 19개 compile/schema 및 변경 Python AST를 확인했습니다.
원본 검토 3건 17항목+실행 2건 7항목 = **24/24** 통합 보존을 대조했습니다.
관련 연구 설계·데이터·Teacher 선정·평가 budget·S*·Jetson 카드도 추가 대조했습니다.
세부 근거: `code-review-verification.json`. 아래 65개 검사·1000-step 기록은 보완 전 이력입니다.

수정된 코드·검증·남은 연구 결정을 같은 Notion 통합 페이지와 실제 실행 카드 2개에 반영했습니다.
검토 예정은 사용자 지정 10/28 아침, 통합 페이지와 실제 실행 카드 상태는 In progress입니다.
실제 GPU·pretrained·TFDS 전처리·real T1 cache·rollout은 여전히 미실행입니다. 최종 epoch,
query/augmentation·temporal 연구 범위·평가 기준·KD/양자화·edge는 연구 결정/실측으로 분리했습니다.

기준: 2026-10-28 한국 시간, 저장소 `C:/Users/kimdo/Desktop/vlade`.
기반 HEAD `600f304292700d2408e226e75cb895e1c1b735c8`와 기존 미커밋 작업을 유지했습니다.
commit/push/branch 변경/submodule 갱신/대용량 다운로드/GPU 실행은 하지 않았습니다.

## 요청과 실제 수행 범위

- Notion Project Tasks의 김동하 담당·Deadline 2026-10-27: 고정 T1 cache와 Qwen Task-only.
- 공식 LIBERO+ split ZIP 다운로드·AVA 전체 가중치 다운로드·Drive 저장·전처리 노트북.
- 기존 FFT에 공통 prepared episode 데이터·tail mask·split/통계 identity·코드 fingerprint 연결.
- 선택한 T1만 사용하는 sequential hidden adapter·episode cache 밀봉/재개·첫 episode replay 비교.
- Qwen native VLM/head Task-only trainer·1-step/reload/overfit/main·optimizer/RNG/query cursor 재개.
- 세 개의 Jupyter notebook·고정 AVA 소스/라이선스·미커밋 VLADE·SHA/commit이 포함된 작은 ZIP.
- 원본 submodule의 파일은 수정하지 않았습니다.
- Submodule Git metadata 읽기는 권한 오류로 제한되었습니다. 상위 저장소의 gitlink로
  참조 commit을 기록하고, 실제 패키지 파일마다 SHA256을 기록해 바이트 출처를 구분했습니다.
  최초 패키지 검사에서 metadata 조회값 null을 발견해 이 fallback과 제한 설명을 추가했습니다.

정식 안내: `docs/environment/colab_kanban_handoff.md`.
노트북: `experiments/colab_01_download_data_weights.ipynb`,
`colab_02_ava_fft_and_fixed_cache.ipynb`, `colab_03_qwen_task_only.ipynb`.
전달본: `C:/Users/kimdo/Documents/ChatGPT/VLADE/output/colab-2026-10-28/`.

## 실제 검증

Windows Anaconda Python 3.13.5, Torch 2.12.0+cpu, Transformers 5.9.0 환경입니다.

- `python -m pytest tests/unit -q -p no:cacheprovider --basetemp=artifacts/testing/colab-2026-10-28-run3 --tb=short`:
  **65 passed / 10.42초**.
- 합성 작은 episode에서 content 중복 split 격리, exact train-only q99, causal history·reset·tail mask·gripper,
  이미지 손상 거부, cache schema·밀봉·reset·replay·resume·출처 혼합 거부, archive 경로 탈출 거부를 검사했습니다.
- 최초 검사에서 cache 입력의 NumPy array JSON 직렬화와 Windows ZIP 작성기의 backslash 정규화 때문에
  두 검사가 실패했습니다. cache 경계에서 Tensor로 변환하고 foreign archive member를 직접 검사하도록
  수정한 뒤 전체 검사 재실행에서 모두 통과했습니다.
- 기존 공식 tokenizer/processor+축소 무작위 native Qwen smoke를 새 출력 경로에서 재실행:
  **passed**, 330 tokens·4 images, action `[1,8,7]`, vision/language/proprio/action gradient 유한·nonzero,
  세 입력 변화 반영, 1 update, save/load exact. 결과: `artifacts/student/colab-handoff-2026-10-28/result.json`.
- Python AST, 노트북 전체 code-cell compile·nbformat schema, 소스 ZIP CRC·파일 SHA/commit·금지 대형 파일
  제외 여부, JSON·문서 링크와 `git diff --check`를 검사합니다. 최종 수치는 validation.json에 기록합니다.

이는 실제 TFDS 전처리·분할 결과, pretrained 0.8B, real Teacher forward/backward, bitsandbytes GPU,
실제 T1 cache, rollout, benchmark 성공률, edge 성능 검증이 아닙니다. Windows에 TensorFlow가 없으며
실제 Colab 설치/데이터/GPU는 사용자가 다음 실행에서 확인해야 합니다.

## 칸반 기록

새 In progress 검토 페이지:
https://app.notion.com/p/406b5dd068708197bcc9df8b8f79e175

- Complete: Teacher 저장 코드, Qwen Student 계약/최소 CPU 검증 코드, Related Work/집필·결과 양식 준비.
- In progress: 실제 고정 T1 cache 생성, 실제 Qwen Task-only 학습. 코드 준비와 실험 완료를 분리했습니다.
- 별도 사용자 검토 카드 3개와 다른 담당자의 데이터/GPU/Teacher 카드는 임의 완료 처리하지 않았습니다.
- 기존 In progress 유지 요청은 이번 사용자의 완료 업무 Complete 이동 요청에 따라 갱신했습니다.
- 검토 페이지에는 실행 전 중요 문제뿐 아니라 미검토 Student 설계/저장 계약/원문·양자화·KD·평가·일정도 포함했습니다.

## 남은 핵심

추가 통합 요청: 검토 예정은 사용자 지정 **10월 28일 아침**으로 정정했습니다.
기존 통합 페이지를 같은 URL에서 개명하고 Deadline을 10/28로 바꿨으며, 25일 연계 검토 3건의
원래 체크리스트·볼 파일·결과 기록 양식 및 cache/Task-only 실행 2건의 완료 조건을 통합했습니다.
원본 검토 카드에도 통합 링크와 수정된 검토 예정일을 안내하고 과거 본문은 보존했습니다.

추가 대화: 사용자는 80/10/10 비율을 확인했습니다. 비율 재결정이 아니라 최종 manifest 대조가
남았습니다. 1000-step은 기존 임시 기본값을 근거 검토 없이 본 학습 노트북에 이어받은 값이며,
기존 '잠정 budget' 문구에 없던 산정 근거 부재·8000 query 처리·전체 epoch 미보장·resume 제한을
Notion 검토 페이지 상단과 정식/전달 문서에 추가했습니다. 실행 코드의 학습량은 변경하지 않았습니다.

데이터 담당 카드 최종 80/10/10과 상세 85/5/5/5 불일치·실제 정본 manifest 전달,
AVA zero-context FFT와 temporal 학습 확장 여부, 첫 GPU step·VRAM·저장공간·Drive I/O,
잠정 budget 확정·T0/T1 validation/rollout·T1 선정, 실제 cache 생성/hidden 형식·용량,
실제 pretrained Student overfit/입력 영향/학습/평가와 후속 KD·양자화·Ablation·edge입니다.

## 출처

- 데이터: https://huggingface.co/datasets/Sylvest/libero_plus_rlds/tree/fb0c7029b076030d5d57227229e4f7460def1f7c
- AVA: https://huggingface.co/LiAuto-DSR/avavla-libero-4in1/tree/090bf68d234ea13bc78476236f82458b25b797b8
- Qwen: https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17
- TFDS ReadConfig/SkipDecoding: https://www.tensorflow.org/datasets/api_docs/python/tfds/ReadConfig ,
  https://www.tensorflow.org/datasets/api_docs/python/tfds/decode/SkipDecoding
- AVA upstream commit `69ae5dfb90f4eb714259693bd2c222e0b3de6456`와 기존 A100 단일 추론 freeze의
  Transformers fork `a03eee5da1794c45f5a911b60a9a4545e6e0ed1f`, dlimp
  `040105d256bd28866cc6620621a3d5f7b6b91b46`를 설치에 사용하도록 고정했습니다.
