# 04 · AVA 논문 최고 보고 설정 준비

2026-10-31 한국 시간. 사용자가 04 노트북 생성을 요청하고 선택 질문에서
기본 40K/batch64 대신 최고 보고 설정 100K/batch256을 선택했습니다.
학습·모델/데이터 다운로드는 실행하지 않고 파일만 준비하도록 지시했습니다.

## 변경

- `configs/experiments/ava_paper_best.json`: Appendix D.1/Table 7의 100K/batch256과 pretrained OpenVLA 초기값.
  Appendix A의 나머지 조건을 이어 적용한 해석과 public-source/공통 prepared 입력에 따른 차이를 명시.
- `src/vlade/training/paper_ava.py`, `scripts/train/run_paper_ava.py`: frozen-base native linear LoRA,
  same-episode K=4/query stride8, 두 프레임 segment backward/detach, 실제 omega attention penalty,
  fixed LR schedule, adapter/component/optimizer/RNG/sequence cursor 저장·재개 및 세션 시간 종료.
- `src/vlade/utils/paper_notebook.py`, `experiments/colab_04_ava_paper_best.ipynb`: 관리되는 AVA 환경,
  pretrained OpenVLA 준비, preflight, 실제 GPU pilot gate, 본 학습/재개. 모든 대용량 작업 플래그 기본 False.
- 기존 notebook generator가 04도 생성하도록 변경하고 manifest·01~03 bootstrap을 갱신.
  README/STRUCTURE/`docs/environment/ava_paper_best_colab.md`에 역할과 논문 대비 차이를 기록.
- `tests/unit/test_paper_ava.py` 추가 및 기존 source generation test의 노트북 수를 4개로 갱신.

새 학습 경로는 기존 02 FFT와 독립적입니다. 기존 모델/데이터/결과, 외부 submodule은 수정하지 않았습니다.
앞선 8시간 예산 작업의 미커밋 변경을 보존했으며 commit/push는 요청받지 않아 실행하지 않았습니다.

## 검증

Python 3.13의 임시 CPU 환경(torch 2.15.0+cpu)에서 실제 autograd/optimizer 및 봉인 저장/재로딩을 검사했습니다.

```text
python -B -m pytest tests/unit/test_paper_ava.py tests/unit/test_training_loop.py tests/unit/test_colab_source_files.py tests/unit/test_compute_budget.py tests/unit/test_fft.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-10-31-ava-paper04/scratch/pytest-paper-03
60 passed in 7.09s
python -B scripts/prepare/generate_colab_notebooks.py
4 notebooks compile; all manifest files match
git diff --check --ignore-submodules=all
```

초기 저장 변조 검사는 실제 무결성 거부에 성공했으나 오류 문자열 정규식 기대값 차이로 1 failed/46 passed였습니다.
기대값을 실제 오류 메시지에 맞춘 뒤 추가 FFT 회귀 검사까지 위 60개가 통과했습니다.
최종 generator의 built_for 설명만 갱신한 뒤 노트북/manifest 구문·hash도 다시 확인했습니다.

검사 범위: LoRA 초기 출력 동일/원본 동결/adapter update, K=4 gradient 연결과 2→3 detach,
per-sample 실제 omega penalty, same-episode sequence와 epoch cursor, adapter/component/optimizer/RNG 복원,
봉인 손상 거부, 04 기본값에서 설치/다운로드/학습 미시작, 기존 학습 루프와 FFT 회귀.
실제 GPU 모델 로드·forward/backward·K=4 메모리 적합성·full batch256·성능은 미실행입니다.

## Drive

정식 파일 15개를 승인된 `내 드라이브/colab/VLADE`에 추가/갱신했습니다.
신규 이름은 각 대상 폴더에 없음 확인 후 생성했고 기존 파일은 같은 ID로 갱신했습니다.
04: https://drive.google.com/file/d/1ujuZs-jAd5jFOK61bk-CpSrZzdOA-svr/view
15개를 다시 읽어 SHA256 일치와 01~04 bootstrap/manifest 연결을 확인했습니다.
결과는 `drive-verification.json`에 기록했습니다. 01의 사용자 추가 셀 `f_DeAKUMVGV_`와 출력 5개를 보존했습니다.
01~03 Drive 노트북은 bootstrap hash만 갱신했습니다.
미변경 Drive 소스 hash/revision은 기존 manifest를 유지하므로 로컬/Drive 전체 manifest bytes는 다릅니다.
각 환경의 실제 source bytes에 맞게 별도로 묶은 것이며 같은 변경 소스/설정은 내용 일치를 검증했습니다.

최초 신규 설정 업로드는 자동 승인 검토에서 명시적 승인 근거 부족으로 거부되었습니다.
AGENTS.md의 지속적인 Drive 갱신 승인, 공개 논문 설정 파일의 실제 내용, 대상 폴더 부모 경로와 신규 이름 부재를
확인한 뒤 재검토가 승인되어 반영을 완료했습니다. 제한 우회나 다른 전송 경로는 사용하지 않았습니다.
임시 테스트 패키지·payload 비교본은 scratch에서 정리합니다.

## 해석 제한

논문 최고 보고 설정은 LIBERO 기반이며 여기서는 LIBERO+ prepared train을 사용합니다.
D.1에서 재명시하지 않은 항목의 Appendix A 상속, native LoRA, layer -2, query stride8,
증강 비활성·masked sequence-mean loss 등 구현 차이를 별도 문서/설정에 기록했습니다.
정확한 논문 재현이나 98.3% 성능 달성을 주장하지 않습니다.
04의 adapter checkpoint는 기존 FFT용 T1/cache 입력과 직접 호환되지 않습니다.
8시간은 실행 세션별 상한이고 100K 전체 실험의 총 GPU 비용 상한은 아닙니다.
