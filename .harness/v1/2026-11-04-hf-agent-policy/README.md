# Hugging Face 협업 지침과 main 병합

2026-11-04 한국 시간. 정본 저장소는 Desktop/vlade입니다.

- AGENTS.md에 조직별 private 저장소, 협업자별 인증, run group, 봉인된 checkpoint 게시·실패 재전송·고정 revision 재개·완료 보고 지침을 추가했습니다. 기존 HF 구현과 안내를 기준으로 작성했습니다. 실제 가중치 전송이나 GPU 학습은 이번 작업에서 실행하지 않았습니다.
- AGENTS.md와 main에서 보존한 테스트 파일의 SHA256, 이를 참조하는 로컬 Colab notebook 6개의 manifest SHA를 맞췄습니다.
- dev/eastha 커밋 5f5ce57775342f7c8bde7db8b3bcb7f5873843c4를 push했습니다. origin/main의 두 canonical 데이터 수정 커밋을 보존하여 main 병합 커밋 41780a6d6ad7ee3f9302c0b6bcb1139946e1317b를 push하고 git ls-remote 결과와 일치를 확인했습니다.
- 병합 충돌은 episode_data.py와 colab_source.json에 있었습니다. main의 EpisodeData 클래스가 HF 경로 해석용 두 문장을 제외하고 dev의 클래스와 AST가 동일함을 확인하여 canonical 검사를 보존했습니다. main의 추가 테스트 54줄도 보존했습니다. 사용자 미추적 파일은 추가·수정하지 않았습니다.
- Anaconda Python으로 test_final_data_review.py, test_colab_source_files.py, test_hf_assets.py, test_hub_checkpoints.py를 실행하여 56 passed in 62.43s를 확인했습니다. pytest 캐시를 끄고 이번 scratch 아래에 임시 파일을 생성했습니다. 병합 후 source manifest 전체 파일 hash와 notebook guard 6개를 확인했고 git diff --check도 통과했습니다.
- Google Drive의 colab/VLADE에서 AGENTS.md, colab_source.json과 관련 notebook 9개를 갱신했습니다. Drive 전용 재개 notebook 3개를 포함합니다. 원래 Drive manifest의 환경별 차이는 보존하고 AGENTS.md pin만 바꿨습니다. notebook은 manifest guard 한 값만 변경하여 기존 셀·실행 결과·metadata를 보존했습니다. 업로드한 11개 파일을 다시 내려받아 exact bytes 일치를 확인했습니다.
- 일회성 다운로드·업로드 준비 파일과 pytest 부산물은 이번 scratch에서 정리했습니다. 실제 HF 인증·업로드·GPU 재개는 이번 검증 범위에 포함하지 않았습니다.
