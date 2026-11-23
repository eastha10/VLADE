# T1 플랜 일치도 및 S1·S2 평가

- `experiments/colab_t1_plan_alignment.ipynb`에 T1의 실행 액션 8스텝, 두 카메라 GIF, 관절·물체 상태와 수동 채점 양식을 연결했습니다.
- `experiments/colab_s1_s2_plan_effect.ipynb`에서 동일 초기 상태로 현재 플랜·과거 플랜·플랜 생략과 S1을 비교합니다. S2 동일 상태 액션 차이와 paired 성공률을 분리해 저장합니다.
- 평가 실행은 `src/vlade/research/plan_evaluation.py`, 시뮬레이터·분석은 `src/vlade/evaluation/plan_rollout.py`, T1 전송은 `plan_observations.py`에 둡니다. 기존 T1 데이터셋 요청 경로와 S0 rollout용 콜백 검사를 통과했습니다.
- 훈련 실행기는 기존 비공개 Hub의 `runs/<RUN_GROUP>/S1`, `runs/<RUN_GROUP>/S2`에 각각 저장합니다. 목표 step 업로드 시 실제 commit SHA와 경로를 담은 평가용 `CHECKPOINT_PINS`를 출력합니다.
- Python 생성 스크립트로 평가·훈련 노트북 4개를 생성했습니다. nbformat, 셀 컴파일, JSON, 문서 링크 및 diff 공백 검사에 통과했습니다.
- `pytest -q -p no:cacheprovider`로 평가, 기존 Teacher 프로토콜, 훈련 실행, Hub 모의 저장 검사 30개 통과했습니다. 실제 CUDA·LIBERO+ rollout과 Hugging Face 업로드는 미실행입니다.
