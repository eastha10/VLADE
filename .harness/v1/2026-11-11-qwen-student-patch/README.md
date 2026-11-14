# Qwen Student 범위 패치 적용·검증

- 작업일: 2026-11-11 (한국 표준시).
- 대상: 현재 Windows VLADE 저장소. `main`, HEAD `ddc86c38632fcfafe48fc4317bda6334a47dffbd`를 유지했습니다.
- 입력 패치: 사용자가 지정한 Documents/ChatGPT/VLADE/work/VLADE 아래 `.harness/v1/2026-11-10-ava-qwen-student/qwen_vla_student_scope.patch` (202,898 bytes). 원본 패치는 수정하지 않았습니다.

## 실제 변경과 보존

`git apply --check` 성공 후 `git apply`로 적용했습니다. 적용 직후 `git apply --reverse --check`도 성공했습니다. 패치의 18개 경로는 추가 6개·수정 6개·삭제 6개입니다.

- `src/vlade/models/students/backbones/qwen.py`와 `students/components/qwen_attention.py`로 이동했습니다. 기존 파일과 내용이 동일함을 확인했습니다.
- `students/components/{action_heads,projectors,attn_weight_generator}.py`에 Student 전용 AVA 정의를 두고, 기존 공통 경로의 해당 래퍼와 `ava_source.py`를 제거했습니다. 원본 submodule의 클래스·함수 20개와 AST가 일치했습니다.
- `students/AVA_LICENSE.txt`를 추가했습니다. 원본 AVA LICENSE와 개행 형식을 제외한 텍스트가 동일했습니다.
- `students/qwen_vla.py`, `tests/unit/test_qwen_vla.py`, `configs/models/student/qwen_vla_0_8b.json`, `README.md`, `STRUCTURE.md`, `docs/environment/qwen_vla_student.md`의 참조·설명을 갱신했습니다.
- 패치 이후 추가 보정: Student 안내의 예제 import 1줄을 새 경로로 수정하고, action_heads.py·projectors.py·qwen_vla.py의 파일 끝 빈 줄 3건을 정리했습니다. 이 보정을 포함한 모든 파일을 패치의 예상 결과와 대조했습니다.

작업 전 README·STRUCTURE의 수정과 Student 관련 미추적 파일이 이미 있었습니다. 패치 대상 밖의 기존 변경 파일 28개는 적용 전후 바이트가 동일했고, 마지막 Git 상태도 유지되었습니다. 기존 작업 기록·데이터·가중치·다른 소스는 수정하지 않았습니다. Git staging·commit·push·브랜치 변경·Drive/HF 업로드는 수행하지 않았습니다.

## 실제 검증

환경: Python 3.13.5, PyTorch 2.12.0+cpu, Transformers 5.9.0, pytest 8.3.4. 검증 시 `PYTHONDONTWRITEBYTECODE=1`, HF offline 설정, 작업 scratch 아래의 TEMP/TMP·HF_HOME·TORCH_HOME을 사용했습니다. 패키지 설치나 외부 자료 다운로드는 수행하지 않았습니다.

실행 명령과 관측 결과:

```powershell
python -B -m pytest tests/unit/test_qwen_vla.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-11-qwen-student-patch/scratch/pytest-qwen-01
# 19 passed, 2 warnings in 20.92s

python -B -m pytest tests/unit/test_student_contract.py tests/unit/test_hf_assets.py tests/unit/test_paper_ava.py -q -p no:cacheprovider --basetemp=.harness/v1/2026-11-11-qwen-student-patch/scratch/pytest-related-01
# 38 passed in 16.30s

python -B scripts/train/smoke_qwen_vla.py --output-dir .harness/v1/2026-11-11-qwen-student-patch/scratch/smoke-qwen-01
# passed: action [1,8,7], recurrent state [1,56,32], loss 1.7554903030395508
```

Qwen 전용 테스트는 최초 적용 상태에서 실행했습니다. 이후 보정은 문서 import와 파일 끝 빈 줄뿐이며, 최종 상태에서 관련 테스트 38개·독립 smoke·아래 검사를 수행했습니다. 두 프레임 backward에서 vision·merger·language·AVA·proprio·action의 유한한 비영 gradient와 저장·재로딩 결과 일치를 확인했습니다. HF 회귀 검사는 미등록/변조 로컬 입력 거부와 고정 revision의 검증된 캐시 수락을 포함합니다.

- 변경·연결 Python 10개를 메모리에서 `compile()`하여 통과했습니다. pyc는 생성하지 않았습니다.
- Teacher·prismatic·이전 공통 모듈 import를 차단한 상태에서 Student 모듈 8개 import가 통과했습니다.
- Student JSON 검증, 설정의 소스/라이선스 경로, README·STRUCTURE·Student 안내의 로컬 Markdown 링크가 통과했습니다.
- `git diff --check`와 미추적 파일까지 포함한 18개 패치 경로의 `git diff --no-index --check`가 통과했습니다. 일반 diff에는 기존 미추적 구현이 표시되지 않으므로 별도로 패치 예상 결과와 대조했습니다.
- 기본 권한의 Git 조회는 submodule 메타데이터 접근 오류가 있었습니다. 확장된 읽기 권한으로 `git status --short`, `git submodule status`, 두 submodule의 `git status --short`, `git diff --check`를 실행하여 정상 확인했습니다.
- AVA `69ae5dfb90f4eb714259693bd2c222e0b3de6456`, OFT `e4287e94541f459edc4feabc4e181f537cd569a8`이며 두 submodule의 변경은 없었습니다. HEAD·main·빈 staging 상태도 유지되었습니다.

경고는 외부 SWIG 형식의 deprecation 경고와 선택적 고속 attention 라이브러리 부재로 인한 PyTorch 대체 경로 안내였습니다. 테스트나 smoke 실패는 없었습니다.

## 검증 범위와 정리

축소 무작위 native Qwen 모델의 CPU 검증입니다. 실제 pretrained 0.8B 가중치, GPU 학습, LIBERO+ rollout, Jetson 성능 및 전체 저장소 테스트는 미실행입니다. 이번 패치 적용·검증 범위에 포함하지 않았습니다.

검증 출력·합성 checkpoint·적용 전 임시 사본은 작업의 `scratch/`에만 생성했으며, Git 제외 규칙을 확인한 후 완료 전에 정리했습니다. 보존 기록은 이 README 하나입니다.
