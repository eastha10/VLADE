# Teacher 출력 저장 및 Qwen Student 계약

기준일: 2026-10-26, 연결 카드:
[저장 코드](https://app.notion.com/p/3e7b5dd06870807384e4f2c0c53b30d3),
[Student 계약](https://app.notion.com/p/403b5dd06870816aa6dec171db863bd1).

## 저장 형식과 사용 경계

`src/vlade/data/teacher_records.py`는 새 디렉터리에 `records.jsonl`과
정상 종료 시의 `manifest.json`을 저장합니다. Tensor는 detach·CPU 변환 후 JSON 숫자로
저장하고 로드 시 action/proprio/hidden은 float32, token은 int64, mask는 bool로 복원합니다.
JSON으로 보존되는 값은 float32 round-trip을 만족하지만 원본 tensor dtype/device 자체를
보존하지는 않습니다. pickle은 사용하지 않습니다.

`TeacherProvenance`에는 checkpoint ID·전체 bundle SHA256, dataset revision·split 해시,
정규화 통계 SHA256, Teacher tokenizer ID/revision, 좌표/단위/gripper 규약,
query 간격과 temporal 모드를 명시합니다. 전체 checkpoint bundle의 해시는 각 구성 파일의
상대 경로·해시를 정렬한 manifest를 기준으로 계산해야 합니다. 임의 문자열이나
학습 step 숫자를 checkpoint 해시로 대체하지 않습니다.

각 레코드의 필수값은 다음과 같습니다.

| 필드 | 의미 |
|---|---|
| episode_id, timestep, episode_start/end | 동일 episode의 쿼리 순서와 실제 reset/end 경계 |
| observation_refs.primary / wrist | 대응하는 두 카메라의 dataset 참조, 영상 자체는 복제하지 않음 |
| instruction, proprio[8] | 원문 지시문과 Teacher와 동일 정규화의 상태 |
| checkpoint_sha256, teacher_step | 출력 시점의 정확한 모델 식별 |
| normalization_sha256 | action/proprio 변환에 사용한 통계 식별 |
| gt_action[8,7] | demonstration의 연속 액션, prediction과 구분 |
| predicted_action[8,7] | Teacher의 연속 액션 chunk |
| action_valid_mask[8,7] | episode 말미 padding을 제외하는 bool mask |
| gt_token_ids[56] (선택) | Teacher action-token ID; tokenizer ID/revision 필수 |
| action_hidden_states[56,D] (선택) | horizon-major/action-dimension-minor 토큰의 hidden; layer·layout 필수 |

`training_debug`는 매 N번째 호출만 기록하고 최대 record 수·총 bytes·record별 bytes를
제한합니다. 서로 다른 Teacher step도 점검용으로 기록할 수 있지만 checkpoint 해시가
바뀌면 별도 stream을 만듭니다. 최대치 이후 `write()`는 False를 반환합니다.
기본 최대 총 64MiB, record당 16MiB이므로 대형 hidden 저장 예산은 호출자가 명시합니다.

`fixed_kd`는 frozen·비양자화·episode_context Teacher만 허용하며 sampling을 금지합니다.
episode 0 시점 reset, 일정 query 간격, episode 종료 후 ID 재등장 금지, 동일 Teacher step을
검사합니다. 용량 초과·잘못된 순서·미완결 episode는 성공 manifest를 게시하지 않습니다.
reader는 전체 스키마·순서·byte 수·SHA256을 검사한 후 레코드를 반환합니다.
`expected_provenance`를 전달하면 요청한 실험과 다른 cache를 거부합니다.
실제 Teacher 상태 reset 여부는 호출 adapter의 책임이며 저장기만으로 입증할 수 없습니다.

사용 예시는 다음과 같습니다. `provenance`와 `trajectory_record`는 실제 어댑터가 위
필드에 따라 구성해야 하며 episode ID·관측 참조를 추정하거나 만들어 넣으면 안 됩니다.

```python
from pathlib import Path
from vlade.data.teacher_records import TeacherRecordWriter, load_teacher_records

# provenance: TeacherProvenance with actual checkpoint/dataset/stats identities
with TeacherRecordWriter(Path("data/teacher_cache/T1-run-001"), provenance,
                         max_bytes=256 * 1024 * 1024) as writer:
    for trajectory_record in ordered_teacher_queries:
        writer.write(trajectory_record)

for sample in load_teacher_records(Path("data/teacher_cache/T1-run-001"),
                                   expected_provenance=provenance):
    target = sample["predicted_action"]
    mask = sample["action_valid_mask"]
```

이 저장 API는 별도 작업인 Teacher rollout adapter와 연결할 경계입니다.
현재 FFT의 shuffled RLDS batch에는 완전한 episode·관측 참조가 없으므로 저장기가
metadata를 임의로 만들거나 zero-context FFT 출력을 정식 T1 cache로 저장하지 않습니다.

## Student의 구현 선택

우선 후보는 **Qwen3.5-0.8B VLM 전체**와 독립 연속 action head입니다.
[공식 config의 고정 revision](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)
`2fc06364715b967f1860aea9cf38778875588b17`을 사용합니다. hidden 1024,
vision output 1024와 native hybrid linear/full attention·mRoPE를 유지하며 기존 AVA의
Llama attention fork나 head 가중치를 이식하지 않습니다. [공식 Transformers 설명](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

대체 후보는 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)입니다.
이는 text-only 경로이므로 기존 vision encoder와 새 multimodal projector의 정렬이 추가로
필요합니다. 전체 VLM 도입 효과와 언어부 단독 교체 효과를 섞지 않기 위해 이번에는
native vision을 제공하는 Qwen3.5를 우선합니다. 성능·엣지 효율의 우월성을 의미하지 않으며
후보별 실측 전 최종 backbone 채택이라고 하지 않습니다.

`configs/models/student/qwen3_5_0_8b.json`의 입력 계약:

- 한 샘플: 같은 episode의 최근 2개 query, 각 query의 primary·wrist 이미지 총 4장,
  query 간격 8 environment steps, 지시문, 현재 proprio 8차원.
- 시작 t=0에서는 첫 이미지를 반복해 2-query window를 채웁니다. 이후 최근 두 query만
  사용하고 camera/time marker를 붙입니다. 호출자는 episode별로 history를 분리합니다.
- official processor의 native patch/grid와 이미지·token budget을 사용합니다.
  기본 이미지 최대 65,536 pixels, 전체 최대 8,192 tokens. 이미지 placeholder 중간을
  자르는 truncation은 허용하지 않고 초과 시 오류를 냅니다.
- backbone은 causal mask와 native vision-language 위치 계산을 유지하고 cache를
  사용하지 않습니다. 매 query마다 window를 재계산하여 Teacher recurrent state를 요구하지 않습니다.
- fused context를 hidden 1024→head 256으로 투영하고 proprio를 별도 memory token으로
  붙입니다. 8개의 학습 가능한 horizon query가 이 context에 cross attention한 뒤
  MLP가 각 7차원 연속 액션을 출력합니다. proprio는 이번 설계에서 backbone 밖 head에 들어갑니다.
- 공통 `PolicyOutput`: actions[B,8,7], action_features[B,8,256]. 출력은 Teacher와
  동일 통계의 정규화 공간이며 환경 adapter에서만 실제 단위로 변환합니다.
- `masked_action_l1`은 padding을 제외한 유효 원소에만 task/action target L1을 계산합니다.
  Teacher raw token ID는 Student label로 복사하지 않습니다. representation KD는
  동일 horizon의 의미 feature에 별도 projection을 학습하는 후속 작업입니다.

## CoTinyVLA와 구분

이번 구현은 **VLADE의 공통 계약·Task-only 시작점**입니다. 원본 CoTinyVLA의
카메라당 8-frame 이력, proprio soft token, Plan/Think 생성과 학습 목적을 재현하지 않습니다.
원본의 reasoning 생성은 추론 경로에도 들어갑니다. 생성 없는 현 경로에 의미 label만
추가한 결과를 원본 CoTinyVLA 재현이라고 보고하면 안 됩니다.
[원문 §3·부록 I–K](https://arxiv.org/html/2607.25487v1).

## 환경과 검증 명령

Teacher fork와 분리된 Student 프로세스를 사용합니다. 이번 CPU 검증 환경은 Python 3.13.5,
torch 2.12.0+cpu, transformers 5.9.0입니다. 공식 processor는 torchvision 0.27.0을
추가로 요구합니다. 이 버전은 [공식 패키지 metadata](https://pypi.org/pypi/torchvision/0.27.0/json)에서
torch 2.12.0 의존성을 확인했습니다. 이번 실행은 전역 환경을 변경하지 않고
`artifacts/student/python-deps`에만 torchvision을 설치했습니다. 이 임시 의존성은
2026-10-29에 `.harness/v0/2026-10-29-artifact-cleanup/student/python-deps`로 옮긴 뒤,
사용자의 불필요한 파일 삭제 지시에 따라 제거했습니다. 임시 의존성을 다시 사용할 경우
현재 `.harness/v1/` 규약의 작업별 `scratch/`에서만 준비하고 사용 후 정리합니다.

독립 Student 가상환경에서는 해당 버전으로 설치한 뒤 아래 명령을 사용합니다.

```bash
python -m pytest tests/unit -q -p no:cacheprovider --basetemp=.harness/v1/2026-10-29-student-smoke/scratch/unique-run
python scripts/train/smoke_student.py --output-dir .harness/v1/2026-10-29-student-smoke/scratch/outputs/tiny-random-check
python scripts/train/smoke_student.py --output-dir .harness/v1/2026-10-29-student-smoke/scratch/outputs/native-input-check \
  --processor-dir checkpoints/students/qwen3_5_0_8b
```

`--basetemp`의 부모 디렉터리를 먼저 만들고 아직 존재하지 않는 실행 경로를 지정하십시오.
마지막 명령은 고정 revision의 tokenizer/config/processor만 로컬에 준비하면 되며 모델
가중치는 사용하지 않습니다. 실제 tokenizer vocabulary의 최대 ID와 embedding 크기,
image placeholder ID, 4-image 입력·backward·save/load를 축소 native Qwen 모델에서 검사합니다.
축소 검증 모델은 무작위 가중치이며 사전학습된 0.8B의 검증이 아닙니다.

실제 0.8B pretrained 파일은 공식 Hub tree 기준 1,746,942,600 bytes입니다.
대용량 다운로드와 전체 모델 검증은 저장소 AGENTS.md의 사전 확인 대상입니다.
사용자는 이번 작업에서 모델 가중치 검토를 추후로 미루고 Hugging Face에서 불러오는
방식을 검토하기로 했습니다. 따라서 이번에는 가중치를 다운로드하거나 실제 0.8B 검증을
실행하지 않았습니다. `QwenStudent.from_pretrained(StudentConfig(), local_files_only=False)`는
추후 명시적으로 실행할 Hub 로딩 인터페이스이며 고정 revision을 사용합니다.
기본값은 `local_files_only=True`이므로 의도하지 않은 다운로드가 발생하지 않습니다.
실제 실행 시 모델 가중치가 Hub cache로 다운로드된다는 점, cache 위치/용량·오프라인
재사용·라이선스·인증 정책을 함께 검토해야 합니다. 현재의 로컬 processor 경로에는
tokenizer·config·전처리 설정·라이선스 약 23MB만 있고 모델 가중치는 없습니다.

실제 가중치·real RLDS batch·GPU FFT/Student 학습·고정 T1 cache·KD·성공률·Jetson 실측은
이 CPU 계약 검증에 포함되지 않습니다.
