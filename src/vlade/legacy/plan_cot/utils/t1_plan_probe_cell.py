"""An isolated, inference-only recovery cell for an already running Colab session."""
from __future__ import annotations

import json
from pathlib import Path

from vlade.utils.colab_notebooks import cell

PROBE_SOURCE = r'''
import contextlib, json, sys, uuid
from pathlib import Path
import torch
from vlade.storage.assets import resolve_input
from vlade.research.registry import prepare_registry
from vlade.training.teacher_inference import teacher_identity
from vlade.training.fixed_cache import SequentialTeacher
from vlade.data.episode_data import EpisodeData
from vlade.legacy.plan_cot.distillation import teacher_plan

code, work = map(Path, sys.argv[1:3])
revision, commit = sys.argv[3:5]
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 CUDA GPU required'
registry = prepare_registry(code, work, revision)
import os
os.environ['VLADE_ASSET_REGISTRY'] = str(registry)
print('HF cache verification; already downloaded weights/data are reused.', flush=True)
data = EpisodeData(resolve_input('hfasset://prepared_d_seed42'))
checkpoint, identity = teacher_identity('hfasset://T1', data)

# Compatibility only inside this inference subprocess; source/checkpoint bytes stay intact.
strict_history_check = teacher_plan.require_plan_supervision
def optional_history_check(metadata):
    if metadata.get('plan_supervision') is not None:
        return strict_history_check(metadata)
    return None
teacher_plan.require_plan_supervision = optional_history_check

output = work / 'runs' / ('t1-plan-generation-probe-' + uuid.uuid4().hex[:12])
output.mkdir(parents=True, exist_ok=False)
report = output / 'probe.jsonl'
policy = None
with report.open('x', encoding='utf-8') as stream:
    def record(row):
        text = json.dumps(row, ensure_ascii=False)
        print(text, file=stream, flush=True)
        print(text, flush=True)
    try:
        record({'status':'identity', 'source_commit':commit, 'teacher':identity,
                'execution':'inference_only_optional_plan_history_v1',
                'plan_supervised':identity.get('plan_supervision') is not None,
                'scope':'generated_text_only; semantic quality needs review'})
        print('Loading the frozen T1 once. Student and optimizer are not loaded.', flush=True)
        policy = SequentialTeacher(checkpoint, device='cuda:0',
                                   temporal_feature_layer=identity['temporal_feature_layer'])
        good = bad = 0
        for split in ('train', 'validation'):
            first = next(data.episodes(split), None)
            if first is None or not first[1]:
                raise ValueError('Missing nonempty split: ' + split)
            policy.reset()
            for sample in first[1][:2]:
                row = {key:sample[key] for key in ('episode_id','timestep','instruction')}
                row.update(split=split, action_status='not_requested_plan_only')
                try:
                    plan = policy.plan(sample, data, max_new_tokens=48)
                    record({**row, 'status':'generated', 'plan':plan, 'semantic_quality':'needs_review'})
                    good += 1
                except teacher_plan.InvalidTeacherPlan as error:
                    record({**row, 'status':'plan_error', 'error':str(error)})
                    bad += 1
        record({'status':'complete' if not bad else 'failed', 'generated':good, 'plan_errors':bad,
                'training_started':False, 'semantic_quality':'not_evaluated'})
    except Exception as error:
        record({'status':'failed', 'error_type':type(error).__name__, 'error':str(error),
                'training_started':False})
        raise
    finally:
        teacher_plan.require_plan_supervision = strict_history_check
        del policy
        torch.cuda.empty_cache()
        print('PROBE_REPORT:', report, flush=True)
'''


def recovery_cell() -> str:
    """Return a self-contained cell that uses the existing CODE/WORKSPACE and AVA venv."""
    setup = '''from pathlib import Path
import os, subprocess
from google.colab import userdata
from vlade.runtime.environment import locate, runtime_environment

work = Path(WORKSPACE)
ava_python = locate(CODE, work, 'ava')  # reuse the installed environment
env = runtime_environment(work)
env.update(PYTHONPATH=str(CODE / 'src'), HF_TOKEN=userdata.get('HF_TOKEN'))
registry_revision = globals().get('REGISTRY_REVISION', '195d32e54c7f48960411c32963d91746183de488')
'''
    return setup + '\nPROBE_SOURCE = ' + repr(PROBE_SOURCE) + '''
subprocess.run([str(ava_python), '-B', '-c', PROBE_SOURCE,
                str(CODE), str(work), registry_revision, SOURCE_COMMIT],
               cwd=CODE, env=env, check=True)
'''


def write_notebook(path: Path) -> Path:
    """Write a recovery notebook whose one code cell is pasted into the current session."""
    cells = [cell('markdown', '''
        # 현재 A100 세션에서 기존 T1 Plan 생성 확인

        **런타임을 끊거나 재시작하지 마십시오.** 아래 코드 셀을 현재 실행 중인 노트북의
        설정 셀 뒤에 복사해 실행합니다. 기존 CODE, SOURCE_COMMIT, WORKSPACE와 설치된
        AVA 환경·HF 캐시를 그대로 사용합니다. 새 런타임을 여는 용도의 노트북이 아닙니다.

        기존 커밋의 잘못된 Plan 학습 이력 필수 조건을 이 추론 자식 프로세스에서만 수정합니다.
        HF 무결성·checkpoint 봉인·실제 텍스트의 EOS/빈 출력/특수 토큰 검사는 유지합니다.
        checkpoint metadata나 고정 소스 파일을 바꾸지 않으며 학습 이력을 만들지 않습니다.

        T1을 한 번 로드해 Train·Validation 첫 Episode에서 각각 최대 2개 현재 관측의 Plan을
        생성합니다. Student·optimizer·액션 추론·학습·HF 게시는 실행하지 않습니다.
        생성 오류도 원문/token ID와 함께 기록합니다. 결과가 나와도 의미 품질은 원문으로 확인해야 합니다.
        이 진단 결과를 S2 probe 통과로 자동 등록하지 않습니다. 보고서는 기존 workspace/runs 아래에 저장합니다.
    ''', 'purpose'), cell('code', recovery_cell(), 'existing-session-probe')]
    value = {'cells':cells, 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
             'nbformat':4, 'nbformat_minor':5}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\r\n')
    return path
