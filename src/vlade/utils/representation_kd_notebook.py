"""Existing-session Colab client: immutable new source, installed runtimes/cache."""
from __future__ import annotations

import json
from pathlib import Path

from vlade.utils.colab_notebooks import cell


CONFIG_CELL = r'''
from pathlib import Path
import datetime, uuid

# 이 구현이 포함되어 원격에서 읽을 수 있는 40자리 Git commit을 지정합니다.
# 현재 CODE/SOURCE_COMMIT와 실행 중인 소스는 덮어쓰지 않습니다.
S2_SOURCE_COMMIT = ""
S2_WORKSPACE = Path(WORKSPACE)  # 현재 세션에서 설치/다운로드한 기존 workspace
S2_RUN_GROUP = globals().get('S2_RUN_GROUP') or (
    's2-representation-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    + '-' + uuid.uuid4().hex[:8])
# 실제 S1과 비교할 때 S1 workspace/runs/<group>/run-contract.json을 지정합니다.
# 공란이면 저장소의 S1 기본 설정과 동일한 조건을 사용합니다.
S1_RUN_CONTRACT = ""
REPRESENTATION_WEIGHT = 1.0  # 초기 실험값이며 검증된 최적값이 아닙니다.
S2_RESUME_REVISIONS = {}  # 재개 예: {'S2': '40자리 HF commit SHA'}
S2_RECOVER_PREFLIGHT_FROM = ''  # 학습 전 실패한 run의 소스 전환에만 이전 commit을 지정합니다.
ACTION = 'plan'  # plan → preflight → pilot → train, 전송 실패 시 retry
'''

RUN_CELL = r'''
import json, os, re, signal, subprocess, sys
from pathlib import Path
from google.colab import userdata

if not re.fullmatch(r'[0-9a-f]{40}', S2_SOURCE_COMMIT):
    raise ValueError('이 구현이 포함된 원격 Git commit 40자리를 S2_SOURCE_COMMIT에 지정하십시오.')
if ACTION not in {'plan', 'preflight', 'pilot', 'train', 'retry'}:
    raise ValueError('기존 세션은 setup/설치/런타임 재시작 없이 실행합니다.')
S2_CODE = Path('/content/vlade_sources') / S2_SOURCE_COMMIT
if not S2_CODE.exists():
    subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none',
                    'https://github.com/eastha10/VLADE.git', str(S2_CODE)], check=True)
    subprocess.run(['git', '-C', str(S2_CODE), 'fetch', '--depth', '1', 'origin', S2_SOURCE_COMMIT], check=True)
    subprocess.run(['git', '-C', str(S2_CODE), 'checkout', '--detach', S2_SOURCE_COMMIT], check=True)
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONPATH':str(S2_CODE / 'src')}
if ACTION != 'plan':
    env['HF_TOKEN'] = userdata.get('HF_TOKEN')
# vlade를 현재 커널에 다시 import하지 않습니다. 별도 프로세스에서 새 소스를 사용합니다.
CLIENT = r"""
import copy, json, re, sys
from pathlib import Path
from vlade.research.distillation import main, load_settings
from vlade.runtime.environment import locate
from vlade.utils.source_identity import source_identity

code, work, commit, group, action, s1_contract, weight, revisions = sys.argv[1:9]
recover_from = sys.argv[9] if len(sys.argv) > 9 else ''
code, work = Path(code), Path(work)
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', group):
    raise ValueError('S2_RUN_GROUP은 영문·숫자·밑줄·하이픈으로 지정하십시오.')
if source_identity(code, require_clean=True)['git_commit'] != commit:
    raise ValueError('S2_CODE와 S2_SOURCE_COMMIT이 일치하지 않습니다.')
folder = work / 'runs' / group
config = folder / 'representation-config.json'
base = json.loads((code / 'configs/experiments/qwen_vla_representation_kd.json').read_text())
if s1_contract:
    contract = json.loads(Path(s1_contract).read_text())
    s1 = contract['variants']['gt_kd']
    kd = s1['distillation']
    if (kd.get('teacher_plan_max_new_tokens', 0) or kd.get('representation') is not None
            or kd.get('teacher_mode') != 'inference' or contract['workflow']['variant'] != 'gt_kd'):
        raise ValueError('온라인 Action-only S1의 원본 run-contract.json이 필요합니다.')
    rep = base['distillation']['representation']
    base = copy.deepcopy(s1)
    base['workflow'] = copy.deepcopy(contract['workflow'])
    base['distillation']['representation'] = rep
    print('S1 조건 상속:', s1_contract, flush=True)
elif config.is_file():
    base = json.loads(config.read_text())
    print('기존 S2 설정·입력 pin 재사용:', config, flush=True)
else:
    print('비교 기준: 저장소 S1 기본 설정. 실제 S1 override/pin은 S1_RUN_CONTRACT로 상속하십시오.', flush=True)
base['distillation']['representation']['weight'] = float(weight)
if action != 'plan':
    # locate는 기존 설치의 규격과 존재만 검증합니다. prepare/install은 호출하지 않습니다.
    locate(code, work, 'student')
    locate(code, work, 'ava')
# 실행 설정은 현재 run 디렉터리에만 두고, 원래 S1 파일을 수정하지 않습니다.
if action == 'plan':
    workflow, variants = load_settings(base, {})
    print(json.dumps({'workflow': workflow, 'variants': variants, 'run_group':group,
        'output':str(folder / 'S2'), 'hub_repo':'VLADE-konkuk/VLADE-checkpoints',
        'hub_runs':{'S2':'runs/' + group + '/S2'}}, indent=2), flush=True)
else:
    folder.mkdir(parents=True, exist_ok=True)
    content = json.dumps(base, indent=2) + '\n'
    if config.exists() and config.read_text() != content:
        raise ValueError('S2 설정이 변경되었습니다. 새 S2_RUN_GROUP을 지정하십시오.')
    config.write_text(content)
    if recover_from:
        if action != 'preflight' or json.loads(revisions):
            raise ValueError('소스 복구는 학습 전 preflight에서만 실행합니다.')
        from vlade.research.representation_recovery import recover_preflight_source
        recover_preflight_source(code, work, group, recover_from)
    main(['--config', str(config), '--workspace', str(work), '--run-group', group,
          '--action', action, '--resume-revisions', revisions])
"""
def stream_s2(command, *, cwd, env):
    # Notebook kernels need a PIPE and explicit print; inherited file descriptors
    # can hide the real child traceback and leave only CalledProcessError.
    process = subprocess.Popen(command, cwd=cwd,
        env={**env, 'PYTHONUNBUFFERED':'1'}, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
        bufsize=1, start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGINT)  # only this launch's process group
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        process.stdout.close()
    if status:
        raise RuntimeError(f'S2 {ACTION} 실패 (exit {status}). 바로 위 자식 프로세스 로그를 확인하십시오.')

stream_s2([sys.executable, '-B', '-u', '-c', CLIENT, str(S2_CODE), str(S2_WORKSPACE),
           S2_SOURCE_COMMIT, S2_RUN_GROUP, ACTION, str(S1_RUN_CONTRACT),
           str(REPRESENTATION_WEIGHT), json.dumps(S2_RESUME_REVISIONS),
           str(globals().get('S2_RECOVER_PREFLIGHT_FROM', ''))],
          cwd=S2_CODE, env=env)
S2_RECOVER_PREFLIGHT_FROM = ''
print('S2 source:', S2_SOURCE_COMMIT, '| run:', S2_RUN_GROUP)
'''


RECOVERY_CELL = r'''
# 아래 셀 하나를 기존 Colab 세션에 복사합니다. 기존 두 셀을 다시 실행할 필요가 없습니다.
import json, re, subprocess
from pathlib import Path

S2_FIX_COMMIT = ''  # 원격 게시된 수정 소스의 40자리 commit을 지정합니다.
if not re.fullmatch(r'[0-9a-f]{40}', S2_FIX_COMMIT):
    raise ValueError('S2_FIX_COMMIT에 수정 소스의 불변 commit을 지정하십시오.')
_s2_old_commit = '49a3ec6b07a18b761d32670c2c05cd165e08cdc7'
_s2_group = 's2-repr-20261125T081316Z-88f7798e'
_s2_work = Path(globals().get('S2_WORKSPACE', '/content/vlade_distillation')).resolve()
if _s2_work != Path('/content/vlade_distillation'):
    raise ValueError('실패한 실행의 기존 S2_WORKSPACE가 아닙니다.')
if globals().get('S2_RUN_GROUP', _s2_group) != _s2_group:
    raise ValueError('다른 S2_RUN_GROUP이 활성화되어 있습니다.')
if globals().get('S2_SOURCE_COMMIT', _s2_old_commit) not in {_s2_old_commit, S2_FIX_COMMIT}:
    raise ValueError('기존 S2_SOURCE_COMMIT을 확인하십시오.')
if globals().get('S2_RESUME_REVISIONS', {}):
    raise ValueError('이 셀은 checkpoint 재개용이 아니라 실패한 preflight 복구용입니다.')
_s2_config = json.loads((_s2_work / 'runs' / _s2_group / 'representation-config.json').read_text())
_s2_inputs = json.loads((_s2_work / 'runs' / _s2_group / 'inputs.json').read_text())
for _key, _revision in {
    'S0': '380f8fb1bd828d42e7f396d6979d699b0b7d0cff',
    'T1': '1fdc672416f2129624b22022e6977345a3b429e3',
    'prepared_d_seed42': '6d2542577647a9e291554524af4e1dd0928e9df6',
}.items():
    if _s2_inputs['assets'][_key]['revision'] != _revision:
        raise ValueError('기존 HF 입력 pin이 다릅니다: ' + _key)
_s2_code = Path('/content/vlade_sources') / S2_FIX_COMMIT
if not _s2_code.exists():
    subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none',
                    'https://github.com/eastha10/VLADE.git', str(_s2_code)], check=True)
    subprocess.run(['git', '-C', str(_s2_code), 'fetch', '--depth', '1', 'origin', S2_FIX_COMMIT], check=True)
    subprocess.run(['git', '-C', str(_s2_code), 'checkout', '--detach', S2_FIX_COMMIT], check=True)
if subprocess.check_output(['git', '-C', str(_s2_code), 'rev-parse', 'HEAD'], text=True).strip() != S2_FIX_COMMIT:
    raise ValueError('수정 소스 경로의 commit이 다릅니다.')
_s2_notebook = json.loads((_s2_code / 'experiments/colab_s2_representation_kd_existing_session.ipynb').read_text())
_s2_run_cell = ''.join(next(c for c in _s2_notebook['cells'] if c['id'] == 'run')['source'])
# 새 실행 셀이 CLIENT까지 다시 정의하므로 옛 CLIENT와 새 경로가 섞이지 않습니다.
# 커널의 CODE/SOURCE_COMMIT 및 기존 환경·cache는 유지합니다.
S2_SOURCE_COMMIT, S2_CODE = S2_FIX_COMMIT, _s2_code
S2_WORKSPACE, S2_RUN_GROUP = _s2_work, _s2_group
S1_RUN_CONTRACT = ''  # 저장된 S2 설정에 이미 반영된 S1 조건을 그대로 사용합니다.
REPRESENTATION_WEIGHT = _s2_config['distillation']['representation']['weight']
S2_RESUME_REVISIONS = {}
S2_RECOVER_PREFLIGHT_FROM = _s2_old_commit
ACTION = 'preflight'
exec(compile(_s2_run_cell, str(_s2_code / 's2-existing-session-run'), 'exec'), globals())
# 성공 후 다음 별도 셀: ACTION = 'pilot'; exec(_s2_run_cell, globals())
# pilot의 저장·HF 게시 성공을 확인한 뒤: ACTION = 'train'; exec(_s2_run_cell, globals())
'''


def write_notebook(path: Path) -> Path:
    """Write cells to paste into the already running A100 notebook."""
    cells = [cell("markdown", """
        # S2 · T1 중간 표현 + Action Distillation · 기존 A100 세션

        아래 두 셀을 기존 노트북에서 실행합니다. 기존 `WORKSPACE`의 Student/AVA 환경과
        HF 캐시를 사용하며 설치·런타임 재시작은 하지 않습니다. 새 Git commit의 소스를
        별도 경로와 자식 프로세스에서 사용하므로 커널에 로드된 옛 모듈과 섞이지 않습니다.
        기존 작업의 GPU 프로세스가 종료된 뒤 실행하십시오. 다른 학습을 강제로 종료하지 않습니다.

        이 코드가 포함된 원격 commit이 필요합니다. 로컬 수정만으로 Colab에 반영되지 않으며,
        이 노트북은 Git push/Drive 업로드를 수행하지 않습니다. `S2_SOURCE_COMMIT`을 지정하고
        `S1_RUN_CONTRACT`로 실제 S1의 S0/T1 pin·데이터·학습 조건을 상속하십시오.
        공란이면 저장소 S1 기본값을 비교 기준으로 사용합니다.

        `ACTION`: **plan → preflight → pilot → train**. Plan 생성 probe는 실행하지 않습니다.
        pilot은 원래 누적 8회/목표 schedule을 유지하는 1 optimizer step이며 이후 train으로
        동일 run을 재개합니다. 중간·best·목표 final은 기존 private HF Hub 경로를 사용합니다.
        재개 시 같은 소스/설정/run을 유지하고 HF에서 복구할 때 불변 revision을 지정합니다.

        손실은 GT L1 + λa Action L1 + λr normalized representation MSE입니다.
        마지막에서 두 번째 hidden state의 56개 슬롯을 (8시점×7액션 차원)으로 정렬하고
        Qwen→Llama2 학습 가능 projection을 적용합니다. 차원·층은 실제 config로 확인합니다.
        validation/best는 S1과 동일한 plan-free GT L1이며 projection은 추론에 필요 없습니다.
        실제 A100 메모리·학습 성공·rollout 성능은 pilot/평가로 확인해야 합니다.
    """, "purpose"), cell("code", CONFIG_CELL, "config"), cell("code", RUN_CELL, "run"),
        cell("markdown", """
        ## 2026-11-25 preflight 실패 실행 복구

        `49a3ec6...`의 text config 오류로 중단한 `s2-repr-20261125T081316Z-88f7798e`만
        아래 셀 하나로 복구합니다. 위 설정/실행 셀을 먼저 실행하지 마십시오.
        수정 commit을 지정하면 기존 workspace·run group·세 HF 입력 pin·effective settings를
        유지하고 새 경로의 실행 셀과 CLIENT를 함께 적재합니다. 설치나 A100 재연결은 없습니다.
        로컬/원격 학습 결과가 있으면 소스 전환을 거절하며, 이전 preflight 계약 원문을 보존합니다.
        이 셀은 preflight까지만 실행합니다. 실제 GPU pilot과 HF 저장 성공은 별도로 확인하십시오.
        """, "recovery-purpose"), cell("code", RECOVERY_CELL, "recover-preflight")]
    value = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
