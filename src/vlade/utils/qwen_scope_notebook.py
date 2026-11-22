"""Generate the native schema-2 QwenVLA trainable-scope comparison notebook."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap


def _cell(kind: str, identifier: str, source: str) -> dict:
    value = {"cell_type": kind, "id": identifier, "metadata": {},
             "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def write_notebook(path: Path) -> Path:
    """Write an output-free notebook; model/trainer implementations stay in Git."""
    cells = [
        _cell("markdown", "overview", """
        # QwenVLA S0 · 백본 전체 학습 / GT·KD 비교

        **기본값:** schema 2 S0에서 시작, 백본까지 **full 학습**, GT-only 100 optimizer step.
        `VARIANTS`를 `both`로 바꾸면 같은 S0·데이터·학습 범위·예산에서 GT-only와 GT+KD를 순서대로 실행합니다.
        `top_layers` 또는 기존 `policy` 동결 방식도 선택할 수 있습니다.

        **셀 순서:** 설정 → Git 소스 → HF 인증 → 환경 → 실행 계약 → 입력 검사 → 1-step 파일럿 → 본 학습 → CSV.
        파일럿과 본 학습은 별도 셀입니다. 전체 모델 학습은 A100에서 먼저 파일럿을 확인하십시오.
        100스텝은 비교 시작 예산이며, 단일 시연/롤아웃 성공이나 충분한 수렴을 보장하지 않습니다.

        새 코드가 배포된 Git ref가 필요합니다. 미배포 소스에서는 입력 다운로드 전에 기능 검사가 실패합니다.
        이 노트북은 코드를 내장 패치하거나 schema를 변환하지 않습니다. 재개할 때 최초 SOURCE_COMMIT과 RUN_GROUP을 그대로 사용합니다.
        기존 동결 GT·KD 실행의 optimizer checkpoint를 여기서 이어 쓰지 않습니다.
        """),
        _cell("code", "settings", """
        from pathlib import Path
        import copy, csv, datetime, json, os, re, signal, statistics, subprocess, sys, time, uuid
        sys.dont_write_bytecode = True
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

        SOURCE_REF = "dev/fish13031e"  #@param {type:"string"}
        REPOSITORY = 'https://github.com/eastha10/VLADE.git'
        WORK = Path('/content/vlade_qwen_unfrozen')
        RUN_GROUP = globals().get('RUN_GROUP') or ('qwen-scope-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        VARIANTS = "gt_only"  #@param ["gt_only", "gt_kd", "both"]
        TRAINABLE_SCOPE = "full"  #@param ["full", "top_layers", "policy"]
        TOP_LAYER_COUNT = 4  #@param {type:"integer"}
        MAX_STEPS = 100  #@param {type:"integer"}
        SAVE_STEPS = 25  #@param {type:"integer"}
        ACCUMULATION_STEPS = 8  #@param {type:"integer"}
        VALIDATION_QUERIES = 32  #@param {type:"integer"}
        LEARNING_RATE = 2e-5  #@param {type:"number"}
        KD_WEIGHT = 1.0  #@param {type:"number"}
        SEED = 42
        REGISTRY_REVISION = '65799b213837eeb9b931a08d8fe6625a5fce43e4'
        S0_ASSET, T1_ASSET, DATA_ASSET = 'S0', 'T1', 'prepared_d_seed42'
        HF_REPO = 'VLADE-konkuk/VLADE-checkpoints'
        # 새 Colab 세션에서 재개할 때는 이전 RUN_GROUP과 SOURCE_COMMIT을 사용합니다.
        # Optional HF restore: fill only the selected variant's original run/revision.
        RESUME_HF = {'gt_only': {'run': '', 'revision': ''}, 'gt_kd': {'run': '', 'revision': ''}}
        SELECTED = ['gt_only', 'gt_kd'] if VARIANTS == 'both' else [VARIANTS]
        assert VARIANTS in ('gt_only', 'gt_kd', 'both') and TRAINABLE_SCOPE in ('full', 'top_layers', 'policy')
        assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', RUN_GROUP)
        assert type(MAX_STEPS) is int and MAX_STEPS >= 1 and 1 <= SAVE_STEPS <= MAX_STEPS
        assert type(ACCUMULATION_STEPS) is int and ACCUMULATION_STEPS >= 1
        assert VALIDATION_QUERIES >= 1 and LEARNING_RATE > 0 and KD_WEIGHT >= 0
        assert TRAINABLE_SCOPE != 'top_layers' or (type(TOP_LAYER_COUNT) is int and TOP_LAYER_COUNT >= 1)
        WORK.mkdir(parents=True, exist_ok=True)
        print('RUN_GROUP:', RUN_GROUP, '| scope:', TRAINABLE_SCOPE, '| variants:', SELECTED)
        """),
        _cell("code", "source", """
        if re.fullmatch(r'[0-9a-f]{40}', SOURCE_REF):
            SOURCE_COMMIT = SOURCE_REF
        else:
            refs = subprocess.check_output(['git', 'ls-remote', REPOSITORY, SOURCE_REF], text=True).splitlines()
            if len(refs) != 1:
                raise ValueError('하나의 Git branch 또는 40자리 commit SHA를 지정하십시오.')
            SOURCE_COMMIT = refs[0].split()[0]
        assert re.fullmatch(r'[0-9a-f]{40}', SOURCE_COMMIT)
        CODE = Path('/content/vlade_sources') / SOURCE_COMMIT
        if any((key == 'vlade' or key.startswith('vlade.')) and getattr(module, '__file__', None)
               and not Path(module.__file__).resolve().is_relative_to(CODE)
               for key, module in list(sys.modules.items())):
            raise RuntimeError('다른 VLADE 소스가 로드되어 있습니다. 런타임을 재시작하십시오.')
        if not CODE.exists():
            subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none', REPOSITORY, str(CODE)], check=True)
            subprocess.run(['git', '-C', str(CODE), 'fetch', '--depth', '1', 'origin', SOURCE_COMMIT], check=True)
            subprocess.run(['git', '-C', str(CODE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
        sys.path[:0] = [str(CODE / 'src'), str(CODE)]
        from vlade.utils.source_identity import source_identity
        assert source_identity(CODE, require_clean=True)['git_commit'] == SOURCE_COMMIT
        required = ['src/vlade/training/scopes.py', 'src/vlade/utils/timing.py']
        if not all((CODE / name).is_file() for name in required):
            raise RuntimeError('선택한 Git 소스에 백본 학습/시간 계측 변경이 없습니다. 변경을 배포한 ref를 지정하십시오.')
        trainer_source = (CODE / 'src/vlade/training/qwen_vla_task.py').read_text()
        if 'configure_trainable_scope' not in trainer_source or 'RunTimings' not in trainer_source:
            raise RuntimeError('새 학습 범위·시간 계측이 연결된 Git commit이 필요합니다.')
        model_settings = json.loads((CODE / 'configs/models/student/qwen_vla_0_8b.json').read_text())
        assert model_settings['student']['schema_version'] == 2
        print('SOURCE_COMMIT:', SOURCE_COMMIT)
        """),
        _cell("code", "authentication", """
        # Colab Secrets에 비공개 HF repo 읽기/쓰기 권한의 HF_TOKEN을 연결합니다.
        try:
            from google.colab import userdata
        except ImportError:
            if not os.environ.get('HF_TOKEN'):
                raise RuntimeError('Colab 외 환경에서는 HF_TOKEN 환경 변수를 연결하십시오.')
        else:
            os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
        from vlade.runtime.environment import runtime_environment, prepare
        ENV = runtime_environment(WORK)
        ENV.update(HF_HUB_CACHE=str(WORK / 'cache/huggingface/hub'),
                   HF_XET_CACHE=str(WORK / 'cache/huggingface/xet'),
                   PIP_CACHE_DIR=str(WORK / 'cache/pip'))
        ENV['PYTHONPATH'] = os.pathsep.join([str(CODE / 'src'), str(CODE)])
        ENV['VLADE_HUB_REPO'] = HF_REPO
        REGISTRY = WORK / 'registries' / (REGISTRY_REVISION + '.json')
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        os.environ.update({key: ENV[key] for key in ('HF_HOME', 'HF_TOKEN_PATH', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'PIP_CACHE_DIR')})
        ENV['VLADE_ASSET_REGISTRY'] = str(REGISTRY)
        print('HF 인증을 연결했습니다. 토큰 값은 출력하지 않습니다.')
        """),
        _cell("code", "environments", """
        STUDENT_PYTHON = prepare(CODE, WORK, 'student')
        TEACHER_PYTHON = prepare(CODE, WORK, 'ava') if 'gt_kd' in SELECTED else None
        # 새 프로세스의 고정 HF client와 런타임 cache를 사용합니다.
        registry_program = 'from pathlib import Path; import sys; from vlade.storage.assets import refresh_registry; refresh_registry(Path(sys.argv[1]), revision=sys.argv[2])'
        subprocess.run([str(STUDENT_PYTHON), '-B', '-c', registry_program, str(REGISTRY), REGISTRY_REVISION],
                       cwd=CODE, env=ENV, check=True)
        registry = json.loads(REGISTRY.read_text())
        for key in [S0_ASSET, DATA_ASSET] + ([T1_ASSET] if 'gt_kd' in SELECTED else []):
            if key not in registry['assets']:
                raise ValueError('고정 HF registry에 필요한 asset이 없습니다: ' + key)
        print('Student runtime:', STUDENT_PYTHON)
        print('Teacher runtime:', TEACHER_PYTHON or 'GT-only: 사용 안 함')
        """),
        _cell("code", "run-contract", """
        COMMON = {'model_config': 'configs/models/student/qwen_vla_0_8b.json',
                  'data': 'hfasset://' + DATA_ASSET, 'initialization': 'student_checkpoint',
                  'initial_checkpoint': 'hfasset://' + S0_ASSET, 'seed': SEED,
                  'max_steps': MAX_STEPS, 'accumulation_steps': ACCUMULATION_STEPS,
                  'learning_rate': LEARNING_RATE, 'weight_decay': 0.01,
                  'save_steps': SAVE_STEPS, 'validation_queries': VALIDATION_QUERIES,
                  'trainable_scope': TRAINABLE_SCOPE,
                  'top_layer_count': TOP_LAYER_COUNT if TRAINABLE_SCOPE == 'top_layers' else 0}
        SETTINGS = {'gt_only': copy.deepcopy(COMMON), 'gt_kd': copy.deepcopy(COMMON)}
        SETTINGS['gt_kd']['distillation'] = {'teacher_mode': 'inference',
            'teacher_checkpoint': 'hfasset://' + T1_ASSET, 'teacher_device': 'cuda:0',
            'teacher_timeout_seconds': 600, 'task_weight': 1.0, 'action_kd_weight': KD_WEIGHT}
        assert {k: v for k, v in SETTINGS['gt_kd'].items() if k != 'distillation'} == SETTINGS['gt_only']
        contract = {'source_commit': SOURCE_COMMIT, 'registry_revision': REGISTRY_REVISION,
                    'run_group': RUN_GROUP, 'variants': SELECTED,
                    'settings': {name: SETTINGS[name] for name in SELECTED}, 'hub_repo': HF_REPO}
        RUN_ROOT = WORK / 'runs' / RUN_GROUP
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        contract_path = RUN_ROOT / 'run-contract.json'
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != contract:
                raise ValueError('기존 실행과 소스/설정이 다릅니다. 새 RUN_GROUP을 사용하십시오.')
        else:
            contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2))
        CONFIGS, OUTPUTS = {}, {}
        for name in SELECTED:
            CONFIGS[name], OUTPUTS[name] = RUN_ROOT / (name + '.json'), RUN_ROOT / name
            CONFIGS[name].write_text(json.dumps(SETTINGS[name], indent=2))
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        """),
        _cell("code", "runner", """
        def invoke(command, env):
            logs = RUN_ROOT / 'logs'
            logs.mkdir(exist_ok=True)
            path = logs / (str(time.time_ns()) + '.log')
            process = None
            try:
                with path.open('x', encoding='utf-8') as log:
                    process = subprocess.Popen([str(item) for item in command], cwd=CODE, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                        start_new_session=(os.name == 'posix'))
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    if process.wait():
                        raise subprocess.CalledProcessError(process.returncode, command)
            finally:
                if process is not None:
                    if process.poll() is None:
                        if os.name == 'posix':
                            os.killpg(process.pid, signal.SIGTERM)
                        else:
                            process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            if os.name == 'posix':
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                            process.wait()
                    process.stdout.close()
            return path

        def variant_env(name):
            return {**ENV, 'VLADE_HUB_RUN_ID': RUN_GROUP + '-' + name}

        def latest_checkpoint(name):
            paths = sorted(OUTPUTS[name].glob('step-*'))
            if not paths:
                return None
            path = paths[-1]
            if not (path / 'COMPLETE').is_file():
                raise ValueError('불완전 checkpoint가 있습니다. 원본을 보존하고 확인하십시오.')
            return path

        def train_to(name, stop):
            previous = latest_checkpoint(name)
            if previous is not None and int(previous.name.removeprefix('step-')) >= stop:
                verify = 'from pathlib import Path; import sys; from vlade.training.run_control import verify_checkpoint; print(verify_checkpoint(Path(sys.argv[1]), accumulation=int(sys.argv[2])))'
                invoke([STUDENT_PYTHON, '-B', '-c', verify, previous, ACCUMULATION_STEPS], variant_env(name))
                print(name, '요청한 step까지 이미 저장되어 있습니다.')
                return
            command = [STUDENT_PYTHON, '-B', '-m', 'vlade.training.qwen_vla_task',
                       '--config', CONFIGS[name], '--output', OUTPUTS[name], '--stop-after', stop]
            if previous is not None:
                command += ['--resume', previous]
            if name == 'gt_kd':
                command += ['--teacher-python', TEACHER_PYTHON]
            invoke(command, variant_env(name))
        """),
        _cell("code", "preflight", """
        for name in SELECTED:
            restore = RESUME_HF[name]
            if bool(restore['run']) != bool(restore['revision']):
                raise ValueError('HF 재개 run과 불변 revision을 함께 지정하십시오.')
            if restore['run']:
                assert re.fullmatch(r'[0-9a-f]{40}', restore['revision'])
                expected = RUN_GROUP + '-' + name + '/' + name
                if restore['run'] != expected:
                    raise ValueError('재개 RUN_GROUP/output 이름이 기존 Hub 실행과 다릅니다.')
                if latest_checkpoint(name) is None:
                    invoke([STUDENT_PYTHON, '-B', CODE / 'scripts/train/hub_checkpoints.py',
                            'download', '--repo', HF_REPO, '--run', restore['run'],
                            '--revision', restore['revision'], '--alias', 'last', '--output', OUTPUTS[name]], variant_env(name))
            invoke([STUDENT_PYTHON, '-B', CODE / 'scripts/train/hub_checkpoints.py',
                    'check', '--output', OUTPUTS[name]], variant_env(name))
        # 기존 source/normalization/schema 검증을 유지합니다. 본문 다운로드가 필요한 경우 이 셀에서 HF 캐시에 준비합니다.
        assets = [S0_ASSET, DATA_ASSET] + ([T1_ASSET] if 'gt_kd' in SELECTED else [])
        check = 'from vlade.storage.assets import resolve_input; ' + '; '.join(
            'print(resolve_input(' + repr('hfasset://' + key) + '))' for key in assets)
        invoke([STUDENT_PYTHON, '-B', '-c', check], ENV)
        """),
        _cell("code", "pilot", """
        # 실제 optimizer 1스텝 + 검증 + checkpoint/Hub 저장. full GT+KD의 실제 메모리도 이 셀에서 확인합니다.
        for name in SELECTED:
            train_to(name, 1)
        print('파일럿 완료. 로그의 trainable_scope/full, backbone_trainable_parameters > 0, Hub pending 여부를 확인하십시오.')
        """),
        _cell("code", "train", """
        for name in SELECTED:
            if latest_checkpoint(name) is None:
                raise RuntimeError('먼저 1-step 파일럿 셀을 실행하십시오.')
            train_to(name, MAX_STEPS)
        print('학습 명령 완료. final은 MAX_STEPS 완료 시에만 표시됩니다. Hub pending은 아래에서 재확인합니다.')
        """),
        _cell("code", "timing-csv", """
        TIMING_FILES = sorted(path for name in SELECTED for path in OUTPUTS[name].glob('timings-*.csv'))
        print('시간 CSV:', *map(str, TIMING_FILES), sep='\\n')
        for path in TIMING_FILES:
            with path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            groups = {}
            for row in rows:
                if row['status'] != 'ok' or row['is_warmup'] == 'True' or row['replay'] == 'True':
                    continue
                groups.setdefault((row['event'], row['phase']), []).append(float(row['seconds']))
            print(path.name, {str(key): {'n': len(values), 'total_s': sum(values), 'median_s': statistics.median(values)}
                              for key, values in groups.items()})
        for name in SELECTED:
            pending = OUTPUTS[name] / 'hub_pending.json'
            print(name, 'Hub pending:', json.loads(pending.read_text()) if pending.exists() else '파일 없음')
        print('중첩 event는 합산하지 않습니다. 실제 T1 시간은 gt_kd를 실행했을 때만 생성됩니다.')
        """),
        _cell("code", "download-csv", """
        DOWNLOAD_CSV = False  #@param {type:"boolean"}
        if DOWNLOAD_CSV:
            import zipfile
            export = RUN_ROOT / 'timing-csv.zip'
            with zipfile.ZipFile(export, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in TIMING_FILES:
                    archive.write(path, path.relative_to(RUN_ROOT).as_posix())
            from google.colab import files
            files.download(str(export))
        """),
    ]
    notebook = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
