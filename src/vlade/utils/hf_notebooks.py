"""Use the same pinned HF inputs in all newly generated Colab notebooks."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

BOOTSTRAP = r'''
# Git supplies source. HF supplies every external model/data input.
ASSET_CACHE = Path('/content/vlade_hf_assets')
ASSET_WORK_ROOT = Path('/content/vlade_asset_work')
RUNS_ROOT = Path('/content/vlade_runs')
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
_HF_FETCHED = set()
ENV.update(VLADE_ASSET_CACHE=str(ASSET_CACHE),
           VLADE_ASSET_REGISTRY=str(CODE/'configs/storage/hf_assets.lock.json'),
           HF_HOME='/content/vlade_hf_home', HF_TOKEN_PATH='/content/vlade_hf_token')
os.environ.update({key: ENV[key] for key in ('VLADE_ASSET_CACHE','VLADE_ASSET_REGISTRY')})

def hf_input(key, python=None, *, fetch=True):
    from vlade.storage.assets import asset_record, cache_path
    record = asset_record(key, Path(ENV['VLADE_ASSET_REGISTRY']))
    target = cache_path(key, record, ASSET_CACHE)
    if fetch and (key, str(target)) not in _HF_FETCHED:
        if python is None:
            raise ValueError('Select the CPU/model Python environment before fetching HF assets')
        run([python, 'scripts/data/hf_assets.py', 'fetch', key,
             '--registry', ENV['VLADE_ASSET_REGISTRY'], '--cache', ASSET_CACHE], env=ENV)
        _HF_FETCHED.add((key, str(target)))
    return target

def hf_path(value, python=None, *, fetch=True):
    value = str(value)
    if not value.startswith('hfasset:'):
        raise ValueError('External inputs must use hfasset:// keys, not Drive/local paths')
    return hf_input(value[len('hfasset:'):].strip('/\\'), python, fetch=fetch)
'''.strip()


def refresh_suite_sources(path: Path, root: Path, source_ref: str) -> None:
    """Upgrade the local Suite notebook; do not contact any running runtime."""
    if not path.is_file():
        return
    from vlade.utils.source import stage_source
    value = json.loads(path.read_text(encoding='utf-8'))
    cells = []
    for item in value['cells']:
        identifier = item.get('id')
        source = ''.join(item['source'])
        if identifier == 'ava-suite-01':
            source = re.sub(r'(?m)^PROJECT_ROOT = .*$', "PROJECT_ROOT = Path('/content/vlade_work')", source)
            source = re.sub(r'(?m)^AUTHENTICATE_DRIVE_CACHE = .*$', '', source)
            source = source.replace('# Read-only Drive ID/version checks enable cross-runtime hash reuse. No CU API is assumed.',
                                    '# Inputs are pinned in configs/storage/hf_assets.lock.json.')
        elif identifier == 'ava-suite-03':
            source = "gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'], text=True, capture_output=True)\nprint('GPU / VRAM:', gpu.stdout.strip() or gpu.stderr.strip())\nprint('HF input preparation:', PREPARE_INPUTS)"
        elif identifier == 'ava-suite-05':
            start = source.index('SOURCE_MANIFEST_SHA =') if 'SOURCE_MANIFEST_SHA =' in source else source.index('\nimport re, tempfile')
            source = source[:start] + "\nimport re, tempfile\n" + inspect.getsource(stage_source) + '''
if PREPARE_INPUTS:
    if RUN_DIR.exists() and RESUME_CHECKPOINT is None:
        raise FileExistsError('Use a new run ID or select an explicit resume checkpoint')
    CODE = stage_source('https://github.com/eastha10/VLADE.git', Path('/content/vlade_sources'),
                        globals().get('SOURCE_REF', 'main'))
    SOURCE_COMMIT = subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip()
    PROJECT_ROOT = CODE
    for folder in ['configs','pilot4','logs','manifests','reports']:
        (RUN_DIR/folder).mkdir(parents=True,exist_ok=True)
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'tmp').mkdir(exist_ok=True)
    ENV = os.environ.copy()
    ENV.update(PYTHONPATH=str(CODE/'src'), PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled', TF_CPP_MIN_LOG_LEVEL='2',
        UV_CACHE_DIR=str(LOCAL/'uv_cache'), TMPDIR=str(LOCAL/'tmp'), XDG_CACHE_HOME=str(LOCAL/'cache'))
    control_write(RUN_DIR/'manifests/source_manifest.json', {'git_commit':SOURCE_COMMIT})
    sys.path.insert(0,str(CODE/'src'))
''' + '\n'.join('    '+line if line else '' for line in BOOTSTRAP.splitlines())
        elif identifier == 'ava-suite-06':
            source = "if PREPARE_INPUTS:\n    assert (CODE/'src/vlade/training/colab_suite_fft.py').is_file()\n"
        elif identifier == 'ava-prepare-cache':
            # The old embedded Drive receipt/cache implementation is no longer used.
            continue
        elif identifier == 'ava-suite-09':
            source = re.sub(r"    source_manifest = json.loads.*?    if not patch.is_file", '    if not patch.is_file', source, flags=re.S)
        elif identifier == 'ava-suite-11':
            source = source.replace("prep_script.write_text(PREP_WORKER_SOURCE)",
                "prep_script.write_text(\"from pathlib import Path\\nimport sys\\nfrom vlade.training.colab_suite_fft import verify_data, write\\nwrite(Path(sys.argv[4]), verify_data(Path(sys.argv[2]), Path(sys.argv[3])))\\n\")")
        elif identifier == 'ava-suite-15':
            source = re.sub(r"(?m)^            drive.flush_and_unmount.*$", '', source)
            source = source.replace('drive_readback_verified','checkpoint_readback_verified').replace('cu_estimate_excludes_unmount_flush','cu_estimate_excludes_runtime_release').replace('Drive flush', 'Checkpoint readback')
        elif identifier == 'ava-preparation-cache-notes':
            source = '입력은 Hugging Face 고정 revision에서 받습니다. 다운로드·캐시 재사용 시 파일 무결성을 검사합니다. 소스는 실행 시 확정한 Git commit을 기록합니다.'
        if item['cell_type'] == 'code':
            compile(source, str(path)+':'+str(identifier), 'exec')
        item['source'] = source.splitlines(keepends=True)
        cells.append(item)
    value['cells'] = cells
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')


def apply_hf_storage(path: Path, root: Path) -> None:
    from vlade.utils.colab_notebooks import cell
    value = json.loads(path.read_text(encoding='utf-8'))
    for item in value['cells']:
        identifier = item.get('id')
        if identifier == 'purpose':
            titles = {'01':'# 01 · Hugging Face 자료 준비', '02':'# 02 · AVA FFT · Teacher 선택 · 캐시',
                      '03':'# 03 · Qwen Task-only', '04':'# 04 · AVA 논문 설정', '05':'# 05 · Qwen 액션 증류'}
            title = {'colab_04_openvla_oft_fft.ipynb': '# 04 · OpenVLA-OFT FFT 재로딩',
                     'colab_05_ava_a100_one_step_reload.ipynb': '# 05 · AVA 단일 Episode FFT 재로딩'}.get(path.name,
                         titles.get(path.name.split('_')[1], '# VLADE'))
            item['source'] = [title+'\n',
                '소스는 Git에서, 가중치·데이터는 Hugging Face 고정 revision에서 받습니다. 실행 시 SOURCE_COMMIT을 기록하고 재개할 때 SOURCE_REF에 지정하십시오.\n',
                '이 로컬 노트북 변경은 기존 Drive 노트북과 실행 중인 Colab에 자동 반영되지 않습니다.\n']
        if item['cell_type'] != 'code':
            continue
        source = ''.join(item['source'])
        if identifier == 'drive-and-source':
            source += '\n\n' + BOOTSTRAP
        source = re.sub(r"ROOT\s*/\s*(['\"])runs/", r"RUNS_ROOT / \1", source)
        source = source.replace("ROOT / 'run-configs'", "RUNS_ROOT / 'run-configs'")
        source = source.replace("ROOT / 'data/processed/libero_plus_episode80'", "hf_input('prepared_d_seed42', " +
                                ('STUDENT_PY' if '_03_' in path.name or '_05_' in path.name else 'AVA_PY') + ')')
        source = source.replace("str(ROOT / 'data/processed/libero_plus_tfds')", 'str(PREPARED)')
        source = source.replace("ROOT / 'checkpoints/teacher/avavla-libero-4in1'", "hf_input('ava_t0', AVA_PY)")
        source = source.replace("ROOT / 'checkpoints/baselines/openvla_oft/openvla-7b-oft-finetuned-libero-spatial-object-goal-10'",
                                "hf_input('openvla_oft_t0', OFT_PY)")
        source = source.replace("ROOT / 'data/teacher_cache'", "ASSET_WORK_ROOT / 'data/teacher_cache'")
        source = source.replace('ROOT / TEACHER_CACHE_RELATIVE', 'TEACHER_CACHE_RELATIVE')
        source = source.replace('shutil.disk_usage(ROOT)', 'shutil.disk_usage(RUNS_ROOT)')
        source = source.replace('planned_checkpoint_space(ROOT,', 'planned_checkpoint_space(RUNS_ROOT,')
        if identifier == 'cache':
            marker = "        run(command + (['--hidden'] if SAVE_HIDDEN else []))"
            source = source.replace(marker, marker + "\n        run([AVA_PY, 'scripts/data/hf_assets.py', 'publish',\n             'teacher_cache_t1_hidden' if SAVE_HIDDEN else 'teacher_cache_t1',\n             '--directory', ASSET_WORK_ROOT / 'data/teacher_cache' / ('T1-full-' + suffix), '--dataset'])")
        if identifier == 'download-student':
            source = source.replace("ROOT / 'checkpoints/students/qwen3_5_0_8b'", "hf_input('qwen_initial', fetch=False)")
            source = re.sub(r"(?m)^(\s*)run\(\[STUDENT_PY, '-c', program, QWEN\]\)$", r"\1hf_input('qwen_initial', STUDENT_PY)", source)
        if identifier == 'paper-config':
            source = source.replace("str(ROOT/config['initial_checkpoint'])", "config['initial_checkpoint']")
            source = source.replace("str(ROOT/config['data'])", "config['data']")
            source = source.replace("str(ROOT/config['output'])", "str(RUNS_ROOT/config['output'].removeprefix('runs/'))")
        if identifier == 'download-pretrained':
            source = re.sub(r"(?m)^(\s*)run\(\[AVA_PY,'-c',program,config\['initial_repository'\],config\['initial_revision'\],config\['initial_checkpoint'\]\]\)$",
                            r"\1hf_input('openvla_pretrained', AVA_PY)\n\1hf_input('prepared_d_seed42', AVA_PY)", source)
        source = source.replace('완료 checkpoint의 Drive 경로', '완료 checkpoint의 hfasset:// 키 또는 현재 실행 출력 경로')
        compile(source, path.name+':'+str(identifier), 'exec')
        item['source'] = source.splitlines(keepends=True)
    if path.name.startswith('colab_01_'):
        keep = {'purpose','drive-and-source','private-hub','install-data','private-hub-check'}
        value['cells'] = [c for c in value['cells'] if c.get('id') in keep]
        value['cells'].append(cell('code', '''
RUN_PREPARE_INPUTS = False
PREPARE_RAW_RLDS = False  # 전체 프레임을 새로 전처리할 때만 필요합니다.
if RUN_PREPARE_INPUTS:
    PREPARED = hf_input('prepared_d_seed42', DATA_PY)
    T0 = hf_input('ava_t0', DATA_PY)
    if PREPARE_RAW_RLDS:
        RAW_RLDS = hf_input('raw_rlds', DATA_PY)
    print('HF 입력 준비 완료:', PREPARED, T0)
else:
    print('자료 다운로드 대기: RUN_PREPARE_INPUTS=False')
''', 'hf-prepare-inputs'))
        from vlade.utils.fullframes_notebook import export_cell
        value['cells'].append(export_cell(root))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')


def refresh_inference_sources(path: Path) -> None:
    """Keep the historical smoke workflow, replacing its external input setup."""
    if not path.is_file():
        return
    from vlade.utils.colab_notebooks import bootstrap
    value = json.loads(path.read_text(encoding='utf-8'))
    for item in value['cells']:
        source = ''.join(item['source'])
        if item.get('id') == 'configure':
            source = bootstrap() + '\n\n' + BOOTSTRAP + '''
from google.colab import userdata
ENV['HF_TOKEN'] = userdata.get('HF_TOKEN')
MODEL = 'openvla_oft'  # or 'ava_vla'
TASK_DESCRIPTION = 'pick up the object'
UNNORM_KEY = 'libero_spatial_no_noops'
MODEL_PATHS = {'openvla_oft': ('third_party/openvla_oft', 'openvla_oft_t0'),
               'ava_vla': ('third_party/ava_vla', 'ava_t0')}
if MODEL not in MODEL_PATHS:
    raise ValueError('Unknown smoke-test model')
source_relative, asset_key = MODEL_PATHS[MODEL]
if MODEL == 'openvla_oft':
    subprocess.run(['git', '-C', str(CODE), 'submodule', 'update', '--init', '--depth', '1', '--', source_relative], check=True)
SOURCE = CODE / source_relative
DOWNLOAD_PY = environment('vlade_asset_download', '3.11', ['huggingface_hub==0.36.2'])
CHECKPOINT = hf_input(asset_key, DOWNLOAD_PY)
COPY_LARGE_FILES_TO_SCRATCH = False
'''
        elif item.get('id') == 'save-result':
            source = source.replace("'checkpoint_path': str(CHECKPOINT.relative_to(PROJECT_ROOT))", "'checkpoint_asset': 'hfasset://' + asset_key")
            source = source.replace("'notebook_has_uncommitted_runtime_compatibility_fixes': True", "'notebook_has_uncommitted_runtime_compatibility_fixes': False")
            source = re.sub(r"'repository_commit': '[^']+'", "'repository_commit': SOURCE_COMMIT", source)
            source = source.replace('PROJECT_ROOT / \'experiments/runs', 'RUNS_ROOT / \'inference')
            source = source.replace("PROJECT_ROOT / 'artifacts", "RUNS_ROOT / 'inference")
        if item['cell_type'] == 'code':
            compile(source, str(path), 'exec')
        item['source'] = source.splitlines(keepends=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
