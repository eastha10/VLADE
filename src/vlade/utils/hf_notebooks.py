"""Keep Colab storage references on HF while preserving existing preparation cells."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

BOOTSTRAP = '''
# HF inputs are cached on the runtime SSD; Drive retains code and experiment records.
ASSET_CACHE = Path('/content/vlade_hf_assets')
ASSET_WORK_ROOT = Path('/content/vlade_asset_work')
RUNS_ROOT = Path('/content/vlade_runs')
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
ENV.update(VLADE_ASSET_CACHE=str(ASSET_CACHE),
           VLADE_ASSET_REGISTRY=str(CODE/'configs/storage/hf_assets.lock.json'),
           HF_HOME='/content/vlade_hf_home', HF_TOKEN_PATH='/content/vlade_hf_token')
os.environ.update({key: ENV[key] for key in ('VLADE_ASSET_CACHE','VLADE_ASSET_REGISTRY')})
_HF_REGISTRY_READY = False

def hf_input(key, python=None, *, fetch=True):
    global _HF_REGISTRY_READY
    from vlade.storage.assets import asset_record, cache_path
    if fetch and not _HF_REGISTRY_READY:
        if python is None:
            raise ValueError('Select the prepared Python environment before fetching HF assets')
        registry = ASSET_CACHE / 'registry.json'
        run([python, 'scripts/data/hf_assets.py', 'refresh-registry', '--output', registry], env=ENV)
        ENV['VLADE_ASSET_REGISTRY'] = str(registry)
        os.environ['VLADE_ASSET_REGISTRY'] = str(registry)
        _HF_REGISTRY_READY = True
    record = asset_record(key, Path(ENV['VLADE_ASSET_REGISTRY']))
    target = cache_path(key, record, ASSET_CACHE)
    if fetch:
        if python is None:
            raise ValueError('Select the prepared CPU/model Python environment before fetching HF assets')
        run([python, 'scripts/data/hf_assets.py', 'fetch', key,
             '--registry', ENV['VLADE_ASSET_REGISTRY'], '--cache', ASSET_CACHE], env=ENV)
    return target

def hf_path(value, python=None, *, fetch=True):
    value = str(value)
    return hf_input(value[len('hfasset:'):].strip('/\\\\'), python, fetch=fetch) if value.startswith('hfasset:') else Path(value)
'''.strip()


def migration_cell(root: Path) -> dict:
    from vlade.utils.colab_notebooks import cell
    names = ("src/vlade/storage/__init__.py", "src/vlade/storage/assets.py", "src/vlade/storage/migrate.py", "src/vlade/storage/drive_checkpoint.py",
             "scripts/data/hf_assets.py", "scripts/data/migrate_hf_assets.py")
    hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names if (root/name).is_file()}
    source = '''
# Existing Drive data/weights -> private HF. This runs CPU transfers, never GPU training.
RUN_HF_MIGRATION = False
if RUN_HF_MIGRATION:
    from pathlib import Path
    import os, sys, subprocess, hashlib
    from google.colab import drive, userdata
    PROJECT_ROOT = Path(globals().get('PROJECT_ROOT', '/content/drive/MyDrive/colab/VLADE'))
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    expected_sources = __HASHES__
    for relative, expected in expected_sources.items():
        if hashlib.sha256((PROJECT_ROOT/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Migration source changed; reopen the matching notebook: '+relative)
    python = Path('/content/vlade_hf_transfer/bin/python')
    if not python.is_file():
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(python.parent.parent)], check=True)
    installed = subprocess.run([str(python), '-c', "import huggingface_hub; assert huggingface_hub.__version__ == '0.36.2'"], capture_output=True)
    if installed.returncode:
        print('HF 이관 환경을 준비합니다.', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install',
                        '--disable-pip-version-check', 'huggingface_hub==0.36.2'], check=True)
    env = os.environ.copy()
    env.update(HF_TOKEN=userdata.get('HF_TOKEN'), HF_HOME='/content/vlade_hf_migration_cache',
               HF_TOKEN_PATH='/content/vlade_hf_token', HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONUNBUFFERED='1')
    import importlib.util
    spec = importlib.util.spec_from_file_location('vlade_drive_checkpoint', PROJECT_ROOT/'src/vlade/storage/drive_checkpoint.py')
    checkpoint_io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoint_io)
    checkpoint = checkpoint_io.prepare_colab_checkpoint(PROJECT_ROOT, python.parent.parent/'checkpoint-source',
        '1YF8pC6rg55KIQrLWAVE5lcvdXAvAhHCB', expected_step=1975)
    log_path = python.parent.parent / 'migration.log'
    print('이관 진행 로그:', log_path, flush=True)
    command = [str(python), '-u', str(PROJECT_ROOT/'scripts/data/migrate_hf_assets.py'), '--root', str(PROJECT_ROOT), '--checkpoint', str(checkpoint)]
    with log_path.open('a', encoding='utf-8') as log:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                clean = line.replace(env['HF_TOKEN'], '[REDACTED]') if env['HF_TOKEN'] else line
                print(clean, end='', flush=True)
                log.write(clean); log.flush()
            returncode = process.wait()
    if returncode:
        raise RuntimeError('이관이 중단되었습니다. 바로 위 migration_failed의 message와 where를 확인하십시오. 로그: '+str(log_path))
else:
    print('Existing-asset migration is off. Use the HF input preparation cells when migration is complete.')
'''.replace('__HASHES__', repr(hashes))
    return cell("code", source, "migrate-existing-assets-to-hf")


def refresh_suite_sources(path: Path, root: Path, digest: str) -> None:
    """Keep the optional existing Suite notebook on the regenerated source manifest."""
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding='utf-8'))
    adapter = (root / 'src/vlade/training/colab_suite_fft.py').read_text(encoding='utf-8')
    for item in value['cells']:
        source = ''.join(item['source'])
        if item.get('id') == 'ava-suite-05':
            source = re.sub(r"(?m)^SOURCE_MANIFEST_SHA\s*=.*$", 'SOURCE_MANIFEST_SHA = '+repr(digest), source)
        if item.get('id') == 'ava-suite-06':
            source = re.sub(r'(?m)^ADAPTER_SOURCE\s*=.*$', lambda _: 'ADAPTER_SOURCE = '+repr(adapter), source)
            source = re.sub(r'(?m)^ADAPTER_SHA256\s*=.*$', 'ADAPTER_SHA256 = '+repr(hashlib.sha256(adapter.encode()).hexdigest()), source)
        if item['cell_type'] == 'code':
            compile(source, str(path), 'exec')
        item['source'] = source.splitlines(keepends=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')


def apply_hf_storage(path: Path, root: Path) -> None:
    from vlade.utils.colab_notebooks import cell
    value = json.loads(path.read_text(encoding="utf-8"))
    for item in value["cells"]:
        if item["cell_type"] != "code":
            continue
        identifier = item.get("id")
        source = "".join(item["source"])
        if identifier == "drive-and-source":
            source = source.split('\n\n# HF inputs are cached',1)[0] + "\n\n" + BOOTSTRAP
        if not path.name.startswith('colab_01_') and identifier not in {"drive-and-source", "private-hub", "migrate-existing-assets-to-hf"}:
            source = re.sub(r"ROOT\s*/\s*(['\"])runs/", r"RUNS_ROOT / \1", source)
            if not path.name.startswith('colab_01_'):
                source = source.replace("ROOT / 'data/processed/libero_plus_episode80'", "hf_input('prepared_d_seed42', " +
                                        ("STUDENT_PY" if '_03_' in path.name or '_05_' in path.name else "AVA_PY") + ")")
            source = source.replace("str(ROOT / 'data/processed/libero_plus_tfds')", "str(PREPARED)")
            source = source.replace("ROOT / 'checkpoints/teacher/avavla-libero-4in1'", "hf_input('ava_t0', AVA_PY)")
            source = source.replace("ROOT / 'data/teacher_cache'", "ASSET_WORK_ROOT / 'data/teacher_cache'")
            source = source.replace("ROOT / TEACHER_CACHE_RELATIVE", "TEACHER_CACHE_RELATIVE")
            source = source.replace("shutil.disk_usage(ROOT)", "shutil.disk_usage(RUNS_ROOT)")
            source = source.replace("planned_checkpoint_space(ROOT,", "planned_checkpoint_space(RUNS_ROOT,")
            if identifier == 'cache' and 'hf_assets.py' not in source:
                marker = "        run(command + (['--hidden'] if SAVE_HIDDEN else []))"
                source = source.replace(marker, marker + "\n        run([AVA_PY, 'scripts/data/hf_assets.py', 'publish',\n             'teacher_cache_t1_hidden' if SAVE_HIDDEN else 'teacher_cache_t1',\n             '--directory', ASSET_WORK_ROOT / 'data/teacher_cache' / ('T1-full-' + suffix), '--dataset'])")
            if identifier == "download-student":
                source = source.replace("ROOT / 'checkpoints/students/qwen3_5_0_8b'", "hf_input('qwen_initial', fetch=False)")
                source = re.sub(r"(?m)^(\s*)run\(\[STUDENT_PY, '-c', program, QWEN\]\)$", r"\1hf_input('qwen_initial', STUDENT_PY)", source)
            if identifier == "paper-config":
                source = source.replace("str(ROOT/config['initial_checkpoint'])", "str(hf_path(config['initial_checkpoint'], fetch=False))")
                source = source.replace("str(ROOT/config['data'])", "str(hf_path(config['data'], fetch=False))")
                source = source.replace("str(ROOT/config['output'])", "str(RUNS_ROOT/config['output'].removeprefix('runs/'))")
                source = re.sub(r"str\(hf_path\(config\['(initial_checkpoint|data)'\], fetch=False\)\)", r"config['\1']", source)
            if identifier == "download-pretrained":
                source = re.sub(r"(?m)^(\s*)run\(\[AVA_PY,'-c',program,config\['initial_repository'\],config\['initial_revision'\],config\['initial_checkpoint'\]\]\)$",
                                r"\1hf_input('openvla_pretrained', AVA_PY)", source)
                if "hf_input('prepared_d_seed42'" not in source:
                    source = source.replace("    hf_input('openvla_pretrained', AVA_PY)", "    hf_input('openvla_pretrained', AVA_PY)\n    hf_input('prepared_d_seed42', AVA_PY)")
        if item["cell_type"] == "code": compile(source, path.name+":"+str(identifier), "exec")
        item["source"] = source.splitlines(keepends=True)

    if path.name.startswith("colab_01_"):
        original = [item for item in value["cells"] if item.get("id") not in {"hf-migration-purpose", "migrate-existing-assets-to-hf"}]
        intro = cell("markdown", "# 01 · HF 자료 이관과 준비\n기존 자료 이관은 다음 셀에서 실행합니다. 기존 다운로드·전처리 셀은 아래 원래 구역에 보존하고 접어 두었습니다.", "hf-migration-purpose")
        previous_intro = next((c for c in value['cells'] if c.get('id')=='hf-migration-purpose'),None)
        if previous_intro:
            intro['metadata'] = previous_intro.get('metadata', {}).copy()
        previous = next((c for c in value["cells"] if c.get("id")=="migrate-existing-assets-to-hf"), None)
        migrate = migration_cell(root)
        if previous:
            migrate["outputs"] = previous.get("outputs", [])
            migrate["execution_count"] = previous.get("execution_count")
            migrate["metadata"] = previous.get("metadata", {})
            flag = re.search(r'(?m)^RUN_HF_MIGRATION\s*=\s*(True|False)', ''.join(previous['source']))
            if flag:
                migrate['source'] = ''.join(migrate['source']).replace('RUN_HF_MIGRATION = False', 'RUN_HF_MIGRATION = '+flag[1]).splitlines(keepends=True)
        value["cells"] = [intro, migrate, *original]
        purpose = next(c for c in original if c.get("id")=="purpose")
        purpose.setdefault("metadata", {})["heading_collapsed"] = True
        metadata = value.setdefault("metadata", {}).setdefault("colab", {})
        metadata["collapsed_sections"] = list(dict.fromkeys([*metadata.get("collapsed_sections", []), "purpose"]))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    if path.name.startswith('colab_01_'):
        from vlade.utils.fullframes_notebook import apply_fullframes
        apply_fullframes(path,root)
