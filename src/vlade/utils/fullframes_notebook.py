"""One resumable CPU export cell; retain and fold the older preparation sections."""
from pathlib import Path
import hashlib
import json

REFERENCE_REVISION='21a81ee2c9cf120019ed18c5d1cd06bf4336c69e'


def export_cell(root: Path) -> dict:
    from vlade.utils.colab_notebooks import cell
    names=('src/vlade/data/tfrecord_frames.py','src/vlade/data/fullframes.py',
        'src/vlade/storage/__init__.py','src/vlade/storage/assets.py','src/vlade/storage/migrate.py',
        'src/vlade/storage/drive_checkpoint.py','scripts/data/prepare_fullframes.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    source='''
# 원본 전체 프레임 → 승인된 새 공개 HF 데이터셋. 기존 원본·비공개 자료는 보존합니다.
RUN_FULLFRAME_EXPORT = False
if RUN_FULLFRAME_EXPORT:
    from pathlib import Path
    import os, sys, subprocess, hashlib, re
    from google.colab import drive, userdata
    PROJECT_ROOT = Path(globals().get('PROJECT_ROOT', '/content/drive/MyDrive/colab/VLADE'))
    if not Path('/content/drive/MyDrive').is_dir():
        drive.mount('/content/drive')
    expected_sources = __HASHES__
    source_id = hashlib.sha256(repr(expected_sources).encode()).hexdigest()[:16]
    source_root = Path('/content/vlade_fullframes_source_'+source_id)
    for relative, expected in expected_sources.items():
        data = (PROJECT_ROOT/relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Drive 코드와 노트북 버전이 다릅니다. 새로고침하십시오: '+relative)
        target = source_root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    python = Path('/content/vlade_fullframes_env/bin/python')
    if not python.is_file():
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(python.parent.parent)], check=True)
    versions = {'huggingface-hub':'0.36.2','pyarrow':'19.0.0','protobuf':'5.29.3','google-crc32c':'1.7.1','numpy':'2.2.6'}
    check = 'from importlib.metadata import version; assert all(version(k)==v for k,v in '+repr(versions)+'.items())'
    if subprocess.run([str(python), '-c', check], capture_output=True).returncode:
        print('전체 프레임 내보내기 CPU 환경을 준비합니다.', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install',
            '--disable-pip-version-check', *[k+'=='+v for k,v in versions.items()]], check=True)
    env = os.environ.copy()
    env.update(HF_TOKEN=userdata.get('HF_TOKEN'), HF_HOME='/content/vlade_fullframes_hf_cache',
        HF_TOKEN_PATH='/content/vlade_hf_token', HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DISABLE_XET='1',
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    env.pop('VLADE_HF_TLS12', None)
    work_root = Path('/content/vlade_fullframes_export')
    work_root.mkdir(parents=True, exist_ok=True)
    log_path = work_root/'export.log'
    print('전체 프레임 내보내기 로그:', log_path, flush=True)
    print('목표: 14,347 에피소드 / 2,238,036 프레임. 완료 구간은 검증 후 재사용합니다.', flush=True)
    command = [str(python), '-u', str(source_root/'scripts/data/prepare_fullframes.py'),
        '--tfds-root', str(PROJECT_ROOT/'data/processed/libero_plus_tfds'), '--work-root', str(work_root),
        '--repo-id', 'VLADE-konkuk/libero-plus-fullframes', '--reference-revision', '__REVISION__', '--public']
    with log_path.open('a', encoding='utf-8') as log:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                clean = line.replace(env['HF_TOKEN'], '[REDACTED]') if env['HF_TOKEN'] else line
                clean = re.sub(r'hf_[A-Za-z0-9]{12,}', '[REDACTED]', clean)
                print(clean, end='', flush=True)
                log.write(clean); log.flush()
            returncode = process.wait()
    if returncode:
        raise RuntimeError('내보내기가 중단되었습니다. 위 fullframes_export_failed 로그를 확인하십시오. 같은 셀로 재개할 수 있습니다.')
else:
    print('전체 프레임 내보내기를 실행하려면 이 셀의 RUN_FULLFRAME_EXPORT를 True로 바꾸십시오.')
'''.replace('__HASHES__',repr(hashes)).replace('__REVISION__',REFERENCE_REVISION)
    return cell('code',source,'export-fullframes-to-hf')


def apply_fullframes(path: Path, root: Path) -> None:
    """Update only the export cell and headings, preserving user outputs and cells."""
    from vlade.utils.colab_notebooks import cell
    value=json.loads(path.read_text('utf-8'))
    previous=next((c for c in value['cells'] if c.get('id')=='export-fullframes-to-hf'),None)
    code=export_cell(root)
    if previous:
        for key in ('outputs','execution_count','metadata'):code[key]=previous.get(key,code[key])
        if 'RUN_FULLFRAME_EXPORT = True' in ''.join(previous['source']):
            code['source']=''.join(code['source']).replace('RUN_FULLFRAME_EXPORT = False','RUN_FULLFRAME_EXPORT = True').splitlines(keepends=True)
    cells=[c for c in value['cells'] if c.get('id') not in {'fullframes-purpose','export-fullframes-to-hf'}]
    for item in cells:
        identifier=item.get('id')
        if identifier=='hf-migration-purpose':
            item['source']=['# 기존 비공개 자료 이관 · 완료 기록\n','기존 데이터와 가중치의 비공개 HF 이관 기록입니다. 재실행할 필요가 없습니다.\n']
        if identifier in {'hf-migration-purpose','purpose'}:item.setdefault('metadata',{})['heading_collapsed']=True
        if identifier=='migrate-existing-assets-to-hf':
            item['source']=''.join(item['source']).replace('RUN_HF_MIGRATION = True','RUN_HF_MIGRATION = False').splitlines(keepends=True)
    intro=cell('markdown','# 01 · 원본 전체 프레임 → 공개 Hugging Face\n'
        '바로 아래 셀에서 `RUN_FULLFRAME_EXPORT = True`로 바꾸고 실행하십시오. 기존 Secrets `HF_TOKEN`을 그대로 사용합니다. '
        'CPU 작업이며 다운로드·전처리·기존 이관 셀을 먼저 실행할 필요가 없습니다.\n\n'
        '기존 Drive의 해제된 원본을 읽고 JPEG·숫자 값을 보존합니다. 기존 D 분할·정규화도 유지합니다. '
        '새 공개 저장소는 `VLADE-konkuk/libero-plus-fullframes`입니다. 128구간 중 게시·검증된 구간은 재시작 시 건너뜁니다. '
        '마지막 `fullframes_export_completed` 또는 `fullframes_export_already_complete` 로그와 원격 `COMPLETE`가 완료 기준입니다.\n\n'
        '기존 다운로드·전처리 셀과 출력은 아래 접힌 구역에 보존했습니다. 기존 학습 입력은 아직 희소 프레임 버전을 사용합니다.',
        'fullframes-purpose')
    previous_intro=next((c for c in value['cells'] if c.get('id')=='fullframes-purpose'),None)
    if previous_intro:intro['metadata']=previous_intro.get('metadata',{}).copy()
    value['cells']=[intro,code,*cells]
    metadata=value.setdefault('metadata',{}).setdefault('colab',{})
    metadata['collapsed_sections']=list(dict.fromkeys([*metadata.get('collapsed_sections',[]),'hf-migration-purpose','purpose']))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
