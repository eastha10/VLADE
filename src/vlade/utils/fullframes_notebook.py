"""Optional full-frame export using Git source and pinned HF raw input."""
from pathlib import Path

REFERENCE_REVISION = '21a81ee2c9cf120019ed18c5d1cd06bf4336c69e'


def export_cell(root: Path) -> dict:
    """New exports read the immutable HF RLDS input, never a Drive mount."""
    from vlade.utils.colab_notebooks import cell
    return cell('code', '''
RUN_FULLFRAME_EXPORT = False
if RUN_FULLFRAME_EXPORT:
    from google.colab import userdata
    ENV['HF_TOKEN'] = userdata.get('HF_TOKEN')
    ENV.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_DISABLE_XET='1', PYTHONUNBUFFERED='1')
    RAW_RLDS = hf_input('raw_rlds', DATA_PY)
    work_root = ASSET_WORK_ROOT / 'fullframes-export'
    work_root.mkdir(parents=True, exist_ok=True)
    run([DATA_PY, 'scripts/data/prepare_fullframes.py', '--tfds-root', RAW_RLDS,
         '--work-root', work_root, '--repo-id', 'VLADE-konkuk/libero-plus-fullframes',
         '--reference-revision', '__REVISION__', '--public'], env=ENV)
else:
    print('전체 프레임 내보내기 대기: RUN_FULLFRAME_EXPORT=False')
'''.replace('__REVISION__', REFERENCE_REVISION), 'export-fullframes-to-hf')
