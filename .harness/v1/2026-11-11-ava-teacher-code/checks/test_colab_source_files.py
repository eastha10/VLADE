"""Git source deployment is immutable and independent of assets/Drive."""
import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(os.environ["VLADE_PROJECT_ROOT"]) / 'src'))
from vlade.utils import colab_notebooks
from vlade.utils.source import stage_source


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / 'remote'
    root.mkdir()
    git(root, 'init', '-b', 'main')
    (root / 'core.py').write_text('VALUE = 1\n')
    git(root, 'add', 'core.py')
    git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'initial')
    return root


def test_stage_pins_commit_and_never_updates_a_running_checkout(tmp_path, repository):
    commit = git(repository, 'rev-parse', 'HEAD')
    code = stage_source(str(repository), tmp_path / 'runtime', commit)
    assert git(code, 'rev-parse', 'HEAD') == commit
    assert stage_source(str(repository), tmp_path / 'runtime', commit) == code
    (repository / 'core.py').write_text('VALUE = 2\n')
    git(repository, 'add', 'core.py')
    git(repository, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'next')
    newer = stage_source(str(repository), tmp_path / 'runtime', 'main')
    assert newer != code
    assert (code / 'core.py').read_text() == 'VALUE = 1\n'
    assert (newer / 'core.py').read_text() == 'VALUE = 2\n'


@pytest.mark.parametrize('change', ['tracked', 'untracked'])
def test_changed_cached_source_is_rejected(tmp_path, repository, change):
    code = stage_source(str(repository), tmp_path / 'runtime')
    (code / ('core.py' if change == 'tracked' else 'unexpected.py')).write_text('changed')
    with pytest.raises(ValueError, match='checkout changed'):
        stage_source(str(repository), tmp_path / 'runtime')


@pytest.mark.parametrize('ref', ['--upload-pack=bad', '../main', 'main;command', ''])
def test_invalid_git_reference_is_rejected_before_fetch(tmp_path, ref):
    with pytest.raises(ValueError, match='reference'):
        stage_source('unused', tmp_path, ref)


def test_generated_notebooks_use_git_and_hf_without_source_file_hashes(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    monkeypatch.setattr(colab_notebooks, 'ROOT', root)
    generated = colab_notebooks.generate()
    manifest = json.loads(generated[-1].read_text())
    assert manifest['schema_version'] == 2 and manifest['distribution'] == 'git_commit'
    assert 'files_sha256' not in manifest
    for path in generated[:-1]:
        value = json.loads(path.read_text(encoding='utf-8'))
        sources = '\n'.join(''.join(c['source']) for c in value['cells'] if c['cell_type'] == 'code')
        assert 'SOURCE_COMMIT' in sources and 'HF_TOKEN' in sources
        assert 'files_sha256' not in sources and 'drive.mount(' not in sources
        assert '/content/drive' not in sources
        assert '/content/vlade-preprocess/libero_plus_episode80' not in sources
        for c in value['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), path.name, 'exec')
    before = generated[-1].read_bytes()
    colab_notebooks.generate()
    assert generated[-1].read_bytes() == before


def test_all_shipped_runtime_notebooks_use_hf_and_git():
    root = Path(os.environ["VLADE_PROJECT_ROOT"])
    for path in (root/'experiments').glob('colab*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8-sig'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        assert 'SOURCE_COMMIT' in code, path.name
        for removed in ['drive.mount(', 'drive.flush_and_unmount(', '/content/drive',
                        'files_sha256', 'PREP_WORKER_SOURCE', 'PREP_CACHE_SOURCE',
                        'CHECKPOINT.relative_to(PROJECT_ROOT)']:
            assert removed not in code, (path.name, removed)
        for c in notebook['cells']:
            if c['cell_type']=='code':
                compile(''.join(c['source']), path.name, 'exec')


def test_source_checkout_preserves_parent_pinned_submodule(tmp_path,repository,monkeypatch):
    dependency=tmp_path/'upstream'; dependency.mkdir();git(dependency,'init','-b','main')
    (dependency/'module.py').write_text('PINNED=True\n');git(dependency,'add','module.py')
    git(dependency,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','dependency')
    expected=git(dependency,'rev-parse','HEAD');monkeypatch.setenv('GIT_ALLOW_PROTOCOL','file')
    git(repository,'-c','protocol.file.allow=always','submodule','add',str(dependency),'third_party/ava_vla')
    git(repository,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-am','pin')
    code=stage_source(str(repository),tmp_path/'runtime')
    assert git(code,'ls-tree','HEAD','third_party/ava_vla').split()[2]==expected
    assert not (code/'third_party/ava_vla/module.py').exists()
    assert stage_source(str(repository),tmp_path/'runtime')==code
