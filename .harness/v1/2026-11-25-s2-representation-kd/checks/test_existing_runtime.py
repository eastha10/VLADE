"""Reuse exact environment metadata; setup requires an explicit Student-only flag."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / 'AGENTS.md').is_file()
sys.path.insert(0,str(ROOT / 'src'))
from vlade.runtime import existing, environment


@pytest.fixture
def fixtures(tmp_path,monkeypatch):
    code, content = tmp_path / 'code', tmp_path / 'content'
    (code / 'configs/runtime').mkdir(parents=True)
    content.mkdir()
    specs = {name:{'name':name,'schema_version':1,'python':'3.11','packages':['synthetic']} for name in ('student','ava')}
    for name,spec in specs.items():
        (code / f'configs/runtime/{name}.json').write_text(json.dumps(spec))
    monkeypatch.setattr(existing,'prepare',lambda *a:pytest.fail('Unexpected setup'))
    def create(work,name,*,spec=None):
        configured = specs[name]
        identity = hashlib.sha256(json.dumps(configured,sort_keys=True).encode()).hexdigest()[:12]
        folder = work / 'environments' / (name+'-'+identity)
        python = environment.interpreter(folder)
        python.parent.mkdir(parents=True)
        python.write_bytes(b'synthetic executable, never launched')
        (folder / 'vlade-runtime.json').write_text(json.dumps(spec or configured))
        return folder
    return code,content,create


def test_current_environments_are_reused_without_setup_or_links(fixtures):
    code,content,create = fixtures
    work = content / 'current'
    folders = {name:create(work,name) for name in ('student','ava')}
    result = existing.reconnect(code,work,search_root=content)
    assert result == {name:str(environment.interpreter(folder)) for name,folder in folders.items()}


def test_missing_student_is_explicit_and_does_not_create_output(fixtures):
    code,content,create = fixtures
    work = content / 'current'
    create(content / 'old','ava')
    with pytest.raises(FileNotFoundError,match='student.*allow-student-setup'):
        existing.reconnect(code,work,search_root=content)
    assert not work.exists()


def test_wrong_spec_is_not_relabelled(fixtures):
    code,content,create = fixtures
    create(content / 'old','student',spec={'name':'student','packages':['wrong']})
    with pytest.raises(FileNotFoundError,match='student'):
        existing.reconnect(code,content / 'current',search_root=content)


def test_multiple_candidates_are_not_arbitrarily_selected(fixtures):
    code,content,create = fixtures
    for name in ('one','two'):
        create(content / name,'student')
    with pytest.raises(ValueError,match='여러 개'):
        existing.reconnect(code,content / 'current',search_root=content)


def test_existing_incomplete_target_is_not_overwritten(fixtures):
    code,content,create = fixtures
    folder = create(content / 'current','student')
    (folder / 'vlade-runtime.json').unlink()
    create(content / 'old','student')
    with pytest.raises(FileExistsError,match='덮어쓰지'):
        existing.reconnect(code,content / 'current',search_root=content)
    assert environment.interpreter(folder).read_bytes() == b'synthetic executable, never launched'


def test_optional_setup_only_student_after_ava_is_verified(fixtures,monkeypatch):
    code,content,create = fixtures
    work = content / 'current'
    create(work,'ava')
    calls = []
    def prepare(code_arg,work_arg,name):
        calls.append(name)
        create(work_arg,name)
    monkeypatch.setattr(existing,'prepare',prepare)
    result = existing.reconnect(code,work,search_root=content,allow_student_setup=True)
    assert calls == ['student'] and set(result) == {'student','ava'}


def test_no_install_if_ava_also_missing(fixtures,monkeypatch):
    code,content,create = fixtures
    with pytest.raises(FileNotFoundError,match='ava'):
        existing.reconnect(code,content / 'current',search_root=content,allow_student_setup=True)


def test_drive_workspace_is_rejected(fixtures):
    code,content,create = fixtures
    with pytest.raises(ValueError,match='Drive'):
        existing.reconnect(code,content / 'drive/work',search_root=content)


def test_found_student_is_linked_while_cache_stays_in_place(fixtures,monkeypatch):
    code,content,create = fixtures
    work, old = content / 'current', content / 'old'
    source = create(old,'student'); create(work,'ava')
    cache = work / 'cache/sentinel'; cache.parent.mkdir(); cache.write_bytes(b'keep')
    # Native symlinks require a Windows privilege; exercise the intended call and
    # subsequent locate with a tiny fixture alias, without copying any repository.
    calls = []
    def link(path,target,*,target_is_directory):
        calls.append((path,target,target_is_directory))
        create(work,'student')
    monkeypatch.setattr(Path,'symlink_to',link)
    result = existing.reconnect(code,work,search_root=content)
    assert calls == [(Path(result['student']).parent.parent,source.resolve(),True)]
    assert cache.read_bytes() == b'keep' and environment.interpreter(source).is_file()
