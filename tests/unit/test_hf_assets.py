"""Portable HF inputs, integrity enforcement and safe legacy checkpoint import."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from vlade.storage import assets
from vlade.storage.migrate import checkpoint_sources, model_sources


def row(data):
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def registry_file(tmp_path, record):
    path=tmp_path/'registry.json'
    path.write_text(json.dumps({'schema_version':1,'assets':{'example':record}}))
    return path


def mock_download(monkeypatch, tmp_path, files):
    import huggingface_hub
    cache=tmp_path/'remote'
    for name,data in files.items():
        path=cache/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    def download(repo, filename, *, repo_type, revision):
        assert repo=='test/private' and repo_type=='dataset' and revision=='a'*40
        return str(cache/filename)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',download)
    monkeypatch.setattr(assets,'configure_http',lambda:None)


def test_directory_asset_download_cache_and_tamper(tmp_path,monkeypatch):
    files={'COMPLETE':b'ok\n','model/tokenizer.json':b'{}','trainer_state.pt':b'optimizer/scheduler'}
    record=dict(repo_id='test/private',repo_type='dataset',revision='a'*40,format='directory',prefix='assets/x',
                local_name='step-0000001',files={k:row(v) for k,v in files.items()})
    registry=registry_file(tmp_path,record)
    mock_download(monkeypatch,tmp_path,{'assets/x/files/'+k:v for k,v in files.items()})
    result=assets.fetch_asset('example',registry=registry,cache=tmp_path/'cache')
    assert result.name=='step-0000001'
    assert (result/'model/tokenizer.json').read_bytes()==b'{}'
    assert assets.fetch_asset('example',registry=registry,cache=tmp_path/'cache')==result
    (result/'trainer_state.pt').write_bytes(b'damaged')
    with pytest.raises(ValueError,match='integrity'):
        assets.fetch_asset('example',registry=registry,cache=tmp_path/'cache')


def test_archive_roundtrip_preserves_dataset_identity(tmp_path,monkeypatch):
    files={'manifest.json':b'{"split":"sealed"}','splits.json':b'[]','normalization.json':b'{}','COMPLETE':b'ok\n'}
    archive=tmp_path/'prepared.tar'
    with tarfile.open(archive,'w') as stream:
        for name,data in files.items():
            member=tarfile.TarInfo('prepared/'+name);member.size=len(data);stream.addfile(member,io.BytesIO(data))
    data=archive.read_bytes()
    record=dict(repo_id='test/private',repo_type='dataset',revision='a'*40,format='archive',archive_path='prepared.tar',
                archive_root='prepared',archive_bytes=len(data),archive_sha256=row(data)['sha256'],
                local_name='prepared',identity_files={k:row(v) for k,v in files.items() if k!='COMPLETE'})
    registry=registry_file(tmp_path,record);mock_download(monkeypatch,tmp_path,{'prepared.tar':data})
    result=assets.fetch_asset('example',registry=registry,cache=tmp_path/'cache')
    assert all((result/k).read_bytes()==v for k,v in files.items())


@pytest.mark.parametrize('kind',['traversal','symlink','duplicate'])
def test_unsafe_dataset_archive_is_never_published(tmp_path,kind):
    archive=tmp_path/'unsafe.tar'
    with tarfile.open(archive,'w') as stream:
        member=tarfile.TarInfo('../outside' if kind=='traversal' else 'payload')
        if kind=='symlink':member.type=tarfile.SYMTYPE;member.linkname='../outside';stream.addfile(member)
        else:
            member.size=1;stream.addfile(member,io.BytesIO(b'x'))
            if kind=='duplicate':stream.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError):assets._extract_archive(archive,tmp_path/'stage',{})
    assert not (tmp_path/'outside').exists()


@pytest.mark.parametrize('change',[{'revision':'main'},{'repo_type':'space'}])
def test_registry_requires_pinned_version_and_expected_repository_type(tmp_path,change):
    record=dict(repo_id='test/private',repo_type='dataset',revision='a'*40,format='directory',files={})
    record.update(change)
    with pytest.raises(ValueError):assets.asset_record('example',registry_file(tmp_path,record))


def test_drive_copy_names_are_reconstructed_from_seal_without_renaming_sources(tmp_path):
    payload={'config.json':b'{}','trainer_state.pt':b'state','tokenizer.json':b'{}'}
    seal=dict(schema_version=1,step=1975,microsteps=15800,files={k:row(v) for k,v in payload.items()})
    for name,data in payload.items():(tmp_path/(name+'의 사본')).write_bytes(data)
    (tmp_path/'checkpoint_manifest.json의 사본').write_text(json.dumps(seal))
    (tmp_path/'COMPLETE의 사본').write_text('ok')
    result,metadata=checkpoint_sources(tmp_path)
    assert metadata==seal and set(payload)<=set(result)
    assert all(path.name.endswith('의 사본') for path in result.values())


def test_incomplete_base_model_is_not_imported_as_complete(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='base weights'):model_sources(tmp_path)
    (tmp_path/'model-00001-of-00002.safetensors').write_bytes(b'one')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model-00001-of-00002.safetensors','b':'model-00002-of-00002.safetensors'}}))
    with pytest.raises(ValueError,match='shards'):model_sources(tmp_path)
