"""Lossless source export, CRC failures and restart after a failed Hub commit."""
import os
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip('google.protobuf')
sys.path.insert(0,str(Path(os.environ["VLADE_PROJECT_ROOT"])/'src'))
from vlade.data import fullframes as ff
from vlade.data import tfrecord_frames as tf


def example(steps=10):
    message=tf.example_type()();features=message.features.feature
    for name,size in [('action',7),('observation/state',8),('observation/joint_state',7)]:
        features['steps/'+name].float_list.value.extend([float(i)/8 for i in range(steps*size)])
    for name in ('image','wrist_image'):
        features['steps/observation/'+name].bytes_list.value.extend([b'\xff\xd8'+name.encode()+bytes([t])+b'\xff\xd9' for t in range(steps)])
    features['steps/language_instruction'].bytes_list.value.extend([b'pick up object']*steps)
    for name,values in [('is_first',[1]+[0]*(steps-1)),('is_last',[0]*(steps-1)+[1]),('is_terminal',[0]*(steps-1)+[1])]:
        features['steps/'+name].int64_list.value.extend(values)
    features['steps/reward'].float_list.value.extend([0.]*(steps-1)+[1.])
    features['steps/discount'].float_list.value.extend([1.]*steps)
    features['episode_metadata/file_path'].bytes_list.value.append(b'original-public-source/demo.hdf5')
    return message


def sealed_row(episode,identifier='episode-0',split='train'):
    return dict(episode_id=identifier,content_group_id='group-'+episode['content_sha256'],
        content_sha256=episode['content_sha256'],split=split,is_calibration=False,
        steps=episode['steps'],queries=(episode['steps']+7)//8)


def write_record(path,payload):
    pytest.importorskip('google_crc32c')
    length=struct.pack('<Q',len(payload))
    path.write_bytes(length+struct.pack('<I',tf.masked_crc(length))+payload+struct.pack('<I',tf.masked_crc(payload)))


def test_crc_and_truncation_stop_before_accepting_source(tmp_path):
    path=tmp_path/'source.tfrecord';payload=example().SerializeToString();write_record(path,payload)
    receipt={};assert list(tf.records(path,receipt))==[payload]
    assert receipt['sha256']==hashlib.sha256(path.read_bytes()).hexdigest() and receipt['episodes']==1
    data=path.read_bytes()
    for damaged in (data[:-1],data[:8]+b'bad!'+data[12:],data[:20]+bytes([data[20]^1])+data[21:]):
        path.write_bytes(damaged);receipt={}
        with pytest.raises(ValueError,match='CRC'):list(tf.records(path,receipt))
        assert not receipt


def test_schema_changes_and_nonfinite_values_are_not_silently_dropped():
    message=example();message.features.feature['unknown'].bytes_list.value.append(b'new')
    with pytest.raises(ValueError,match='Unexpected source fields'):tf.decode_episode(message.SerializeToString())
    message=example();message.features.feature['steps/action'].float_list.value[0]=float('nan')
    with pytest.raises(ValueError,match='numeric'):tf.decode_episode(message.SerializeToString())


def test_every_frame_and_all_original_fields_survive_parquet(tmp_path):
    pq=pytest.importorskip('pyarrow.parquet')
    episode=tf.decode_episode(example().SerializeToString());row=sealed_row(episode)
    reference=ff.Reference([row],{})
    value,files=ff.write_unit([episode],reference,tmp_path,0,0)
    table=pq.read_table(next(iter(files.values())));rows=table.to_pylist()
    assert len(rows)==10 and value['frames']=={'train':10}
    assert json.loads(table.schema.metadata[b'huggingface'])['info']['features']['image']['_type']=='Image'
    for i,result in enumerate(rows):
        assert result['timestep']==i and result['source_episode_index']==0
        for name in ('image','wrist_image'):assert result[name]['bytes']==episode[name][i]
        for dest,source in [('action','actions'),('state','states'),('joint_state','joints')]:assert result[dest]==episode[source][i]
        for name in ('reward','discount','is_first','is_last','is_terminal'):assert result[name]==episode[name][i]
        assert result['language_instruction']==episode['language'][i]
        assert result['episode_file_path']==episode['source_file_path']


def test_duplicates_preserve_multiplicity_and_do_not_cross_splits():
    episode=tf.decode_episode(example().SerializeToString());a=sealed_row(episode,'a');b=sealed_row(episode,'b')
    reference=ff.Reference([a,b],{});reference.consume_completed(['a'])
    assert reference.take(episode['content_sha256'],10)['episode_id']=='b'
    with pytest.raises(ValueError,match='exhausted'):reference.take(episode['content_sha256'],10)
    with pytest.raises(ValueError,match='ordering'):ff.Reference([a,b],{}).consume_completed(['b'])
    with pytest.raises(ValueError,match='crosses'):ff.Reference([a,{**b,'split':'test'}],{})


@pytest.mark.parametrize('stride',[1,4,8])
def test_query_stride_is_independent_of_horizon_and_preserves_boundaries(stride):
    frames=[dict(episode_id=episode,timestep=t,action=[float(t)]*6+[1.]) for episode in ('a','b') for t in range(10)]
    queries=list(tf.iter_action_queries(frames,query_stride=stride))
    assert [(q['episode_id'],q['timestep']) for q in queries]==[(e,t) for e in ('a','b') for t in range(0,10,stride)]
    for query in queries:
        t=query['timestep'];valid=min(8,10-t)
        assert query['action_valid_mask']==[True]*valid+[False]*(8-valid)
        assert [a[0] for a in query['raw_action_chunk'][:valid]]==list(range(t,t+valid))
        assert query['raw_action_chunk'][valid:]==[[0.]*6+[1.]]*(8-valid)
    with pytest.raises(ValueError,match='consecutive'):list(tf.iter_action_queries(frames[1:]))


class FakeApi:
    def __init__(self,root):
        self.root=root;self.files={};self.head='0'*40;self.history={self.head:{}};self.fail_unit=1;self.registered=None
    def create_repo(self,**kw):assert kw['private'] is False
    def dataset_info(self,repo,revision=None,files_metadata=False):
        files=self.history[revision] if revision else self.files
        return SimpleNamespace(private=False,sha=self.head,siblings=[SimpleNamespace(rfilename=n,size=len(b),
            lfs=SimpleNamespace(sha256=hashlib.sha256(b).hexdigest())) for n,b in files.items()])
    def create_commit(self,*,operations,parent_commit,**kw):
        assert parent_commit==self.head
        if self.fail_unit is not None and any(op.path_in_repo==f'_progress/unit-{self.fail_unit:04d}.json' for op in operations):
            self.fail_unit=None;raise ConnectionError('simulated upload interruption')
        for op in operations:
            source=op.path_or_fileobj
            self.files[op.path_in_repo]=Path(source).read_bytes() if isinstance(source,(str,Path)) else source.getvalue()
        self.head=f'{len(self.history)+1:040x}';self.history[self.head]=dict(self.files)
        return SimpleNamespace(oid=self.head)
    def download(self,repo,name,*,repo_type,revision):
        from huggingface_hub.errors import EntryNotFoundError
        source=self.history[revision]
        if name not in source:raise EntryNotFoundError(name)
        path=self.root/'downloads'/revision/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(source[name]);return str(path)


def test_restart_reuses_only_verified_units_and_finalizes_all_frames(tmp_path,monkeypatch):
    pytest.importorskip('pyarrow');import huggingface_hub
    payload=example(3).SerializeToString();episode=tf.decode_episode(payload)
    rows=[sealed_row(episode,str(i)) for i in range(2)]
    values={'reference/splits.json':rows,'reference/normalization.json':{'train_only':True},
        'source/dataset_info.json':{'name':'libero_mix','fileFormat':'tfrecord','splits':[{'name':'train','shardLengths':[1,1]}]},
        'source/features.json':{}}
    paths={}
    for name,value in values.items():
        path=tmp_path/'reference'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(ff.json_bytes(value));paths[name]=path
    monkeypatch.setattr(ff,'REF_HASHES',{n:ff.sha256(p) for n,p in paths.items()})
    monkeypatch.setattr(ff,'COUNTS',{'train':2});monkeypatch.setattr(ff,'FRAME_COUNTS',{'train':6})
    original=ff.load_reference;monkeypatch.setattr(ff,'load_reference',lambda paths:original(paths,strict=False))
    source=tmp_path/'source';source.mkdir()
    for name in ('dataset_info.json','features.json'):(source/name).write_bytes(paths['source/'+name].read_bytes())
    for index in range(2):write_record(source/f'libero_mix-train.tfrecord-{index:05d}-of-00002',payload)
    api=FakeApi(tmp_path);monkeypatch.setattr(huggingface_hub,'hf_hub_download',api.download)
    monkeypatch.setattr(ff,'update_registry',lambda api,key,record:setattr(api,'registered',(key,record)))
    reads=[];reader=ff.records
    def count_reads(path,receipt):reads.append(path.name);yield from reader(path,receipt)
    monkeypatch.setattr(ff,'records',count_reads)
    with pytest.raises(ConnectionError):ff.run_export(source,paths,tmp_path/'work',api,shards_per_unit=1)
    assert '_progress/unit-0000.json' in api.files and 'COMPLETE' not in api.files and api.registered is None
    reads.clear();ff.run_export(source,paths,tmp_path/'work',api,shards_per_unit=1)
    assert reads==['libero_mix-train.tfrecord-00001-of-00002']
    manifest=json.loads(api.files['dataset_manifest.json']);complete=json.loads(api.files['COMPLETE'])
    assert manifest['total_frames']==6 and manifest['total_episodes']==2
    assert hashlib.sha256(api.files['dataset_manifest.json']).hexdigest()==complete['manifest_sha256']
    assert api.registered[0]==ff.KEY and api.registered[1]['revision']==api.head
    reads.clear();ff.run_export(source,paths,tmp_path/'work',api,shards_per_unit=1);assert reads==[]


def test_missing_units_never_publish_complete_marker(tmp_path):
    api=FakeApi(tmp_path);hub=ff.HubExport(api,'test/public',{'num_units':1})
    with pytest.raises(ValueError,match='missing'):hub.finish([],ff.Reference([],{}))
    assert not api.files


def test_fullframe_cell_uses_hf_and_stays_disabled_by_default(capsys):
    from vlade.utils.fullframes_notebook import export_cell
    cell = export_cell(Path(os.environ["VLADE_PROJECT_ROOT"]))
    source = ''.join(cell['source'])
    assert "hf_input('raw_rlds', DATA_PY)" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "CUDA_VISIBLE_DEVICES=''" in source
    assert '/content/drive' not in source and 'files_sha256' not in source
    exec(compile(source, 'export-cell', 'exec'), {})
    assert 'RUN_FULLFRAME_EXPORT=False' in capsys.readouterr().out
