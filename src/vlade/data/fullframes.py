"""Lossless full-frame Parquet export of the sealed LIBERO+ D/seed42 source."""
from __future__ import annotations

from collections import Counter,deque
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shutil

from vlade.data.tfrecord_frames import decode_episode,records
from vlade.storage.assets import configure_http,sha256,safe_relative
from vlade.storage.migrate import safe_error,update_registry

REPO='VLADE-konkuk/libero-plus-fullframes'
KEY='fullframes_d_seed42'
SCHEMA_VERSION=1
REF_HASHES={
 'reference/splits.json':'379376e5cfb19d0b9d0a5bdd548e01e4c658f09de3fdd1bff178b3aad7ed9900',
 'reference/normalization.json':'e1dbf29ea83cc2e0fbcb90e68a7ad15660eb210baa5a333d20bd16c80d275af3',
 'source/dataset_info.json':'d30a03246c8e4c6380780fa98eb23f9cae8dce427555c2af0725d85f98a420e8',
 'source/features.json':'d6f56537585d15aa99772c6da942e5957cceda26b57a18098ef8e3164cdaffde'}
COUNTS={'train':11477,'validation':1428,'test':1442}
FRAME_COUNTS={'train':1791269,'validation':223485,'test':223282}


def emit(status,**values):print(json.dumps(dict(status=status,**values),ensure_ascii=False),flush=True)
def json_bytes(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')


class Reference:
    """Consume sealed episode identities, preserving duplicate multiplicity and split labels."""
    def __init__(self,rows,normalization):
        self.rows=rows;self.normalization=normalization;self.by_id={r['episode_id']:r for r in rows}
        if len(self.by_id)!=len(rows):raise ValueError('Duplicate reference episode ID')
        self.available={};self.used=set();groups={}
        for row in rows:
            if row['split'] not in COUNTS or row['steps']<1 or row['queries']!=(row['steps']+7)//8:
                raise ValueError('Invalid reference split/step/query metadata')
            if len(row['content_sha256'])!=64 or type(row['is_calibration']) is not bool:raise ValueError('Invalid canonical identity')
            label=(row['split'],row['is_calibration'])
            for group in (row['content_group_id'],row['content_sha256']):
                if group in groups and groups[group]!=label:raise ValueError('Duplicate content crosses split/calibration')
                groups[group]=label
            self.available.setdefault(row['content_sha256'],deque()).append(row)

    def take(self,digest,steps):
        queue=self.available.get(digest)
        if not queue:raise ValueError('Source episode is absent/exhausted in canonical D: '+digest)
        row=queue[0]
        if row['steps']!=steps:raise ValueError('Source episode length differs from canonical D')
        queue.popleft();self.used.add(row['episode_id']);return row

    def consume_completed(self,ids):
        for episode_id in ids:
            if episode_id not in self.by_id:raise ValueError('Unknown completed episode ID')
            row=self.by_id[episode_id]
            if self.take(row['content_sha256'],row['steps'])['episode_id']!=episode_id:
                raise ValueError('Completed-unit duplicate ordering differs from the sealed reference')


def load_reference(paths,*,strict=True):
    if strict:
        for name,digest in REF_HASHES.items():
            if sha256(paths[name])!=digest:raise ValueError('Reference/source metadata hash mismatch: '+name)
    rows=json.loads(paths['reference/splits.json'].read_text('utf-8'))
    norm=json.loads(paths['reference/normalization.json'].read_text('utf-8'))
    reference=Reference(rows,norm)
    if strict and (dict(Counter(r['split'] for r in rows))!=COUNTS or
        {s:sum(r['steps'] for r in rows if r['split']==s) for s in COUNTS}!=FRAME_COUNTS or
        sum(r['is_calibration'] for r in rows)!=728 or not norm.get('train_only')):
        raise ValueError('Reference does not match canonical D/seed42')
    return reference


def frame_schema():
    import pyarrow as pa
    value=lambda dtype:{'dtype':dtype,'_type':'Value'}
    feature={name:value(dtype) for name,dtype in {
        'episode_id':'string','content_group_id':'string','content_sha256':'string','source_episode_index':'int64',
        'timestep':'int32','is_calibration':'bool','language_instruction':'string','episode_file_path':'string',
        'reward':'float32','discount':'float32','is_first':'bool','is_last':'bool','is_terminal':'bool'}.items()}
    for name,length in [('action',7),('state',8),('joint_state',7)]:
        feature[name]={'feature':value('float32'),'length':length,'_type':'Sequence'}
    feature.update(image={'_type':'Image'},wrist_image={'_type':'Image'})
    image=pa.struct([('bytes',pa.binary()),('path',pa.string())])
    types={'string':pa.string(),'int64':pa.int64(),'int32':pa.int32(),'bool':pa.bool_(),'float32':pa.float32()}
    fields=[]
    for name,item in feature.items():
        dtype=image if item['_type']=='Image' else (pa.list_(pa.float32(),item['length']) if item['_type']=='Sequence' else types[item['dtype']])
        fields.append(pa.field(name,dtype))
    schema=pa.schema(fields,metadata={b'huggingface':json.dumps({'info':{'features':feature}}).encode()})
    return schema,feature


def frame_rows(episode,row,source_index):
    for t in range(episode['steps']):
        yield dict(episode_id=row['episode_id'],content_group_id=row['content_group_id'],content_sha256=row['content_sha256'],
            source_episode_index=source_index,timestep=t,is_calibration=row['is_calibration'],
            image={'bytes':episode['image'][t],'path':None},wrist_image={'bytes':episode['wrist_image'][t],'path':None},
            action=episode['actions'][t],state=episode['states'][t],joint_state=episode['joints'][t],
            language_instruction=episode['language'][t],episode_file_path=episode['source_file_path'],
            **{key:episode[key][t] for key in ('reward','discount','is_first','is_last','is_terminal')})


def write_unit(episodes,reference,output: Path,unit:int,source_start:int):
    """Write complete episodes to per-split Parquet files; JPEG bytes are never re-encoded."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    output.mkdir(parents=True,exist_ok=True)
    schema,_=frame_schema();writers={};files={};counts=Counter();ids=[]
    try:
        for index,episode in enumerate(episodes,start=source_start):
            row=reference.take(episode['content_sha256'],episode['steps']);split=row['split'];ids.append(row['episode_id'])
            if split not in writers:
                path=output/f'{split}-part-{unit:04d}.parquet';files[split]=path
                writers[split]=pq.ParquetWriter(path,schema,compression='zstd',write_page_checksum=True,
                    use_dictionary=['episode_id','content_group_id','content_sha256','language_instruction','episode_file_path'])
            table=pa.Table.from_pylist(list(frame_rows(episode,row,index)),schema=schema)
            writers[split].write_table(table,row_group_size=episode['steps']);counts[split]+=episode['steps']
            emit('episode_prepared',unit=unit,episode_id=row['episode_id'],frames=episode['steps'],episodes_in_unit=len(ids))
    finally:
        for writer in writers.values():writer.close()
    rows={}
    for split,path in files.items():
        if pq.read_metadata(path).num_rows!=counts[split]:raise ValueError('Parquet row count mismatch')
        rows['data/'+path.name]={'bytes':path.stat().st_size,'sha256':sha256(path),'frames':counts[split],'split':split}
    return dict(unit=unit,episode_ids=ids,frames=dict(counts),files=rows),{'data/'+p.name:p for p in files.values()}


def export_config(paths,shards_per_unit=8):
    info=json.loads(paths['source/dataset_info.json'].read_text('utf-8'));split=info['splits'][0]
    if len(info['splits'])!=1 or split['name']!='train' or info['fileFormat']!='tfrecord':raise ValueError('Unexpected source TFDS layout')
    if shards_per_unit<1:raise ValueError('shards_per_unit must be positive')
    code={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('tfrecord_frames.py'))}
    return dict(schema_version=SCHEMA_VERSION,kind='fullframes_parquet',source_repo='Sylvest/libero_plus_rlds',
        source_revision='fb0c7029b076030d5d57227229e4f7460def1f7c',source_metadata_hashes={k:sha256(v) for k,v in paths.items()},
        stored_frame_interval_steps=1,action_horizon=8,pixels='original_JPEG_bytes_no_resize_no_reencoding',
        split_policy='canonical_D_content_group_80_10_10_seed_42',seed=42,episode_counts=COUNTS,frame_counts=FRAME_COUNTS,
        source_shard_lengths=[int(n) for n in split['shardLengths']],source_name=info['name'],
        shards_per_unit=shards_per_unit,num_units=math.ceil(len(split['shardLengths'])/shards_per_unit),exporter_source_hashes=code)


def ensure_public_repo(api,repo):
    api.create_repo(repo_id=repo,repo_type='dataset',private=False,exist_ok=True)
    if api.dataset_info(repo).private is not False:raise ValueError('Existing destination must be public; private repositories are never made public automatically')


class HubExport:
    def __init__(self,api,repo,config):
        self.api=api;self.repo=repo;self.config=config;self.identity=hashlib.sha256(json_bytes(config)).hexdigest()

    def read_json(self,name,revision=None):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        head=revision or self.api.dataset_info(self.repo).sha
        try:return json.loads(Path(hf_hub_download(self.repo,name,repo_type='dataset',revision=head)).read_text('utf-8'))
        except EntryNotFoundError:return None

    def verify_files(self,rows,revision):
        from huggingface_hub import hf_hub_download
        info=self.api.dataset_info(self.repo,revision=revision,files_metadata=True)
        if info.private is not False:raise ValueError('Public dataset visibility changed')
        remote={item.rfilename:item for item in info.siblings}
        for name,row in rows.items():
            item=remote.get(name)
            if item is None or item.size!=row['bytes']:raise ValueError('Remote export file size mismatch: '+name)
            if item.lfs:
                digest=getattr(item.lfs,'sha256',None)
                if digest is None and isinstance(item.lfs,dict):digest=item.lfs.get('sha256')
            else:
                if row['bytes']>10*1024**2:raise ValueError('Unexpected large Git blob')
                digest=sha256(Path(hf_hub_download(self.repo,name,repo_type='dataset',revision=revision)))
            if digest!=row['sha256']:raise ValueError('Remote export SHA256 mismatch: '+name)

    def initialize(self,paths):
        from huggingface_hub import CommitOperationAdd
        ensure_public_repo(self.api,self.repo)
        current=self.read_json('export_config.json')
        if current is not None:
            if current!=self.config:raise ValueError('Destination contains a different immutable export; use a new repository')
            self.verify_files({name:{'bytes':path.stat().st_size,'sha256':sha256(path)} for name,path in paths.items()},self.api.dataset_info(self.repo).sha)
            return self.api.dataset_info(self.repo).sha
        existing={item.rfilename for item in self.api.dataset_info(self.repo).siblings}
        if existing- {'.gitattributes'}:raise ValueError('Refusing to replace an unrelated dataset')
        card=dataset_card(self.config,complete=False)
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in paths.items()]
        operations += [CommitOperationAdd(path_in_repo='export_config.json',path_or_fileobj=io.BytesIO(json_bytes(self.config))),
                       CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=io.BytesIO(card.encode()))]
        commit=self.api.create_commit(repo_id=self.repo,repo_type='dataset',operations=operations,parent_commit=self.api.dataset_info(self.repo).sha,
                                      commit_message='Initialize approved public full-frame D/seed42 export')
        self.verify_files({name:{'bytes':path.stat().st_size,'sha256':sha256(path)} for name,path in paths.items()},commit.oid)
        return commit.oid

    def completed_unit(self,unit):
        head=self.api.dataset_info(self.repo).sha
        value=self.read_json(f'_progress/unit-{unit:04d}.json',head)
        if value is None:return None
        if value.get('identity')!=self.identity or value.get('unit')!=unit:raise ValueError('Completed unit belongs to a different export')
        self.verify_files(value['files'],head);return value

    def publish_unit(self,value,files):
        from huggingface_hub import CommitOperationAdd
        value={**value,'identity':self.identity}
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in files.items()]
        operations.append(CommitOperationAdd(path_in_repo=f"_progress/unit-{value['unit']:04d}.json",path_or_fileobj=io.BytesIO(json_bytes(value))))
        emit('unit_uploading',unit=value['unit'],files=len(files),bytes=sum(row['bytes'] for row in value['files'].values()))
        existing=self.completed_unit(value['unit'])
        if existing is not None:
            if existing!=value:raise ValueError('Concurrent exporter published a different unit')
            return existing
        parent=self.api.dataset_info(self.repo).sha
        try:
            commit=self.api.create_commit(repo_id=self.repo,repo_type='dataset',operations=operations,parent_commit=parent,
                commit_message=f"Export verified full-frame source unit {value['unit']:04d}")
            self.verify_files(value['files'],commit.oid)
        except Exception:
            found=self.completed_unit(value['unit'])
            if found!=value:raise
        emit('unit_published_verified',unit=value['unit'],frames=value['frames']);return value

    def finish(self,units,reference):
        from huggingface_hub import CommitOperationAdd
        if len(units)!=self.config['num_units'] or reference.used!=set(reference.by_id):raise ValueError('Export still has missing episodes or units')
        counts=Counter();files={}
        for unit in units:
            if unit['identity']!=self.identity:raise ValueError('Mixed export identity')
            counts.update(unit['frames']);files.update(unit['files'])
        if sorted(unit['unit'] for unit in units)!=list(range(self.config['num_units'])):
            raise ValueError('Missing or duplicate export units')
        ids=[episode_id for unit in units for episode_id in unit['episode_ids']]
        if len(ids)!=len(reference.rows) or set(ids)!=set(reference.by_id):raise ValueError('Missing or repeated exported episode identities')
        if dict(counts)!=self.config['frame_counts']:raise ValueError('Export frame counts differ from canonical source')
        parent=self.api.dataset_info(self.repo).sha
        self.verify_files(files,parent)
        manifest={**self.config,'identity':self.identity,'files':files,'total_episodes':len(reference.rows),'total_frames':sum(counts.values()),
                  'normalization_applied':False,'raw_action_convention':'original_LIBERO_RLDS; use sealed normalization only when training'}
        data=json_bytes(manifest);complete=json_bytes({'identity':self.identity,'manifest_sha256':hashlib.sha256(data).hexdigest()})
        commit=self.api.create_commit(repo_id=self.repo,repo_type='dataset',operations=[
            CommitOperationAdd(path_in_repo='dataset_manifest.json',path_or_fileobj=io.BytesIO(data)),
            CommitOperationAdd(path_in_repo='COMPLETE',path_or_fileobj=io.BytesIO(complete)),
            CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=io.BytesIO(dataset_card(manifest,complete=True).encode()))],
            parent_commit=parent,commit_message='Complete and seal full-frame canonical D/seed42 dataset')
        self.verify_files(files,commit.oid)
        self.register(commit.oid)
        emit('fullframes_export_completed',repo_id=self.repo,revision=commit.oid,episodes=len(reference.rows),frames=sum(counts.values()))

    def register(self,revision):
        from huggingface_hub import hf_hub_download
        names=['dataset_manifest.json','COMPLETE','reference/splits.json','reference/normalization.json']
        metadata={}
        for name in names:
            path=Path(hf_hub_download(self.repo,name,repo_type='dataset',revision=revision))
            metadata[name]={'bytes':path.stat().st_size,'sha256':sha256(path)}
        update_registry(self.api,KEY,dict(repo_id=self.repo,repo_type='dataset',revision=revision,format='parquet_frames',
            local_name=KEY,metadata_files=metadata,stored_frame_interval_steps=1,action_horizon=8))


def dataset_card(config,*,complete):
    yaml='---\nlicense: mit\ntags:\n- robotics\n- vision-language-action\n- full-frames\n'
    if complete:
        yaml+='configs:\n- config_name: fullframes_d_seed42\n  default: true\n  data_files:\n'
        for split in COUNTS:yaml+=f'  - split: {split}\n    path: data/{split}-part-*.parquet\n'
    yaml+='---\n'
    return yaml+f'''# LIBERO+ full frames · canonical D seed42

Status: {"COMPLETE — sealed and verified" if complete else "PREPARING — do not use partial data for training"}.

Derived from [Sylvest/libero_plus_rlds](https://huggingface.co/datasets/Sylvest/libero_plus_rlds/tree/fb0c7029b076030d5d57227229e4f7460def1f7c), retaining its MIT license and attribution. Original benchmark: [LIBERO+](https://arxiv.org/abs/2510.13626).

All 14,347 episodes / 2,238,036 timesteps are retained. Each row is one timestep with both original 256x256 JPEG byte payloads; no resizing or JPEG re-encoding is performed. Raw 7-D actions, 8-D state, 7-D joint state, language, reward, discount, first/last/terminal flags, and original source episode file metadata are preserved. `source_episode_index` records source TFRecord order. Local Drive paths, credentials and private model weights are not added.

Canonical D episode counts: train 11,477; validation 1,428; test 1,442. Frame counts: train 1,791,269; validation 223,485; test 223,282. Duplicate content keeps its sealed split and calibration labels; identical duplicates consume sealed episode IDs deterministically. The 728 calibration episodes remain part of Train.

`reference/splits.json` and `reference/normalization.json` preserve the existing D/seed42 reference bytes. Numeric data is raw: gripper conversion and train-only normalization are applied by the training consumer, not during storage. Do not frame-shuffle before constructing action chunks or cross episode boundaries.

Stored frame interval is 1. Training query stride (1/4/8) is independent of action horizon (8). Existing sparse datasets/checkpoints are not replaced. Use immutable revisions and check `COMPLETE` / `dataset_manifest.json` before training. Source units are uploaded atomically and remotely hash-verified; `_progress` supports restart without rereading completed source units.

After completion, standard `datasets.load_dataset("{REPO}", revision="<completed commit>", split="train", streaming=True)` reads the Parquet data. `vlade.data.tfrecord_frames.iter_action_queries` provides raw action chunks with an explicit padding mask; apply the sealed normalization before model training.
'''


def run_export(tfds_root,paths,work_root,api,repo=REPO,shards_per_unit=8):
    reference=load_reference(paths);config=export_config(paths,shards_per_unit);hub=HubExport(api,repo,config)
    hub.initialize(paths)
    complete=hub.read_json('COMPLETE')
    if complete:
        manifest=hub.read_json('dataset_manifest.json')
        if complete.get('identity')!=hub.identity or hashlib.sha256(json_bytes(manifest)).hexdigest()!=complete['manifest_sha256']:
            raise ValueError('Completed export identity/manifest mismatch')
        head=api.dataset_info(repo).sha;hub.verify_files(manifest['files'],head);hub.register(head)
        emit('fullframes_export_already_complete',repo_id=repo,revision=head);return
    infos=list(tfds_root.rglob('dataset_info.json'))
    if len(infos)!=1 or sha256(infos[0])!=REF_HASHES['source/dataset_info.json']:raise ValueError('Find exactly the pinned original TFDS dataset_info.json')
    if sha256(infos[0].parent/'features.json')!=REF_HASHES['source/features.json']:raise ValueError('Source TFDS features changed')
    lengths=config['source_shard_lengths'];n=len(lengths)
    sources=[infos[0].parent/f"{config['source_name']}-train.tfrecord-{i:05d}-of-{n:05d}" for i in range(n)]
    if any(not p.is_file() for p in sources):raise FileNotFoundError('Original TFDS shard is missing; no archive is redownloaded automatically')
    if sum(lengths)!=len(reference.rows):raise ValueError('Source/reference episode count differs')
    work_root.mkdir(parents=True,exist_ok=True);units=[]
    for index,start in enumerate(range(0,n,shards_per_unit)):
        end=min(n,start+shards_per_unit);expected=sum(lengths[start:end]);completed=hub.completed_unit(index)
        if completed is not None:
            if len(completed['episode_ids'])!=expected:raise ValueError('Completed source unit episode count mismatch')
            if (completed.get('source_shard_start'),completed.get('source_shard_end'))!=(start,end):raise ValueError('Completed source unit range mismatch')
            frame_counts=Counter()
            for episode_id in completed['episode_ids']:
                row=reference.by_id.get(episode_id)
                if row is None:raise ValueError('Unknown completed episode ID')
                frame_counts[row['split']]+=row['steps']
            if dict(frame_counts)!=completed['frames']:raise ValueError('Completed unit frame counts differ from sealed episodes')
            for name,row in completed['files'].items():
                safe_relative(name)
                if name!=f"data/{row['split']}-part-{index:04d}.parquet" or row['frames']!=frame_counts[row['split']]:
                    raise ValueError('Completed unit file metadata mismatch')
            if {row['split'] for row in completed['files'].values()}!=set(frame_counts):raise ValueError('Completed unit is missing a split file')
            reference.consume_completed(completed['episode_ids']);units.append(completed)
            emit('unit_reused_verified',unit=index,total_units=config['num_units']);continue
        size=sum(p.stat().st_size for p in sources[start:end])
        if shutil.disk_usage(work_root).free<size*1.3+1024**3:raise OSError('Insufficient local space for one export unit')
        source_receipts=[]
        def episodes():
            for i in range(start,end):
                emit('source_shard_reading',unit=index,source_shard=i,total_source_shards=n)
                receipt={};count=0
                for payload in records(sources[i],receipt):count+=1;yield decode_episode(payload)
                if count!=lengths[i]:raise ValueError('TFRecord shard episode count differs from pinned metadata')
                source_receipts.append(receipt)
        output=work_root/hub.identity[:16]/f'unit-{index:04d}'
        value,files=write_unit(episodes(),reference,output,index,sum(lengths[:start]))
        if len(value['episode_ids'])!=expected:raise ValueError('Prepared unit is incomplete')
        value.update(source_shards=source_receipts,source_shard_start=start,source_shard_end=end)
        units.append(hub.publish_unit(value,files))
        for path in files.values():
            if path.is_symlink() or not path.resolve().is_relative_to(work_root.resolve()):raise ValueError('Unexpected export staging path')
            path.unlink()
        output.rmdir()
    hub.finish(units,reference)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tfds-root',type=Path,required=True);parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--repo-id',default=REPO);parser.add_argument('--reference-revision',required=True)
    parser.add_argument('--shards-per-unit',type=int,default=8);parser.add_argument('--public',action='store_true')
    args=parser.parse_args()
    from vlade.storage.assets import resolve_input
    args.tfds_root = resolve_input(args.tfds_root)
    if not args.public:raise SystemExit('Explicit --public is required for the approved new dataset; existing private assets are never changed')
    configure_http()
    from huggingface_hub import HfApi,hf_hub_download
    try:
        paths={name:Path(hf_hub_download(args.repo_id,name,repo_type='dataset',revision=args.reference_revision)) for name in REF_HASHES}
        run_export(args.tfds_root,paths,args.work_root,HfApi(),args.repo_id,args.shards_per_unit)
    except Exception as error:
        emit('fullframes_export_failed',error_type=type(error).__name__,message=safe_error(error));raise SystemExit(1) from None


if __name__=='__main__':main()
