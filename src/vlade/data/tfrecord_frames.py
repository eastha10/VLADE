"""TensorFlow-free, CRC-checked reader for the pinned LIBERO+ TFRecord schema."""
from __future__ import annotations

from collections import deque
import hashlib
import math
from pathlib import Path
import struct
from typing import Iterator

SOURCE_KEYS=frozenset({'steps/action','steps/observation/state','steps/observation/joint_state',
    'steps/observation/image','steps/observation/wrist_image','steps/language_instruction',
    'steps/is_first','steps/is_last','steps/is_terminal','steps/reward','steps/discount','episode_metadata/file_path'})
_EXAMPLE=None


def example_type():
    global _EXAMPLE
    if _EXAMPLE is not None:return _EXAMPLE
    from google.protobuf import descriptor_pb2,descriptor_pool,message_factory
    schema=descriptor_pb2.FileDescriptorProto(name='vlade_example.proto',package='vlade_tf',syntax='proto3')
    def field(parent,name,number,kind,*,repeated=False,type_name=None):
        item=parent.field.add(name=name,number=number,type=kind,label=3 if repeated else 1)
        if type_name:item.type_name=type_name
        return item
    for name,kind in [('BytesList',12),('FloatList',2),('Int64List',3)]:
        item=schema.message_type.add(name=name);field(item,'value',1,kind,repeated=True)
    feature=schema.message_type.add(name='Feature');feature.oneof_decl.add(name='kind')
    for number,name,kind in [(1,'bytes_list','BytesList'),(2,'float_list','FloatList'),(3,'int64_list','Int64List')]:
        field(feature,name,number,11,type_name='.vlade_tf.'+kind).oneof_index=0
    features=schema.message_type.add(name='Features');entry=features.nested_type.add(name='Entry');entry.options.map_entry=True
    field(entry,'key',1,9);field(entry,'value',2,11,type_name='.vlade_tf.Feature')
    field(features,'feature',1,11,repeated=True,type_name='.vlade_tf.Features.Entry')
    example=schema.message_type.add(name='Example');field(example,'features',1,11,type_name='.vlade_tf.Features')
    pool=descriptor_pool.DescriptorPool();pool.Add(schema)
    _EXAMPLE=message_factory.GetMessageClass(pool.FindMessageTypeByName('vlade_tf.Example'))
    return _EXAMPLE


def masked_crc(data: bytes) -> int:
    import google_crc32c
    value=google_crc32c.value(data)
    return (((value>>15)|(value<<17))+0xa282ead8)&0xffffffff


def records(path: Path, receipt: dict) -> Iterator[bytes]:
    """Check both TFRecord CRC fields and hash the complete source file in one read."""
    digest=hashlib.sha256();count=0;total=0;before=path.stat()
    with path.open('rb') as source:
        while True:
            header=source.read(12)
            if not header:break
            if len(header)!=12 or masked_crc(header[:8])!=struct.unpack('<I',header[8:])[0]:
                raise ValueError('TFRecord length/CRC mismatch: '+path.name)
            size=struct.unpack('<Q',header[:8])[0]
            if size>512*1024**2:raise ValueError('Unexpected oversized TFRecord example')
            data=source.read(size);footer=source.read(4)
            if len(data)!=size or len(footer)!=4 or masked_crc(data)!=struct.unpack('<I',footer)[0]:
                raise ValueError('TFRecord payload/CRC mismatch: '+path.name)
            digest.update(header);digest.update(data);digest.update(footer)
            count+=1;total+=12+size+4
            yield data
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source TFRecord changed during export')
    receipt.update(path=path.name,bytes=total,sha256=digest.hexdigest(),episodes=count,mtime_ns=before.st_mtime_ns)


def decode_episode(data: bytes) -> dict:
    feature=example_type().FromString(data).features.feature
    if set(feature)!=SOURCE_KEYS:raise ValueError('Unexpected source fields; refusing to discard unknown fields')
    def get(name,kind):
        item=feature[name]
        if item.WhichOneof('kind')!=kind:raise ValueError('Unexpected TF feature type: '+name)
        return list(getattr(item,kind).value)
    actions=get('steps/action','float_list')
    if not actions or len(actions)%7:raise ValueError('Invalid action trajectory length')
    length=len(actions)//7
    def numeric(name,dimension):
        values=get(name,'float_list')
        if len(values)!=length*dimension or not all(math.isfinite(x) for x in values):raise ValueError('Invalid numeric trajectory: '+name)
        return [values[t*dimension:(t+1)*dimension] for t in range(length)]
    episode=dict(actions=numeric('steps/action',7),states=numeric('steps/observation/state',8),
        joints=numeric('steps/observation/joint_state',7),steps=length)
    for key,name in [('image','image'),('wrist_image','wrist_image')]:
        values=get('steps/observation/'+name,'bytes_list')
        if len(values)!=length or any(not value.startswith(b'\xff\xd8') for value in values):raise ValueError('Invalid source JPEG payload')
        episode[key]=values
    language=get('steps/language_instruction','bytes_list')
    if len(language)!=length:raise ValueError('Invalid language trajectory')
    episode['language']=[value.decode('utf-8') for value in language]
    if any(not value.strip() for value in episode['language']):raise ValueError('Missing instruction')
    for key in ('is_first','is_last','is_terminal'):
        values=get('steps/'+key,'int64_list')
        if len(values)!=length or any(value not in {0,1} for value in values):raise ValueError('Invalid episode flags')
        episode[key]=[bool(value) for value in values]
    if episode['is_first']!=[True]+[False]*(length-1) or episode['is_last']!=[False]*(length-1)+[True]:
        raise ValueError('Invalid episode boundary flags')
    for key in ('reward','discount'):
        episode[key]=[value[0] for value in numeric('steps/'+key,1)]
    path=get('episode_metadata/file_path','bytes_list')
    if len(path)!=1:raise ValueError('Missing original episode file path')
    episode['source_file_path']=path[0].decode('utf-8')
    digest=hashlib.sha256()
    for t in range(length):
        for value in (struct.pack('<7f',*episode['actions'][t]),struct.pack('<8f',*episode['states'][t]),
                      language[t],episode['image'][t],episode['wrist_image'][t]):
            digest.update(len(value).to_bytes(8,'little'));digest.update(value)
    episode['content_sha256']=digest.hexdigest()
    return episode


def iter_action_queries(frames, *, query_stride: int=8, action_horizon: int=8):
    """Produce raw, contiguous action chunks; sampling never changes stored frames."""
    if query_stride not in {1,4,8} or action_horizon!=8:raise ValueError('Use query stride 1/4/8 with action horizon 8')
    window=deque();current=None;expected=0
    def sample():
        first=window[0];valid=min(len(window),action_horizon)
        actions=[row['action'] for row in list(window)[:valid]]
        actions += [[0.]*6+[actions[-1][6]] for _ in range(action_horizon-valid)]
        return dict(episode_id=first['episode_id'],timestep=first['timestep'],frame=first,
                    raw_action_chunk=actions,action_valid_mask=[True]*valid+[False]*(action_horizon-valid),normalization_applied=False)
    for frame in frames:
        if frame['episode_id']!=current:
            while window:
                if window[0]['timestep']%query_stride==0:yield sample()
                window.popleft()
            current=frame['episode_id'];expected=0
        if frame['timestep']!=expected:raise ValueError('Frames must be consecutive within each complete episode')
        expected+=1;window.append(frame)
        if len(window)==action_horizon:
            if window[0]['timestep']%query_stride==0:yield sample()
            window.popleft()
    while window:
        if window[0]['timestep']%query_stride==0:yield sample()
        window.popleft()
