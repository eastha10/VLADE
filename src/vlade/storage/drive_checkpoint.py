"""Read a copied sealed checkpoint without renaming or trusting Drive mount aliases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import unicodedata


def canonical_name(name: str) -> str:
    name=unicodedata.normalize('NFC',name).strip()
    while True:
        value=re.sub(r'(?:\s*의\s+사본|\s+-\s+Copy)(?:\s*\(\d+\))?$', '',name).rstrip()
        if value==name:return name
        name=value


def relative_path(name: str) -> Path:
    path=PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe checkpoint file path')
    return Path(*path.parts)


def checkpoint_sources(directory: Path) -> tuple[dict,dict]:
    candidates={}
    observed=[]
    for path in directory.rglob('*'):
        observed.append(path.relative_to(directory).as_posix())
        if not path.is_file():continue
        name='/'.join(canonical_name(part) for part in path.relative_to(directory).parts)
        if name in candidates:raise ValueError('Ambiguous copied checkpoint file names')
        candidates[name]=path
    missing=sorted({'checkpoint_manifest.json','COMPLETE'}-set(candidates))
    if missing:
        raise ValueError('Completed sealed checkpoint required; missing='+repr(missing)+'; observed='+repr(observed[:40]))
    seal=json.loads(candidates['checkpoint_manifest.json'].read_text(encoding='utf-8'))
    names=set(seal['files'])|{'checkpoint_manifest.json','COMPLETE'}
    if not names<=set(candidates):raise ValueError('Copied checkpoint is missing sealed files: '+repr(sorted(names-set(candidates))))
    if not {'trainer_state.pt','state.pt'}&names:raise ValueError('Resumable checkpoint state required')
    for name in names:relative_path(name)
    return {name:candidates[name] for name in names},seal


def file_hash(path: Path) -> str:
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):result.update(block)
    return result.hexdigest()


def prepare_colab_checkpoint(project_root: Path, work_root: Path, folder_id: str, *, expected_step: int=1975) -> Path:
    for manifest in (project_root/'checkpoints/fft').rglob('checkpoint_manifest.json*'):
        try:
            _,seal=checkpoint_sources(manifest.parent)
            if seal['step']==expected_step:
                print(json.dumps({'status':'using_mounted_checkpoint','path':str(manifest.parent)}),flush=True)
                return manifest.parent
        except (ValueError,OSError) as error:
            print(json.dumps({'status':'mounted_checkpoint_unavailable','message':str(error)},ensure_ascii=False),flush=True)
    print('마운트 파일명 대신 Drive 파일 ID로 체크포인트를 읽습니다. 필요하면 Google 계정 접근을 허용하십시오.',flush=True)
    from google.colab import auth
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    auth.authenticate_user()
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
    with AuthorizedSession(credentials) as session:
        return stage_drive_checkpoint(session,folder_id,work_root,expected_step=expected_step)


def stage_drive_checkpoint(session, folder_id: str, work_root: Path, *, expected_step: int=1975) -> Path:
    """Use an authorized Google session; stage only seal-verified bytes on runtime SSD."""
    import requests
    if not re.fullmatch(r'[A-Za-z0-9_-]+',folder_id):raise ValueError('Invalid checkpoint Drive folder ID')
    base='https://www.googleapis.com/drive/v3/files'
    files={};page=None
    while True:
        response=session.get(base,params={'q':f"'{folder_id}' in parents and trashed = false",'fields':'nextPageToken,files(id,name,size,mimeType)',
            'pageSize':1000,'pageToken':page or '', 'supportsAllDrives':'true','includeItemsFromAllDrives':'true'},timeout=120)
        response.raise_for_status();listing=response.json()
        for item in listing['files']:
            name=canonical_name(item['name'])
            if name in files:raise ValueError('Ambiguous Drive checkpoint file names')
            files[name]=item
        page=listing.get('nextPageToken')
        if not page:break
    if not {'checkpoint_manifest.json','COMPLETE'}<=set(files):raise ValueError('Drive checkpoint is not sealed and complete')
    if int(files['checkpoint_manifest.json']['size'])>1024**2:raise ValueError('Unexpected checkpoint manifest size')
    def media(item,**kwargs):
        return session.get(base+'/'+item['id'],params={'alt':'media','supportsAllDrives':'true'},timeout=120,**kwargs)
    response=media(files['checkpoint_manifest.json']);response.raise_for_status();seal_bytes=response.content
    seal=json.loads(seal_bytes)
    if seal.get('step')!=expected_step:raise ValueError('Unexpected Drive checkpoint step')
    if not {'trainer_state.pt','state.pt'}&set(seal['files']):raise ValueError('Resumable checkpoint state required')
    expected=dict(seal['files']);expected['checkpoint_manifest.json']={'bytes':len(seal_bytes),'sha256':hashlib.sha256(seal_bytes).hexdigest()}
    if not set(expected)<=set(files):raise ValueError('Drive checkpoint is missing sealed files')
    for name,row in expected.items():
        relative_path(name)
        if int(files[name]['size'])!=row['bytes']:raise ValueError('Drive source size differs from seal: '+name)
    destination=work_root/folder_id/hashlib.sha256(seal_bytes).hexdigest()[:16]
    destination.mkdir(parents=True,exist_ok=True)
    remaining=0
    for name,row in expected.items():
        target=destination/relative_path(name);partial=target.with_name('.'+target.name+'.partial')
        present=max((path.stat().st_size for path in (target,partial) if path.is_file()),default=0)
        remaining+=max(0,row['bytes']-present)
    if shutil.disk_usage(destination).free<remaining+2*1024**3:raise OSError('Insufficient Colab SSD space for the sealed checkpoint copy')
    for name,row in sorted(expected.items()):
        target=destination/relative_path(name);target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink() or not target.resolve().is_relative_to(destination.resolve()):raise ValueError('Checkpoint staging must not contain symlinks')
        if target.is_file() and target.stat().st_size==row['bytes'] and file_hash(target)==row['sha256']:
            print(json.dumps({'status':'drive_file_reused','file':name}),flush=True);continue
        if target.is_file():target.unlink()  # Only this owned, seal-addressed runtime cache is replaced.
        partial=target.with_name('.'+target.name+'.partial')
        if partial.is_symlink():raise ValueError('Invalid checkpoint staging partial')
        print(json.dumps({'status':'drive_file_download','file':name,'bytes':row['bytes']}),flush=True)
        for attempt in range(5):
            offset=partial.stat().st_size if partial.is_file() else 0
            if offset>row['bytes']:partial.unlink();offset=0
            if offset==row['bytes']:break
            try:
                with media(files[name],headers={'Range':f'bytes={offset}-'},stream=True) as response:
                    response.raise_for_status()
                    if response.status_code==206:
                        span=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                        if not span or int(span[1])!=offset or int(span[3])!=row['bytes']:raise ValueError('Invalid Drive byte range')
                    elif response.status_code==200:offset=0
                    else:raise ValueError('Unexpected Drive media response')
                    last=time.monotonic()
                    with partial.open('ab' if offset else 'wb') as stream:
                        for block in response.iter_content(8*1024**2):
                            if not block:continue
                            offset+=len(block)
                            if offset>row['bytes']:raise ValueError('Drive file exceeds sealed size')
                            stream.write(block)
                            if time.monotonic()-last>=5:
                                print(json.dumps({'status':'drive_download_progress','file':name,'bytes':offset,'total':row['bytes']}),flush=True);last=time.monotonic()
                break
            except requests.RequestException as error:
                status=getattr(getattr(error,'response',None),'status_code',None)
                if attempt==4 or status is not None and status<500 and status not in {408,429}:raise
                time.sleep(min(attempt+1,3))
        if not partial.is_file() or partial.stat().st_size!=row['bytes'] or file_hash(partial)!=row['sha256']:
            if partial.is_file():partial.unlink()
            raise ValueError('Downloaded Drive checkpoint hash mismatch: '+name)
        partial.replace(target)
        print(json.dumps({'status':'drive_file_verified','file':name}),flush=True)
    response=media(files['COMPLETE']);response.raise_for_status()
    if len(response.content)!=int(files['COMPLETE']['size']) or not response.content:raise ValueError('Invalid Drive COMPLETE marker')
    (destination/'COMPLETE').write_bytes(response.content)
    checkpoint_sources(destination)
    return destination
