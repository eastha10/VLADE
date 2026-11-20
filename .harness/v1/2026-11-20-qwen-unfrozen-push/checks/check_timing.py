"""Verify instrumentation preserves updates, episode replay, and CSV append safety."""
import csv
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

source, scratch = map(Path, sys.argv[1:])
sys.path.insert(0, str(source / 'src'))
import torch
from vlade.training.loop import run_loop, make_scheduler, LoopState
from vlade.training.teacher_inference import EpisodeTeacher
from vlade.utils.timing import RunTimings

scratch.mkdir(parents=True,exist_ok=False)
stream = io.StringIO()
syncs=[]
with RunTimings(scratch,role='checks',device='cpu',stream=stream,
                synchronize=lambda:syncs.append(True)) as timer:
    with timer.measure('operation',step=1):
        pass
    try:
        with timer.measure('operation_failed',step=2):
            raise ValueError('expected')
    except ValueError:
        pass
    first_path=timer.path
rows=list(csv.DictReader(first_path.open()))
assert [r['status'] for r in rows]==['ok','failed']
assert rows[1]['error_type']=='ValueError' and len(syncs)==4
assert all(float(r['seconds'])>=0 for r in rows)
with RunTimings(scratch,role='checks',device='cpu',stream=stream) as timer:
    second_path=timer.path
assert second_path!=first_path and len(list(csv.DictReader(first_path.open())))==2

def train(timer=None):
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.fill_(.5)
    opt=torch.optim.SGD(model.parameters(),lr=.1)
    metrics=[]
    state=run_loop(modules=[model],optimizer=opt,scheduler=make_scheduler(opt,2),
        batches=[torch.tensor([[1.]]) for _ in range(4)],
        loss_fn=lambda batch,step:((model(batch)-1)**2).mean(),max_steps=2,
        accumulation_steps=2,on_step=lambda s,m:metrics.append(dict(m)),timings=timer)
    return model.weight.detach().clone(),metrics,state
plain,_,_=train()
with RunTimings(scratch,role='s0',device='cpu',stream=stream) as timer:
    timed,metrics,state=train(timer)
    train_csv=timer.path
assert torch.equal(plain,timed) and state.microsteps==4
assert all(m['s0_train_step_seconds']>=0 for m in metrics)
rows=list(csv.DictReader(train_csv.open()))
assert sum(r['event']=='s0_backward' for r in rows)==4
assert sum(r['event']=='s0_optimizer' for r in rows)==2
assert [r['is_warmup'] for r in rows if r['event']=='s0_training_step']==['True','False']

class Data:
    def episodes(self,split,order):
        yield {},[{'timestep':t} for t in (0,8,16)]
class Teacher:
    def __init__(self):self.calls=[];self.resets=0
    def reset(self):self.resets+=1
    def predict(self,sample,data):
        self.calls.append(sample['timestep'])
        return torch.full((8,7),float(sample['timestep'])),None
teacher=Teacher()
with RunTimings(scratch,role='t1',device='cpu',stream=stream) as timer:
    seq=EpisodeTeacher(teacher,Data(),timings=timer)
    action=seq.get('ep0',16)
    assert torch.equal(action,torch.full((8,7),16.))
    assert teacher.calls==[0,8,16] and teacher.resets==1
    seq.get('ep0',0)
    assert teacher.calls==[0,8,16,0] and teacher.resets==2
    teacher_csv=timer.path
rows=list(csv.DictReader(teacher_csv.open()))
assert [r['replay'] for r in rows]==['True','True','False','False']
assert [r['is_warmup'] for r in rows]==['True','True','False','False']
assert all(line.startswith('TIMING ') for line in stream.getvalue().splitlines())

# Exercise the worker entrypoint: timing logs must never corrupt JSON stdout.
from vlade.training import teacher_inference as worker
from vlade.training import fixed_cache
from vlade.data import episode_data
from vlade.storage import assets
wire,errors=io.StringIO(),io.StringIO()
signature={'checkpoint_manifest_sha256':'synthetic','temporal_feature_layer':-2}
with (patch.object(sys,'argv',['teacher','--checkpoint','unused','--data','unused','--device','cpu',
                              '--timing-dir',str(scratch),'--timing-session','protocolcheck']),
      patch.object(sys,'stdin',io.StringIO('{"episode_id":"ep0","timestep":16}\n')),
      patch.object(sys,'stdout',wire),patch.object(sys,'stderr',errors),
      patch.object(assets,'resolve_input',side_effect=Path),
      patch.object(episode_data,'EpisodeData',return_value=Data()),
      patch.object(worker,'teacher_identity',return_value=(Path('unused'),signature)),
      patch.object(fixed_cache,'SequentialTeacher',side_effect=lambda *a,**kw:Teacher())):
    worker.main()
messages=[json.loads(line) for line in wire.getvalue().splitlines()]
assert len(messages)==2 and messages[0]=={'ready':signature}
assert messages[1]['episode_id']=='ep0' and messages[1]['action'][0][0]==16
assert 'TIMING ' in errors.getvalue()

# Exercise the exact model-forward scope without loading a 7B checkpoint.
import numpy as np
policy=fixed_cache.SequentialTeacher.__new__(fixed_cache.SequentialTeacher)
policy.device=torch.device('cpu')
policy.context=None
policy.temporal_feature_layer=-2
policy._timing_predictions=0
policy.head=policy.proprio=policy.generator=None
policy.processor=lambda *a,**kw:{'pixel_values':torch.zeros(1,3,4,4)}
class Model:
    llm_dim=4
    def predict_action(self,**kw):return np.zeros((8,7),np.float32),torch.zeros(1,56,4)
policy.model=Model()
class Images:
    def image(self,*a):return None
with RunTimings(scratch,role='forward',device='cpu',stream=stream) as timer:
    policy.timings=timer
    result,hidden=policy.predict({'instruction':'test','episode_id':'ep0','timestep':0,
                                 'proprio':np.zeros(8)},Images())
    forward_csv=timer.path
assert result.shape==(8,7) and hidden.shape==(56,4)
assert list(csv.DictReader(forward_csv.open()))[0]['event']=='t1_model_forward'
print(json.dumps({'checks':'passed','optimizer_updates_equal':True,'teacher_replay_preserved':True,
                  'worker_stdout_json_only':True,'teacher_model_forward_scope':True,
                  'csv_columns':RunTimings.FIELDS,'csv_files':len(list(scratch.glob('*.csv')))}))
