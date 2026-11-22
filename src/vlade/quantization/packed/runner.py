"""Explicit HF-pinned calibration, packed export/reload and action evaluation."""
from __future__ import annotations
import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
METHODS = ('actquant', 'qvla', 'quantvla')


def configuration(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if cfg.get('schema_version') != 1 or set(cfg.get('methods', {})) != set(METHODS):
        raise ValueError('Require packed PTQ schema1 and the three independent methods')
    for name in ('checkpoint', 'data'):
        if not isinstance(cfg.get(name), str) or not cfg[name].startswith('hfasset://'):
            raise ValueError('Packed PTQ external inputs require immutable HF keys')
    for name, low in (('calibration_queries',3),('eval_queries',1),('group_size',1),('seed',0)):
        if type(cfg.get(name)) is not int or cfg[name] < low:
            raise ValueError('Invalid packed PTQ setting: '+name)
    if cfg.get('backend') not in ('auto','cuda','portable'):
        raise ValueError('Unknown packed backend')
    for name, method in cfg['methods'].items():
        budget = method.get('target_bpw')
        if (not isinstance(budget,(float,int)) or isinstance(budget,bool) or not math.isfinite(budget)
                or not (2 if name=='actquant' else 0) <= budget <= 16
                or method.get('scope') not in ('language','language_vision')):
            raise ValueError('Invalid method scope/budget')
    if cfg['methods']['quantvla'] != {'target_bpw':8.,'scope':'language'}:
        raise ValueError('The non-DiT QuantVLA route requires language W8A16')
    return cfg


def registry_path(cfg: dict) -> Path:
    path = Path(cfg['registry'])
    return path if path.is_absolute() else ROOT/path


def plan(cfg: dict) -> dict:
    from vlade.storage.assets import asset_record
    inputs = {name: asset_record(cfg[name][len('hfasset://'):], registry_path(cfg)) for name in ('checkpoint','data')}
    return {'methods':cfg['methods'], 'backend':cfg['backend'], 'downloads':False, 'gpu_execution':False,
            'inputs': {n:{k:r[k] for k in ('repo_id','repo_type','revision')} for n,r in inputs.items()},
            'representation':'packed codes; fused CUDA or bounded-tile CPU; A16',
            'calibration_queries':cfg['calibration_queries'],'eval_queries':cfg['eval_queries'],
            'rollouts':0,'automatic_winner':None}


def run(cfg: dict, method: str, output: Path) -> None:
    import torch
    from transformers import AutoProcessor
    from vlade.storage.assets import resolve_input, asset_record
    from vlade.training.run_control import verify_checkpoint
    from vlade.data.episode_data import EpisodeData
    from vlade.models.students.qwen_vla import QwenVLA
    from vlade.models.students.backbones.qwen import prepare_observation
    from vlade.evaluation.s0_quantization import observations, select_layers
    from vlade.evaluation.quantization_comparison import source_identity, protect_output, atomic_json
    from . import Sample, calibrate, replace
    from .artifact import save, _load_owned, qwen_architecture, digest

    protect_output(output, {**cfg, 'rollouts': 0})
    checkpoint = resolve_input(cfg['checkpoint'], registry=registry_path(cfg))
    data_path = resolve_input(cfg['data'], registry=registry_path(cfg))
    verify_checkpoint(checkpoint)
    data = EpisodeData(data_path)
    identity = json.loads((checkpoint/'identity.json').read_text(encoding='utf-8'))
    if digest(data.root/'manifest.json') != identity['data_sha256']:
        raise ValueError('Checkpoint and prepared data identity differ')
    device = torch.device('cuda' if cfg['backend'] != 'portable' else 'cpu')
    if device.type == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('A CUDA BF16 environment is required for the selected backend')
    torch.manual_seed(cfg['seed'])
    model = QwenVLA.load(checkpoint/'model').to(dtype=torch.bfloat16,device=device).eval()
    student_config = model.student_config
    processor = AutoProcessor.from_pretrained(checkpoint/'processor',local_files_only=True,trust_remote_code=False,
        use_fast=False,size={'shortest_edge':model.student_config.image_size**2,'longest_edge':model.student_config.image_size**2})
    samples, query_ids = [], []
    def prepare(sample,state):
        inputs=prepare_observation(processor,student_config,instruction=sample['instruction'],
            primary=data.image(sample,'primary'),wrist=data.image(sample,'wrist'))
        inputs={k:v.to(device) for k,v in inputs.items()}
        proprio=torch.as_tensor(sample['proprio'],device=device)[None]
        return inputs,proprio,state
    state,episode=None,None
    with torch.no_grad(), torch.autocast(device.type,dtype=torch.bfloat16):
        for sample in observations(data,'train',cfg['seed'],cfg['calibration_queries']):
            if episode!=sample['episode_id']: state=None
            episode=sample['episode_id']
            args=prepare(sample,state)
            samples.append(Sample(args,torch.as_tensor(sample['gt_action'],device=device)[None],
                torch.as_tensor(sample['action_valid_mask'],device=device,dtype=torch.bool)[None]))
            state=model(*args).recurrent_state.detach()
            query_ids.append([episode,sample['timestep']])
    if len(samples)!=cfg['calibration_queries']:
        raise ValueError('Insufficient calibration queries')
    def evaluate(policy):
        state,episode=None,None
        results=[]
        with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16):
            for sample in observations(data,'validation',cfg['seed'],cfg['eval_queries']):
                if episode!=sample['episode_id']:state=None
                episode=sample['episode_id']
                prediction=policy(*prepare(sample,state))
                state=prediction.recurrent_state.detach()
                valid=torch.as_tensor(sample['action_valid_mask'],dtype=torch.bool)
                predicted=prediction.actions[0].detach().float().cpu()[valid]
                target=torch.as_tensor(sample['gt_action']).float()[valid]
                if not torch.isfinite(predicted).all():raise FloatingPointError('Non-finite packed evaluation')
                results.append((predicted,target,[episode,sample['timestep']]))
        if len(results)!=cfg['eval_queries']:raise ValueError('Insufficient validation queries')
        return results
    baseline=evaluate(model)
    layers=select_layers(model,cfg['methods'][method]['scope'])
    with torch.autocast(device.type,dtype=torch.bfloat16):
        modules,report=calibrate(model,layers,samples,method=method,target_bpw=cfg['methods'][method]['target_bpw'],
            group_size=cfg['group_size'],backend=cfg['backend'],seed=cfg['seed'],**cfg.get(method,{}))
    replace(model,modules)
    del layers,samples,modules
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    converted=evaluate(model)
    provenance={'source':source_identity(),'checkpoint':asset_record(cfg['checkpoint'][len('hfasset://'):],registry_path(cfg)),
                'data':asset_record(cfg['data'][len('hfasset://'):],registry_path(cfg)),
                'calibration_query_ids':query_ids,'seed':cfg['seed'],'torch':torch.__version__,
                'device':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'}
    output.mkdir(parents=True,exist_ok=False)
    checksum=save(model,output/'artifact',architecture=qwen_architecture(model),provenance=provenance,report=report,
                  processor_source=checkpoint/'processor',normalization=data.stats)
    del model
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    restored,_=_load_owned(output/'artifact',checksum,backend=cfg['backend'])
    restored=restored.to(device).eval()
    reloaded=evaluate(restored)
    for a,b in zip(converted,reloaded):
        if a[2]!=b[2]:raise ValueError('Reload validation query pairing differs')
        torch.testing.assert_close(a[0],b[0],rtol=.01,atol=.002)
    result={'status':'completed','method':method,'report':report,'artifact_manifest_sha256':checksum,
            'action_mae_vs_ground_truth':float(torch.cat([(a-t).abs() for a,t,_ in reloaded]).mean()),
            'action_mae_vs_bf16':float(torch.cat([(a[0]-b[0]).abs() for a,b in zip(reloaded,baseline)]).mean()),
            'validation_query_ids':[r[2] for r in reloaded],'roundtrip_checked':True,
            'rollouts':0,'selection_eligible':False}
    atomic_json(output/'report.json',result)
    print(json.dumps({'output':str(output),'method':method,'status':'completed'}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--method',choices=(*METHODS,'all'),default='all')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    cfg=configuration(args.config)
    if not args.run:
        print(json.dumps(plan(cfg),indent=2));return
    if args.output is None:parser.error('--run requires an explicit new --output path')
    if args.output.exists():parser.error('--output must be new; artifacts are never overwritten')
    if args.method!='all':
        run(cfg,args.method,args.output.resolve());return
    from vlade.evaluation.quantization_comparison import protect_output
    protect_output(args.output.resolve(),{**cfg, 'rollouts':0})
    args.output.mkdir(parents=True)
    failures=[]
    for method in METHODS:
        command=[sys.executable,str(ROOT/'scripts/quantization/run_packed_ptq.py'),'--config',str(args.config.resolve()),
                 '--method',method,'--run','--output',str((args.output/method).resolve())]
        with (args.output/(method+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:failures.append(method)
    if failures:raise SystemExit('Failed packed conditions: '+','.join(failures))


if __name__=='__main__':main()
