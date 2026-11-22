"""Actual native Qwen vision+language+AVA/action forward and deployment reload."""
from pathlib import Path
import sys
import torch
import pytest

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))


def tiny_student():
    from transformers import Qwen3_5Config,Qwen3_5Model
    from vlade.models.students.qwen_vla import QwenVLA
    from vlade.models.students.configuration_qwen_vla import QwenVLAConfig
    config=Qwen3_5Config(
        text_config={'vocab_size':128,'hidden_size':128,'intermediate_size':256,'num_hidden_layers':2,
          'num_attention_heads':4,'num_key_value_heads':2,'head_dim':32,
          'layer_types':['full_attention','full_attention'],
          'rope_parameters':{'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,
                             'mrope_section':[6,5,5],'mrope_interleaved':True}},
        vision_config={'depth':1,'hidden_size':32,'intermediate_size':64,'num_heads':4,
          'patch_size':16,'temporal_patch_size':2,'spatial_merge_size':2,'out_hidden_size':128,
          'num_position_embeddings':16,'deepstack_visual_indexes':[]},image_token_id=127,vision_start_token_id=125,vision_end_token_id=126)
    config._attn_implementation='eager'
    student=QwenVLAConfig(hidden_size=128,image_size=32,patch_grid=1,ava_embed_dim=128,
                          action_head_hidden_dim=128,max_sequence_length=128)
    return QwenVLA(Qwen3_5Model(config),student).bfloat16().eval()


def tiny_inputs():
    ids=torch.tensor([[1,125,127,126,2,125,127,126,3]],device='cuda')
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
            'pixel_values':torch.randn(8,1536,device='cuda',dtype=torch.bfloat16),
            'image_grid_thw':torch.tensor([[1,2,2],[1,2,2]],device='cuda')}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual native GPU integration')
@pytest.mark.parametrize('method,budget',[('actquant',3.),('qvla',4.),('quantvla',8.)])
def test_native_student_export_reload_forward(tmp_path,method,budget):
    from vlade.evaluation.s0_quantization import select_layers
    from vlade.quantization.packed import Sample,calibrate,replace,PackedLinear
    from vlade.quantization.packed import artifact
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model=tiny_student().cuda()
    inputs=tiny_inputs()
    args=(inputs,torch.zeros(1,8,device='cuda'),None)
    samples=[Sample(args,torch.randn(1,8,7,device='cuda'),torch.ones(1,8,7,device='cuda',dtype=torch.bool)) for _ in range(3)]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        layers=select_layers(model,'language' if method=='quantvla' else 'language_vision')
        modules,report=calibrate(model,layers,samples,method=method,target_bpw=budget,
                                  backend='cuda',refine_channels=1,group_size=16,rotation_block_size=16)
    replace(model,modules)
    del layers,modules
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        before=model(*args)
    assert before.actions.shape==(1,8,7) and torch.isfinite(before.actions).all()
    checksum=artifact.save(model,tmp_path/'artifact',architecture=artifact.qwen_architecture(model),
                           provenance={'input':'synthetic-native-Qwen'},report=report)
    loaded,metadata=artifact._load_owned(tmp_path/'artifact',checksum,backend='cuda')
    loaded=loaded.cuda()
    assert any(isinstance(m,PackedLinear) for m in loaded.modules())
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        after=loaded(*args)
    torch.testing.assert_close(before.actions,after.actions,rtol=0,atol=0)
    assert metadata['report']['method']==method


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real native GPU pipeline')
@pytest.mark.parametrize('method',['actquant','qvla','quantvla'])
def test_research_runner_exports_processor_and_evaluates_reloaded_model(tmp_path,monkeypatch,method):
    """Only external input/processor/data boundaries are synthetic; model and CUDA are real."""
    import json,hashlib
    import numpy as np
    from transformers import AutoProcessor
    from vlade.storage import assets
    from vlade.training import run_control
    from vlade.data import episode_data
    from vlade.models.students import qwen_vla
    from vlade.models.students.backbones import qwen
    from vlade.evaluation import s0_quantization
    from vlade.quantization.packed import runner
    torch.set_num_threads(2)
    checkpoint,data_path=tmp_path/'checkpoint',tmp_path/'prepared'
    (checkpoint/'processor').mkdir(parents=True)
    data_path.mkdir()
    (data_path/'manifest.json').write_text('{}')
    (checkpoint/'identity.json').write_text(json.dumps({'data_sha256':hashlib.sha256(b'{}').hexdigest()}))
    for name in ('chat_template.jinja','processor_config.json','tokenizer.json','tokenizer_config.json'):
        (checkpoint/'processor'/name).write_text('{}')
    class Data:
        root=data_path
        stats={'train_only':True}
        def image(self,*args):return None
    monkeypatch.setattr(assets,'resolve_input',lambda value,**kw:checkpoint if str(value)=='hfasset://S0' else data_path)
    monkeypatch.setattr(run_control,'verify_checkpoint',lambda path:{'step':100})
    monkeypatch.setattr(episode_data,'EpisodeData',lambda root:Data())
    monkeypatch.setattr(qwen_vla.QwenVLA,'load',lambda path:tiny_student())
    monkeypatch.setattr(AutoProcessor,'from_pretrained',lambda *args,**kw:None)
    inputs=tiny_inputs()
    monkeypatch.setattr(qwen,'prepare_observation',lambda *args,**kw:inputs)
    def observations(data,split,seed,limit):
        for i in range(limit):
            yield {'episode_id':'synthetic-episode','timestep':i*8,'instruction':'test',
                   'proprio':np.zeros(8,dtype=np.float32),'gt_action':np.zeros((8,7),dtype=np.float32),
                   'action_valid_mask':np.ones((8,7),dtype=bool)}
    monkeypatch.setattr(s0_quantization,'observations',observations)
    cfg=runner.configuration(ROOT/'configs/quantization/packed_student.json')
    cfg.update(calibration_queries=3,eval_queries=2)
    cfg['qvla']['refine_channels']=1
    runner.run(cfg,method,tmp_path/'result')
    report=json.loads((tmp_path/'result/report.json').read_text())
    assert report['status']=='completed' and report['roundtrip_checked']
    assert len(report['validation_query_ids'])==2 and report['rollouts']==0
    assert (tmp_path/'result/artifact/processor/tokenizer.json').is_file()
