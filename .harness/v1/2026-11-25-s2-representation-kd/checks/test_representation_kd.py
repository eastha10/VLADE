"""CPU semantic alignment, native Qwen gradients, paired T1 wire and resume checks."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".harness/v1/2026-11-11-ava-teacher-code/checks")]
from qwen_vla_smoke import tiny_inputs, tiny_student
from vlade.distillation.representation import (ActionRepresentationKD, representation_settings,
    representation_contract, save_representation, load_representation)
from vlade.distillation.action import action_distillation_loss
from vlade.research import distillation
from vlade.training import qwen_vla_task as trainer
from vlade.training.teacher_inference import EpisodeTeacher, InferenceActions, TeacherRepresentation
from vlade.training.loop import (LoopState, make_scheduler, run_loop, training_state, restore_training_state)
from vlade.training.run_control import seal_checkpoint, verify_checkpoint
from vlade.models.students.qwen_vla import QwenVLA

CONFIG = ROOT / "configs/experiments/qwen_vla_representation_kd.json"


def interleaved_student():
    # The legacy 4-layer fixture has full attention only at its final layer.
    # Its -2 action slots stay zero on the first query. Real 24-layer Qwen has
    # earlier full-attention layers; keep that property in the reduced model.
    from transformers import Qwen3_5Model
    seed_model = tiny_student()
    config = copy.deepcopy(seed_model.backbone.config)
    config.text_config.layer_types = ['linear_attention','full_attention','linear_attention','full_attention']
    config._attn_implementation = 'eager'
    return QwenVLA(Qwen3_5Model(config), seed_model.student_config)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(42)


def test_s2_matches_s1_except_representation_and_routes_to_s2():
    s1 = json.loads((CONFIG.parent / "qwen_vla_action_kd.json").read_text())
    s2 = json.loads(CONFIG.read_text())
    rep = s2["distillation"].pop("representation")
    assert s1 == s2
    s2["distillation"]["representation"] = rep
    snapshot = copy.deepcopy(s2)
    _, variants = distillation.load_settings(s2, {})
    assert s2 == snapshot  # plan does not mutate caller's settings
    assert distillation._checkpoint_location("gt_kd", variants["gt_kd"], "run") == ("run", "S2")
    assert distillation._checkpoint_location("gt_kd", s1, "run") == ("run", "S1")


@pytest.mark.parametrize("override", [
    {"teacher_plan_max_new_tokens":48}, {"teacher_mode":"cache"},
    {"representation":{"weight":float('nan'), "loss":"normalized_mse", "alignment":"action_slots_8x7", "student_source":"temporal_hidden"}},
    {"representation":{"weight":1, "loss":"normalized_mse", "alignment":"text_token_ids", "student_source":"temporal_hidden"}},
])
def test_reject_incompatible_representation_settings(override):
    kd = json.loads(CONFIG.read_text())["distillation"]
    with pytest.raises(ValueError):
        representation_settings({**kd, **override})


def test_projection_dimension_mask_detach_and_alignment():
    rng = torch.get_rng_state().clone()
    projection = ActionRepresentationKD(3, 5)
    assert torch.equal(torch.get_rng_state(), rng)
    student = torch.randn(1,56,3,requires_grad=True)
    teacher = torch.randn(1,56,5,requires_grad=True)
    valid = torch.zeros(1,8,7,dtype=torch.bool)
    valid[0,0,:] = True
    expected = projection(student, teacher, valid)
    swapped = teacher.detach().flip(1)
    assert not torch.allclose(expected, projection(student, swapped, valid))
    dirty_s = student.detach().clone(); dirty_t = teacher.detach().clone()
    dirty_s[:,7:] = float('nan'); dirty_t[:,7:] = float('inf')
    dirty_s.requires_grad_()
    actual = projection(dirty_s, dirty_t, valid)
    torch.testing.assert_close(expected, actual)
    actual.backward()
    assert torch.isfinite(dirty_s.grad).all() and dirty_s.grad[:,7:].count_nonzero() == 0
    assert projection.projection.weight.grad.abs().sum() > 0
    assert teacher.grad is None
    with pytest.raises(ValueError):
        projection(student, teacher[:,:55], valid)
    with pytest.raises(ValueError):
        projection(student, teacher, torch.zeros_like(valid))


def test_actual_dimensions_from_teacher_config_and_wrong_backbone_rejection(tmp_path):
    path = tmp_path / "config.json"
    value = {"llm_backbone_id":"llama2-7b-pure", "text_config":{"hidden_size":4096,"num_hidden_layers":32}}
    path.write_text(json.dumps(value))
    contract = representation_contract(tmp_path, student_dim=1024, student_layer=-2, teacher_layer=-2)
    assert contract["teacher_dim"] == 4096 and contract["tokens"] == 56
    value["llm_backbone_id"] = "vicuna-v15-7b"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Llama2"):
        representation_contract(tmp_path, student_dim=1024, student_layer=-2, teacher_layer=-2)


def test_s2_preflight_verifies_s0_t1_dimensions_without_cuda(tmp_path, monkeypatch, capsys):
    _, variants = distillation.load_settings(CONFIG,{})
    settings = variants['gt_kd']
    settings['distillation']['teacher_python'] = sys.executable
    for name in ('manifest.json','checkpoint_manifest.json'):
        (tmp_path / name).write_text('{}')
    (tmp_path / 'config.json').write_text(json.dumps({'llm_backbone_id':'llama2-7b-pure',
        'text_config':{'hidden_size':4096,'num_hidden_layers':32}}))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(trainer,'source_identity',lambda *args,**kwargs:{'git_commit':'a'*40})
    monkeypatch.setattr(trainer,'resolve_input',lambda reference:tmp_path)
    monkeypatch.setattr(trainer,'EpisodeData',lambda path:SimpleNamespace(root=path,manifest={'counts':{}}))
    checked = []
    def s0(*args): checked.append('s0'); return 800
    def t1(*args): checked.append('t1'); return tmp_path, {'step':42,'temporal_feature_layer':-2}
    monkeypatch.setattr(trainer,'verified_s0',s0)
    monkeypatch.setattr(trainer,'teacher_identity',t1)
    output = tmp_path / 'not-created'
    trainer.run(settings,output,preflight=True)
    report = json.loads(capsys.readouterr().out)
    assert checked == ['s0','t1'] and not output.exists()
    assert report['representation']['student_dim'] == 1024
    assert report['representation']['teacher_dim'] == 4096
    assert report['teacher_plan_max_new_tokens'] == 0 and report['gpu_forward_tested'] is False


def test_episode_teacher_pair_replay_uses_one_frozen_llama_forward_per_query():
    from transformers import LlamaConfig, LlamaModel
    model = LlamaModel(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)).eval().requires_grad_(False)
    class Policy:
        def reset(self):
            self.context = 0
        @torch.inference_mode()
        def predict(self, sample, data):
            self.context += 1
            out = model(input_ids=torch.full((1,56), sample['timestep']//8+1), output_hidden_states=True)
            hidden = out.hidden_states[-2][0] + self.context
            return hidden[:,:7].reshape(8,7,7).mean(-1), hidden
    queries = [{"timestep":x,"episode_id":"a"} for x in (0,8,16)]
    data = SimpleNamespace(episodes=lambda split, order: iter([({}, queries)]))
    teacher = EpisodeTeacher(Policy(), data)
    action = teacher.get("a", 16); hidden = teacher.hidden.clone()
    assert teacher.predictions == 3 and not teacher.hidden.requires_grad
    torch.testing.assert_close(action, hidden[:,:7].reshape(8,7,7).mean(-1))
    teacher.get("a", 16)
    assert teacher.predictions == 6
    torch.testing.assert_close(hidden, teacher.hidden)
    assert all(p.grad is None for p in model.parameters())


def test_wire_pair_and_observation_layer_layout_rejection():
    client = InferenceActions.__new__(InferenceActions)
    sample = {"episode_id":"a","timestep":8}
    client.timings = None
    wire = {**sample,"signal":"action_and_representation", "action":torch.zeros(8,7).tolist(),
            "hidden":torch.randn(56,13).tolist(), "hidden_layer":-2,
            "hidden_layout":"horizon_major_action_dimension_minor"}
    calls = []
    def send(request):
        calls.append(request)
        return wire
    client._send = send
    result = client.get_representation(sample, dimension=13, layer=-2)
    assert len(calls) == 1 and result.hidden.shape == (56,13) and result.action.shape == (8,7)
    wire["timestep"] = 0
    with pytest.raises(ValueError, match="observation"):
        client.get_representation(sample, dimension=13, layer=-2)
    wire["timestep"] = 8; wire["hidden_layer"] = -1
    with pytest.raises(ValueError, match="layer/layout"):
        client.get_representation(sample, dimension=13, layer=-2)
    wire["hidden_layer"] = -2
    with pytest.raises(ValueError, match="hidden slots"):
        client.get_representation(sample, dimension=14, layer=-2)


def test_worker_main_serializes_paired_hidden_without_plan(tmp_path, monkeypatch, capsys):
    import io
    from vlade.training import teacher_inference as wire
    from vlade.training import fixed_cache, residency
    sample = {'episode_id':'train-a','timestep':0}
    data = SimpleNamespace(root=tmp_path, episodes=lambda split,order:iter([({},[sample])]))
    class Policy:
        def __init__(self,*args,**kwargs): pass
        def reset(self): pass
        def predict(self,sample,data):
            return torch.zeros(8,7), torch.arange(56*16).reshape(56,16).float()
        def plan(self,*args,**kwargs):
            raise AssertionError('Representation S2 must not call text Plan')
    monkeypatch.setattr(sys,'argv',['teacher','--checkpoint','hfasset://T1','--data','hfasset://data','--device','cpu'])
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps({**sample,'signal':'action_and_representation'})+'\n'))
    monkeypatch.setattr(wire,'teacher_identity',lambda *args:(tmp_path,{'temporal_feature_layer':-2}))
    import vlade.storage.assets as assets
    import vlade.data.episode_data as episode_data
    monkeypatch.setattr(assets,'resolve_input',lambda value:tmp_path)
    monkeypatch.setattr(episode_data,'EpisodeData',lambda path:data)
    monkeypatch.setattr(fixed_cache,'SequentialTeacher',Policy)
    monkeypatch.setattr(residency,'teacher_on_device',lambda *args,**kwargs:nullcontext())
    wire.main()
    ready,reply = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert ready == {'ready':{'temporal_feature_layer':-2}}
    assert reply['episode_id'] == sample['episode_id'] and reply['timestep'] == 0
    assert reply['hidden_layer'] == -2 and len(reply['hidden']) == 56
    assert reply['hidden'][1][0] == 16 and reply['action'] == torch.zeros(8,7).tolist()


def test_native_qwen_loss_callback_has_representation_gradients_and_plan_free_validation(monkeypatch):
    model, projection = interleaved_student(), ActionRepresentationKD(32,16)
    sample = {"episode_id":"a", "timestep":0,"proprio":np.zeros(8,np.float32),
              "gt_action":np.zeros((8,7),np.float32),"action_valid_mask":np.ones((8,7),bool)}
    signal = TeacherRepresentation(torch.randn(8,7), torch.randn(56,16))
    calls = []
    def get_representation(row, **kwargs):
        calls.append(kwargs)
        return signal
    teacher = SimpleNamespace(get_representation=get_representation)
    monkeypatch.setattr(trainer.torch,"autocast",lambda *args,**kwargs:nullcontext())
    terms = []
    loss, get_history, _ = trainer.action_loss(model,None,None,None,'cpu',teacher=teacher,
        representation=projection,representation_weight=.3,loss_terms=terms)
    inputs = tiny_inputs()
    total = loss((sample,inputs,None),0)
    # AVA contains training dropout: compare terms from the same forward.
    row = terms[0]
    assert float(total.detach()) == pytest.approx(row['task_l1'] + row['action_kd_l1'] + .3*row['representation_mse'])
    total.backward()
    assert get_history()[1].grad_fn is None
    assert terms[0]['representation_mse'] > 0 and len(calls) == 1
    assert projection.projection.weight.grad.abs().sum() > 0
    for module in (model.vision_backbone,model.language_model,model.ava,model.action_head):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    validation,_,_ = trainer.action_loss(model,None,None,None,'cpu',phase='validation')
    validation((sample,inputs,None),0)
    assert len(calls) == 1  # same GT-only validation as S1; no teacher or plan


def test_native_projection_optimizer_seal_resume_matches_uninterrupted(tmp_path):
    model, projection = interleaved_student(), ActionRepresentationKD(32,16)
    inputs, proprio = tiny_inputs(), torch.randn(1,8)
    target, hidden = torch.randn(1,8,7), torch.randn(1,56,16)
    valid = torch.ones(1,8,7,dtype=torch.bool)
    contract = {"student_dim":32,"teacher_dim":16,"teacher_layer":-2}
    def optimizer_for(m,p):
        return torch.optim.AdamW(list(m.parameters())+list(p.parameters()),lr=1e-4,foreach=False)
    optimizer = optimizer_for(model,projection); scheduler = make_scheduler(optimizer,2)
    def loss_for(m,p):
        def loss(batch,step):
            out = m(inputs,proprio,batch_idx=step)
            actions,_ = action_distillation_loss(out,target,target*.5,valid)
            return actions + p(out.recurrent_state,hidden,valid)
        return loss
    state = run_loop(modules=[model,projection],optimizer=optimizer,scheduler=scheduler,
        batches=[None]*4,loss_fn=loss_for(model,projection),max_steps=1,accumulation_steps=2)
    folder = tmp_path / "step-0000001"; folder.mkdir()
    model.save(folder / "model")
    save_representation(projection,folder / "representation.pt",contract)
    torch.save(training_state(state,optimizer,scheduler),folder / "trainer_state.pt")
    seal_checkpoint(folder,1,2); (folder / "COMPLETE").write_text("ok\n")
    seal = verify_checkpoint(folder,accumulation=2)
    assert 'representation.pt' in seal['files']
    payload = torch.load(folder / "trainer_state.pt",weights_only=True)
    run_loop(modules=[model,projection],optimizer=optimizer,scheduler=scheduler,state=state,
        batches=[None]*2,loss_fn=loss_for(model,projection),max_steps=2,accumulation_steps=2)
    restored = QwenVLA.load(folder / "model")
    rep = ActionRepresentationKD(32,16)
    load_representation(rep,folder / "representation.pt",contract)
    opt = optimizer_for(restored,rep); sched = make_scheduler(opt,2)
    recovered = restore_training_state(payload,opt,sched)
    run_loop(modules=[restored,rep],optimizer=opt,scheduler=sched,state=recovered,
        batches=[None]*2,loss_fn=loss_for(restored,rep),max_steps=2,accumulation_steps=2)
    for first,second in zip(list(model.parameters())+list(projection.parameters()),
                            list(restored.parameters())+list(rep.parameters())):
        torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert recovered.step == 2 and recovered.microsteps == 4
    assert not any('projection' in name for name in restored.state_dict())
    with pytest.raises(ValueError,match='contract'):
        load_representation(rep,folder / 'representation.pt',{**contract,'teacher_layer':-1})


def test_existing_session_cells_compile_and_keep_old_kernel_source():
    from vlade.utils.representation_kd_notebook import CONFIG_CELL, RUN_CELL
    compile(CONFIG_CELL,'config','exec'); compile(RUN_CELL,'run','exec')
    # The nested child script must compile independently of the notebook kernel.
    import ast
    tree = ast.parse(RUN_CELL)
    script = next(node.value.value for node in tree.body if isinstance(node,ast.Assign)
                  and any(isinstance(x,ast.Name) and x.id == 'CLIENT' for x in node.targets))
    compile(script,'client','exec')
    assert 'prepare(' not in script and "locate(code, work, 'ava')" in script
    assert "sys.path" not in RUN_CELL and 'pkill' not in RUN_CELL
    assert 'os.killpg(process.pid, signal.SIGINT)' in RUN_CELL


def test_native_trainer_pilot_and_resume_save_projection_with_optimizer(tmp_path, monkeypatch):
    """Run the actual adapter on CPU; replace GPU availability and external inputs only."""
    from vlade.training.memory import adamw_storage
    model = interleaved_student()
    initial = tmp_path / 'initial'; initial.mkdir()
    model.save(initial / 'model')
    (initial / 'checkpoint_manifest.json').write_text('{}')
    data_root = tmp_path / 'data'; data_root.mkdir()
    (data_root / 'manifest.json').write_text('{}')
    teacher_root = tmp_path / 'teacher'; teacher_root.mkdir()
    (teacher_root / 'config.json').write_text(json.dumps({'llm_backbone_id':'llama2-7b-pure',
        'text_config':{'hidden_size':16,'num_hidden_layers':2}}))
    data = SimpleNamespace(root=data_root,manifest={'counts':{},'normalization_sha256':'b'*64})
    config = model.student_config
    _, variants = distillation.load_settings(CONFIG,{})
    settings = variants['gt_kd']
    settings.update(max_steps=2,accumulation_steps=2,save_steps=1,validation_queries=1)
    settings['distillation']['teacher_python'] = sys.executable
    signature = {'step':99,'temporal_feature_layer':-2,'frozen':True}
    class Processor:
        def save_pretrained(self,path):
            path.mkdir(); (path / 'processor.json').write_text('{}')
    class CPUFacade:
        cuda = SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,
            manual_seed_all=lambda *args:None,get_device_name=lambda *args:'synthetic CPU',
            reset_peak_memory_stats=lambda *args:None,OutOfMemoryError=torch.cuda.OutOfMemoryError)
        def __getattr__(self,name): return getattr(torch,name)
        def device(self,name): return torch.device('cpu' if str(name).startswith('cuda') else name)
        def autocast(self,*args,**kwargs): return nullcontext()
    closed = []
    class Teacher:
        def __init__(self,*args,**kwargs): self.signature = signature
        def get_representation(self,row,*,dimension,layer):
            assert dimension == 16 and layer == -2
            return TeacherRepresentation(torch.ones(8,7)*.1,torch.arange(56*16).reshape(56,16).float()/100)
        def close(self): closed.append(True)
    sample = {'episode_id':'a','timestep':0,'proprio':np.ones(8,np.float32)*.2,
              'gt_action':np.ones((8,7),np.float32)*.3,'action_valid_mask':np.ones((8,7),bool)}
    inputs = tiny_inputs()
    def batches(*args,**kwargs):
        while True:
            yield sample,inputs,None
    monkeypatch.delenv('VLADE_HUB_REPO',raising=False)
    monkeypatch.delenv('VLADE_HUB_RUN_ID',raising=False)
    monkeypatch.setattr(trainer,'torch',CPUFacade())
    monkeypatch.setattr(trainer,'source_identity',lambda *args,**kwargs:{'git_commit':'a'*40})
    monkeypatch.setattr(trainer,'resolve_input',lambda value:initial if value == 'hfasset://S0' else data_root)
    monkeypatch.setattr(trainer,'EpisodeData',lambda path:data)
    monkeypatch.setattr(trainer.QwenVLAConfig,'from_json',lambda path:config)
    monkeypatch.setattr(trainer,'verified_s0',lambda *args:800)
    monkeypatch.setattr(trainer,'teacher_identity',lambda *args:(teacher_root,signature))
    monkeypatch.setattr(trainer,'load_saved_processor',lambda *args:Processor())
    monkeypatch.setattr(trainer,'check_adamw_capacity',lambda model,*args,**kwargs:adamw_storage(model))
    monkeypatch.setattr(trainer,'cuda_memory',lambda *args:{})
    monkeypatch.setattr(trainer,'InferenceActions',Teacher)
    monkeypatch.setattr(trainer,'batches',batches)
    output = tmp_path / 'S2'
    trainer.run(settings,output,stop_after=1)
    first = output / 'step-0000001'
    assert 'representation.pt' in verify_checkpoint(first,accumulation=2)['files']
    first_rep = torch.load(first / 'representation.pt',weights_only=True)['state_dict']['projection.weight']
    trainer.run(settings,output,resume=first,stop_after=2)
    last = output / 'step-0000002'
    assert verify_checkpoint(last,accumulation=2)['microsteps'] == 4
    saved = torch.load(last / 'representation.pt',weights_only=True)
    assert not torch.equal(first_rep,saved['state_dict']['projection.weight'])
    state = torch.load(last / 'trainer_state.pt',weights_only=True)
    assert any(row.get('exp_avg',torch.empty(0)).shape == (16,32) for row in state['optimizer']['state'].values())
    metrics = json.loads((last / 'metrics.json').read_text())
    assert metrics['representation_mse'] > 0 and np.isfinite(metrics['validation_l1'])
    assert closed == [True,True]
    assert not any('projection' in key for key in QwenVLA.load(last / 'model').state_dict())


def test_colab_child_inherits_actual_s1_pins_and_budget_without_install(tmp_path, monkeypatch, capsys):
    import ast
    from vlade.utils.representation_kd_notebook import RUN_CELL
    from vlade.utils import source_identity
    from vlade.runtime import environment
    tree = ast.parse(RUN_CELL)
    script = next(node.value.value for node in tree.body if isinstance(node,ast.Assign)
                  and any(isinstance(x,ast.Name) and x.id == 'CLIENT' for x in node.targets))
    s1 = json.loads((CONFIG.parent / 'qwen_vla_action_kd.json').read_text())
    workflow = s1.pop('workflow')
    workflow['checkpoint_pins'] = {'S0':{'repo_id':'VLADE-konkuk/VLADE-checkpoints',
        'revision':'1'*40,'prefix':'runs/real-s0/S0/checkpoints/step-0000800'}}
    s1.update(max_steps=600,accumulation_steps=4,learning_rate=1e-5)
    contract = tmp_path / 's1-contract.json'
    contract.write_text(json.dumps({'workflow':workflow,'variants':{'gt_kd':s1}}))
    work = tmp_path / 'unused-work'
    located, calls = [], []
    monkeypatch.setattr(source_identity,'source_identity',lambda *args,**kwargs:{'git_commit':'a'*40})
    monkeypatch.setattr(environment,'locate',lambda code,work,name:located.append(name))
    def main(args):
        path = Path(args[args.index('--config')+1])
        calls.append(json.loads(path.read_text()))
    monkeypatch.setattr(distillation,'main',main)
    def invoke(action,weight='1.0'):
        monkeypatch.setattr(sys,'argv',['child',str(ROOT),str(work),'a'*40,'synthetic-s2',
            action,str(contract),weight,'{}'])
        exec(script,{})
    invoke('plan')
    assert not work.exists() and not located and not calls
    text = capsys.readouterr().out
    planned = json.loads(text[text.index('{'):])
    assert planned['workflow'] == workflow
    actual = planned['variants']['gt_kd']
    actual['distillation'].pop('representation')
    assert actual == s1
    invoke('preflight')
    assert located == ['student','ava']
    effective = calls[0]
    effective['distillation'].pop('representation')
    assert effective == {**s1,'workflow':workflow}
    with pytest.raises(ValueError,match='S2 설정'):
        invoke('pilot','0.5')
