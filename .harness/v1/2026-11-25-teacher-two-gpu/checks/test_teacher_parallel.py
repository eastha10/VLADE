"""CPU verification of Teacher-only placement, offloaded FFT updates and contracts."""
from contextlib import nullcontext
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path[:0] = [str(ROOT / "src")]
from vlade.training import teacher_parallel as placement_module
from vlade.training.teacher_parallel import TeacherPlacement, generation_outputs, validate_parallel
from vlade.training.gradient_offload import OffloadedGradients, GradientOverflow, restore_paged_optimizer
from vlade.training.loop import (LoopState, run_loop, make_scheduler, training_state,
                                 restore_training_state, capture_rng, restore_rng)
from vlade.training.temporal import sequence_objective
from vlade.training.fft import FFTConfig, save_checkpoint, validate_resume
from vlade.training.run_control import verify_checkpoint
from vlade.research.configuration import load


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.set_num_threads(2)
    torch.manual_seed(42)


def options():
    return json.loads((ROOT / "configs/research/ava_fft_tp2.json").read_text())["teacher_parallel"]


@pytest.mark.parametrize("name", ["ava_fft", "oft_fft", "cache_teacher"])
def test_profiles_preserve_original_research_controls(name):
    original = load(ROOT / f"configs/research/{name}.json")
    value = load(ROOT / f"configs/research/{name}_tp2.json")
    for key in original:
        if key not in {"run_group", "asset_key"}:
            assert value[key] == original[key], key
    assert value["teacher_parallel"]["dtype"] == "float16"
    if "train" in value:
        assert value["train"]["batch_size"] * value["train"]["accumulation_steps"] == 8


@pytest.mark.parametrize("field,value", [("devices", ["cuda:0", "cuda:0"]),
    ("devices", ["cpu", "cuda:1"]), ("dtype", "float32"), ("split_layer", 0),
    ("reserve_gib", -1), ("loss_scale", float("nan")), ("max_overflow_retries", -1)])
def test_invalid_parallel_controls_rejected(field, value):
    config = options(); config[field] = value
    with pytest.raises(ValueError): validate_parallel(config)


@pytest.mark.parametrize("name", ["ava_fft", "oft_fft", "cache_teacher"])
def test_plan_does_not_start_gpu_or_create_workspace(name, tmp_path):
    work = tmp_path / "not-created"
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/research/run.py"),
        "--config", str(ROOT / f"configs/research/{name}_tp2.json"), "--workspace", str(work)],
        cwd=ROOT, text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["execution"] == "not_started"
    assert plan["config"]["teacher_parallel"]["devices"] == ["cuda:0", "cuda:1"]
    assert not work.exists()


def test_streamed_adamw_matches_original_full_accumulation_and_global_clip():
    reference = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    student = copy.deepcopy(reference)
    a = torch.optim.AdamW(reference.parameters(), lr=.01, weight_decay=.01, foreach=False)
    b = torch.optim.AdamW(student.parameters(), lr=.01, weight_decay=.01, foreach=False)
    batches = [torch.randn(1, 3) for _ in range(8)]
    manager = OffloadedGradients(student.parameters(), loss_scale=128)
    state = run_loop(modules=[student], optimizer=b, scheduler=make_scheduler(b, 3), batches=iter(batches * 3),
        loss_fn=lambda x, step: student(x).square().mean(), max_steps=3, accumulation_steps=8,
        max_grad_norm=.3, gradient_manager=manager)
    for _ in range(3):
        a.zero_grad(set_to_none=True)
        for x in batches: (reference(x).square().mean() / 8).backward()
        nn.utils.clip_grad_norm_(reference.parameters(), .3)
        a.step()
    assert (state.step, state.microsteps) == (3, 24)
    for p, q in zip(reference.parameters(), student.parameters()):
        torch.testing.assert_close(p, q)
        torch.testing.assert_close(a.state[p]["exp_avg"], b.state[q]["exp_avg"])
        assert b.state[q]["step"] == 3
    assert not manager.buffers
    manager.close()


def test_fp16_overflow_retries_same_window_rng_and_does_not_advance_schedule():
    parameter = nn.Parameter(torch.tensor(1., dtype=torch.float16))
    module = nn.ParameterList([parameter])
    optimizer = torch.optim.AdamW(module.parameters(), lr=.001, foreach=False)
    manager = OffloadedGradients(module.parameters(), loss_scale=128, max_retries=8)
    consumed, calls, updates = [], [], []
    def batches():
        for i in range(8): consumed.append(i); yield i
    def loss(i, step):
        calls.append((i, float(torch.rand(()))))
        return parameter * 10000
    scheduler = make_scheduler(optimizer, 1)
    state = run_loop(modules=[module], optimizer=optimizer, scheduler=scheduler, batches=batches(),
        loss_fn=loss, max_steps=1, accumulation_steps=8, gradient_manager=manager,
        on_step=lambda state, row: updates.append(state.step))
    assert consumed == list(range(8)) and len(calls) > 8
    for start in range(8, len(calls), 8): assert calls[start:start+8] == calls[:8]
    assert manager.loss_scale <= 4 and updates == [1]
    assert optimizer.state[parameter]["step"] == 1 and scheduler.last_epoch == 1
    assert (state.step, state.microsteps) == (1, 8)
    manager.close()


def test_overflow_exhaustion_updates_no_parameters_or_cursors():
    module = nn.Linear(1, 1).half()
    before = copy.deepcopy(module.state_dict())
    optimizer = torch.optim.AdamW(module.parameters(), foreach=False)
    manager = OffloadedGradients(module.parameters(), loss_scale=2**20, max_retries=0)
    state = LoopState()
    with pytest.raises(GradientOverflow):
        run_loop(modules=[module], optimizer=optimizer, scheduler=make_scheduler(optimizer, 1),
                 batches=iter([torch.ones(1, 1).half()] * 8), loss_fn=lambda x, step: module(x).sum(),
                 max_steps=1, accumulation_steps=8, state=state, gradient_manager=manager)
    assert not optimizer.state and (state.step, state.microsteps) == (0, 0)
    assert not manager.buffers
    for key, value in before.items(): torch.testing.assert_close(value, module.state_dict()[key])
    manager.close()


def test_tbptt_k4_and_offload_keep_gradients_across_two_frame_segments():
    a = nn.Linear(1, 1); b = copy.deepcopy(a)
    inputs = [torch.randn(1, 1) for _ in range(4)]
    def forward(model):
        def frame(x, context):
            state = model(x) + (0 if context is None else context)
            return state.square().mean(), state
        return frame
    sequence_objective(inputs, forward(a), detach_every=2, backward_scale=1)
    expected = [p.grad.clone() for p in a.parameters()]
    manager = OffloadedGradients(b.parameters(), loss_scale=128)
    sequence_objective(inputs, forward(b), detach_every=2, backward_scale=128)
    manager.prepare(1e6)
    for parameter, grad in zip(b.parameters(), expected):
        torch.testing.assert_close(manager.buffers[parameter], grad)
    assert all(p.grad is None for p in b.parameters())
    manager.clear(); manager.close()


def test_loss_scale_and_optimizer_round_trip_and_incompatible_resume_rejected(tmp_path):
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
    sched = make_scheduler(opt, 2)
    manager = OffloadedGradients(model.parameters(), loss_scale=128)
    inputs = [torch.randn(1, 2) for _ in range(8)]
    state = run_loop(modules=[model], optimizer=opt, scheduler=sched, batches=iter(inputs),
        loss_fn=lambda x, s: model(x).square().mean(), max_steps=1, accumulation_steps=8, gradient_manager=manager)
    payload = training_state(state, opt, sched, gradient_manager=manager)
    torch.save(payload, tmp_path / "state.pt")
    restored = copy.deepcopy(model)
    other = torch.optim.AdamW(restored.parameters(), lr=.01, foreach=False)
    schedule = make_scheduler(other, 2)
    second = OffloadedGradients(restored.parameters(), loss_scale=1)
    value = torch.load(tmp_path / "state.pt", weights_only=True)
    resumed = restore_training_state(value, other, schedule, gradient_manager=second)
    assert second.loss_scale == 128 and second.successful_steps == 1
    with pytest.raises(ValueError, match="resume contract"):
        restore_training_state(value, other, schedule)
    with pytest.raises(ValueError, match="resume contract"):
        restore_training_state({k: v for k, v in value.items() if k != "gradient_offload"}, other, schedule,
                               gradient_manager=second)
    rng = capture_rng()
    for module, optimizer, scheduler, controller, cursor in [(model,opt,sched,manager,state),
                                                            (restored,other,schedule,second,resumed)]:
        restore_rng(rng)
        run_loop(modules=[module], optimizer=optimizer, scheduler=scheduler, batches=iter(inputs),
            loss_fn=lambda x, s: module(x).square().mean(), max_steps=2, accumulation_steps=8,
            state=cursor, gradient_manager=controller)
    for p, q in zip(model.parameters(), restored.parameters()): torch.testing.assert_close(p, q)
    manager.close(); second.close()


def tiny_components():
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
                intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                pad_token_id=0, eos_token_id=2, max_position_embeddings=256))
            self.vision_backbone = nn.Linear(3, 32)
            self.projector = nn.Linear(32, 32)
        def save_pretrained(self, path, **kw):
            torch.save(self.state_dict(), path / "model.pt")
    return Teacher(), nn.Linear(32, 7), nn.Linear(8, 32), nn.Linear(32, 32)


def test_native_decoder_hooks_preserve_output_backward_and_state_dict(monkeypatch):
    model, head, proprio, ava = tiny_components()
    before = copy.deepcopy(model).eval()
    route = placement_module._route
    monkeypatch.setattr(placement_module, "_route", lambda device: route("cpu"))
    placement = TeacherPlacement(model, head, proprio, ava, options())
    assert 1 <= placement.split < 4
    assert str(placement.device_for("language_model.model.layers.0.self_attn.q_proj.weight")) == "cuda:0"
    assert str(placement.device_for("language_model.model.norm.weight")) == "cuda:1"
    assert sum(placement.weight_bytes) == sum(p.numel()*p.element_size() for m in (model,head,proprio,ava) for p in m.parameters())
    ids = torch.tensor([[1, 4, 5, 2]])
    model.eval()
    a = before.language_model(ids, output_hidden_states=True)
    b = model.language_model(ids, output_hidden_states=True)
    torch.testing.assert_close(a.logits, b.logits)
    a.logits.square().mean().backward(); b.logits.square().mean().backward()
    for p, q in zip(before.parameters(), model.parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad, q.grad)
    assert list(before.state_dict()) == list(model.state_dict())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two real CUDA GPUs unavailable")
def test_real_two_gpu_native_decoder_and_offloaded_backward():
    model, head, proprio, ava = tiny_components()
    for module in (model,head,proprio,ava): module.half()
    placement = TeacherPlacement(model,head,proprio,ava,options()); placement.activate()
    assert {p.device.index for p in model.parameters()} == {0,1}
    manager = OffloadedGradients(model.parameters(), loss_scale=128)
    output = model.language_model(torch.tensor([[1,4,5,2]],device="cuda:0"))
    (output.logits.float().square().mean()*128).backward()
    manager.prepare(1.)
    assert manager.buffers and all(g.device.type == "cpu" for g in manager.buffers.values())
    manager.close()


def test_saved_teacher_seal_includes_scaler_and_rejects_changed_placement(tmp_path, monkeypatch):
    model, head, proprio, ava = tiny_components()
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    scheduler = make_scheduler(optimizer, 10)
    manager = OffloadedGradients(model.parameters(), loss_scale=128)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "synthetic CPU check")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(total_memory=0))
    class Processor:
        def save_pretrained(self, path): (path / "processor.json").write_text("{}")
    cfg = FFTConfig("ava_vla", tmp_path, tmp_path, tmp_path / "output", save_optimizer_state=True,
                    teacher_parallel=options())
    output = save_checkpoint(cfg,1,model,Processor(),head,proprio,ava,optimizer,
                            SimpleNamespace(dataset_statistics={}),scheduler=scheduler,
                            loop_state=LoopState(1,8),gradient_manager=manager)
    assert verify_checkpoint(output,accumulation=8)["step"] == 1
    state = torch.load(output / "trainer_state.pt", weights_only=True)
    assert state["gradient_offload"]["loss_scale"] == 128
    validate_resume(replace(cfg,resume=output))
    changed = {**options(),"dtype":"bfloat16"}
    with pytest.raises(ValueError,match="placement/precision"):
        validate_resume(replace(cfg,resume=output,teacher_parallel=changed))
    manager.close()


def test_notebook_stays_teacher_only_and_defaults_to_plan():
    nb=json.loads((ROOT/'experiments/teacher_two_gpu.ipynb').read_text(encoding='utf-8'))
    cells={c['id']:''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'}
    assert list(cells)==['platform','source','config','run']
    for source in cells.values(): compile(source,'notebook','exec')
    scope={'CODE':ROOT,'BASE':Path('/synthetic')}
    exec(cells['config'],scope)
    assert scope['ACTION']=='plan' and scope['PROFILE']=='ava_fft_tp2'
    assert 'qwen' not in cells['run'] and 'student' not in cells['config']


def test_paged_resume_reallocates_moments_without_dense_loader_or_dtype_change():
    parameter = nn.Parameter(torch.ones(4, dtype=torch.float16))
    allocations = []
    class Optimizer(torch.optim.AdamW):
        is_paged = True
        def get_state_buffer(self, p, *, dtype):
            allocations.append((p, dtype))
            return torch.empty_like(p, dtype=dtype)
        def load_state_dict(self, saved): pytest.fail("Dense BNB loader must not be used")
    optimizer = Optimizer([parameter], lr=.1)
    saved = {"state": {0: {"step": 42, "state1": torch.tensor([1,2,3,4],dtype=torch.uint8),
                             "state2": torch.tensor([4,3,2,1],dtype=torch.uint8), "qmap1":torch.ones(256)}},
             "param_groups": [{**optimizer.param_groups[0], "params":[0], "lr":.01}]}
    restore_paged_optimizer(optimizer,saved)
    assert len(allocations)==2 and optimizer.state[parameter]["step"]==42
    assert optimizer.state[parameter]["state1"].dtype==torch.uint8
    assert optimizer.state[parameter]["qmap1"].dtype==torch.float32
    torch.testing.assert_close(optimizer.state[parameter]["state2"],saved["state"][0]["state2"])
    assert optimizer.param_groups[0]["params"][0] is parameter and optimizer.param_groups[0]["lr"]==.01


def test_native_checkpoint_recomputation_keeps_transfer_hooks_and_gradients(monkeypatch):
    model,head,proprio,ava=tiny_components()
    route=placement_module._route
    monkeypatch.setattr(placement_module,'_route',lambda device:route('cpu'))
    TeacherPlacement(model,head,proprio,ava,options())
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    calls=[]
    model.language_model.model.layers[-1].register_forward_pre_hook(lambda *a:calls.append(1))
    model.language_model(torch.tensor([[1,4,5,2]]),use_cache=False).logits.square().mean().backward()
    assert len(calls)>=2
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in layer.parameters())
               for layer in model.language_model.model.layers)


def test_teacher_refuses_missing_gpus_before_resolving_external_inputs(monkeypatch,tmp_path):
    import vlade.training.fft as fft
    monkeypatch.setattr(torch.cuda,'device_count',lambda:0)
    monkeypatch.setattr(fft,'resolve_config_assets',lambda *a,**kw:pytest.fail('external input resolved'))
    with pytest.raises(RuntimeError,match='before resolving'):
        fft.train(FFTConfig('ava_vla',tmp_path,tmp_path,tmp_path,teacher_parallel=options()))


def test_actual_fft_loss_k4_backward_with_native_llama_and_original_policy_components(monkeypatch):
    """Real Teacher mixin/L1/TBPTT/optimizer; synthetic vision and CPU-routed devices."""
    from vlade.models.teachers.components.multimodal import MultimodalForwardMixin
    from vlade.models.teachers.components.action_heads import L1RegressionActionHead
    from vlade.models.teachers.components.projectors import PrismaticProjector,ProprioProjector
    from vlade.models.teachers.components.attn_weight_generator import AttentionWeightGenerator
    from vlade.training.temporal import fft_objective
    from vlade.training.fft import compute_loss
    class Teacher(MultimodalForwardMixin,nn.Module):
        def __init__(self):
            super().__init__()
            self.config=SimpleNamespace(output_attentions=False,output_hidden_states=True,use_return_dict=True)
            self.language_model=LlamaForCausalLM(LlamaConfig(vocab_size=32064,hidden_size=32,
                intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
                pad_token_id=0,eos_token_id=2,max_position_embeddings=256))
            self.vision_backbone=nn.Linear(16,16)
            self.projector=PrismaticProjector(True,16,32)
            self.extra_layer_ids=[]  # native CPU Llama, AVA fork attention requires hardware runtime
        def get_input_embeddings(self):return self.language_model.get_input_embeddings()
    model=Teacher()
    head=L1RegressionActionHead(32,64,7)
    proprio=ProprioProjector(32,8)
    ava=AttentionWeightGenerator(128,image_shape=(2,2,32),action_shape=(8,7,32),text_dim=32,
        num_images=2,max_steps=100,sink_ids=None,sink_weight=1.)
    for component in (model,head,proprio,ava): component.bfloat16()
    route=placement_module._route
    monkeypatch.setattr(placement_module,'_route',lambda device:route('cpu'))
    selected={**options(),'dtype':'bfloat16'}
    placement=TeacherPlacement(model,head,proprio,ava,selected)
    model._vlade_teacher_placement=placement
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    frames=[]
    for index in range(4):
        frames.append(dict(input_ids=torch.tensor([[1,10,29871]+[1]*56+[2]]),
            attention_mask=torch.ones(1,60,dtype=torch.long),pixel_values=torch.randn(1,8,16),
            labels=torch.tensor([[-100]*3+[31744]*56+[2]]),proprio=torch.randn(1,8),actions=torch.zeros(1,8,7),
            action_valid_mask=torch.ones(1,8,7,dtype=torch.bool),
            sample_metadata=[{'episode_id':'e','timestep':8*index}]))
    cfg=SimpleNamespace(sequence_length=4,detach_every=2,temporal_feature_layer=-2,plan_loss_weight=0.)
    components=(model,head,proprio,ava)
    parameters=[p for m in components for p in m.parameters()]
    optimizer=torch.optim.AdamW(parameters,lr=2e-5,foreach=False)
    manager=OffloadedGradients(parameters,loss_scale=128)
    checked=[]
    def diagnostic(state):
        checked.append(manager.diagnostics(dict(vision=model.vision_backbone,language=model.language_model,
            projector=model.projector,action=head,proprio=proprio,attention_generator=ava)))
    state=run_loop(modules=list(components),optimizer=optimizer,scheduler=make_scheduler(optimizer,1),
        batches=iter([{'sequences':[frames]}]*8),loss_fn=lambda *a:pytest.fail('wrong backward path'),
        backward_fn=lambda batch,step,scale:fft_objective(cfg,components,batch,torch.device('cpu'),9,step,
            backward_scale=scale),max_steps=1,accumulation_steps=8,gradient_manager=manager,on_before_step=diagnostic)
    assert (state.step,state.microsteps)==(1,8) and checked
    assert all(value['finite_nonzero'] for value in checked[0].values())
    manager.close()


def test_fp16_teacher_sink_correction_preserves_dtype_and_finite_backward():
    from vlade.models.teachers.components.attn_weight_generator import AttentionWeightGenerator
    ava=AttentionWeightGenerator(128,image_shape=(2,2,32),action_shape=(8,7,32),text_dim=32,
        num_images=2,max_steps=100,sink_ids=[1,5],sink_weight=1.).half()
    slots,(weights,_)=ava(torch.randn(1,8,32).half(),None,torch.randn(1,3,32).half(),batch_idx=0)
    assert weights.dtype==torch.float16 and bool((weights[:,[1,5]]>=1).all())
    weights.square().sum().backward()
    assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in ava.parameters())
