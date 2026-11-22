"""Real packed storage, algorithm invariants and safe deployment round trips."""
import copy
from pathlib import Path
import sys
import torch
from torch import nn
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from vlade.quantization.packed import Sample, PackedLinear, calibrate, replace
from vlade.quantization.packed.linear import quantize_rows, dequantize_rows
from vlade.quantization.packed.transforms import dual_transform, block_rotate
from vlade.quantization.packed import allocation, artifact


@pytest.mark.parametrize('bits', [0, 2, 3, 4, 8, 16])
def test_actual_integer_storage_and_decoder(bits):
    torch.manual_seed(11)
    layer = nn.Linear(33, 7).bfloat16()
    packed = PackedLinear.from_linear(layer, bits, group_size=8)
    packed.validate()
    assert not list(packed.parameters())
    assert packed.codes.dtype == torch.uint8
    assert packed.codes.numel() == 7 * ((33*bits+7)//8)
    codes, scale = quantize_rows(layer.weight, bits, 8)
    expected = dequantize_rows(codes, scale, bits, 8)
    assert torch.equal(packed.dequantized_rows(0, 7), expected)
    x = torch.randn(2, 3, 33).bfloat16()
    bias = layer.bias.float() if bits else torch.zeros(7)
    reference = nn.functional.linear(x.float(), expected, bias).bfloat16()
    torch.testing.assert_close(packed(x), reference)
    before = packed.scales.clone()
    packed.to(dtype=torch.bfloat16)
    assert torch.equal(packed.scales, before) and packed.scales.dtype == torch.float32
    assert packed(x[:, :0]).shape == (2, 0, 7)


def test_mixed_precision_bit_boundaries_and_pruning_bias():
    layer = nn.Linear(17, 6).bfloat16()
    packed = PackedLinear.from_linear(layer, [0,2,3,4,8,16], group_size=6)
    packed.validate()
    assert packed.bias[0] == 0
    assert torch.equal(packed.dequantized_rows(5, 6), layer.weight[5:6].float())
    packed.offsets[2] += 1
    with pytest.raises(ValueError): packed.validate()


def test_amf_scale_fit_never_worsens_weighted_objective():
    torch.manual_seed(5)
    w, fisher = torch.randn(9, 35), torch.rand(9,35)
    q0,s0 = quantize_rows(w,3,8)
    q1,s1 = quantize_rows(w,3,8,fisher)
    # Per-element AMF omega includes the local magnitude factor (Eq. 10).
    padded = nn.functional.pad(w,(0,5)).reshape(9,5,8)
    local = padded.square().mean(-1).repeat_interleave(8,1)[:,:35]
    omega = fisher * (local+w.square()).sqrt()
    a = ((w-dequantize_rows(q0,s0,3,8)).square()*omega).sum()
    b = ((w-dequantize_rows(q1,s1,3,8)).square()*omega).sum()
    assert b <= a + 1e-5


@pytest.mark.parametrize('width', [7, 16, 33])
def test_dual_transform_preserves_original_linear_map(width):
    torch.manual_seed(7)
    w,x = torch.randn(9,width),torch.randn(5,width)
    transformed, state = dual_transform(w, x.abs().amax(0), block_size=8)
    z = block_rotate(x/state['input_scale'],state['rotation1'])
    z = block_rotate(z[:,state['permutation']],state['rotation2'])
    torch.testing.assert_close(z@transformed.T,x@w.T,rtol=2e-4,atol=2e-4)


def test_budget_allocators_and_infeasible_pruning():
    sizes={'layers.0.a':32,'layers.0.b':32,'layers.1.a':32,'layers.1.b':32}
    assignment=allocation.actquant(dict.fromkeys(sizes,1.),sizes,3.)
    assert sum(assignment[n]*size for n,size in sizes.items()) <= 3*sum(sizes.values())
    assert all(b in (2,3,4,8,16) for b in assignment.values())
    costs=torch.tensor([[0.,.5,.1,.01,0.],[1.,.8,.2,.02,0.]])
    bits=allocation.qvla(costs,[8,8],0.,max_prune_fraction=1.,prune_threshold=2.)
    assert bits==[0,0]
    with pytest.raises(ValueError): allocation.qvla(costs,[8,8],0.,max_prune_fraction=0.)


def test_external_artifact_cannot_bypass_hf_input_gate(tmp_path,monkeypatch):
    monkeypatch.delenv('VLADE_LOCAL_INPUTS',raising=False)
    with pytest.raises(ValueError): artifact.load(tmp_path)


def test_packed_plan_never_resolves_inputs_or_allocates_gpu(monkeypatch):
    from vlade.quantization.packed import runner
    from vlade.storage import assets
    monkeypatch.setattr(assets,'resolve_input',lambda *a,**kw:pytest.fail('Planning cannot fetch inputs'))
    cfg=runner.configuration(ROOT/'configs/quantization/packed_student.json')
    result=runner.plan(cfg)
    assert result['downloads'] is False and result['gpu_execution'] is False
    assert set(result['methods'])=={'qvla','actquant','quantvla'}
    assert result['rollouts']==0


def policy():
    torch.manual_seed(31)
    return nn.Sequential(nn.Linear(8,7),nn.Tanh(),nn.Linear(7,3)).bfloat16()


def test_failed_calibration_restores_model_and_removes_hooks():
    model=policy().train()
    flags=[p.requires_grad for p in model.parameters()]
    invalid=Sample((torch.randn(1,8).bfloat16(),),torch.zeros(1,3),torch.ones(1,3,dtype=torch.long))
    with pytest.raises(ValueError):
        calibrate(model,{'0':model[0],'2':model[2]},[invalid]*3,method='actquant',target_bpw=3.)
    assert model.training and flags==[p.requires_grad for p in model.parameters()]
    assert all(not m._forward_hooks for m in model.modules())


@pytest.mark.parametrize('method,budget', [('actquant',3.),('qvla',4.),('quantvla',8.)])
def test_three_complete_conversion_paths_and_self_contained_roundtrip(tmp_path,method,budget):
    model=policy()
    samples=[Sample((torch.randn(1,8).bfloat16(),),torch.randn(1,3),torch.ones(1,3,dtype=torch.bool)) for _ in range(4)]
    before=copy.deepcopy(model.state_dict())
    flags=[p.requires_grad for p in model.parameters()]
    modules,report=calibrate(model,{'0':model[0],'2':model[2]},samples,method=method,target_bpw=budget,
                             group_size=4,refine_channels=2,rotation_block_size=4)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert flags==[p.requires_grad for p in model.parameters()]
    assert report['weight_bpw'] <= budget
    replace(model,modules)
    with torch.no_grad(): original=model(*samples[0].args)
    assert torch.isfinite(original).all()
    output=tmp_path/method
    identity=artifact.save(model,output,architecture={'kind':'custom_policy'},provenance={'source':'synthetic'},report=report)
    restored,manifest=artifact._load_owned(output,identity,builder=lambda spec:policy())
    with torch.no_grad(): torch.testing.assert_close(restored(*samples[0].args),original,atol=0,rtol=0)
    assert set(p.name for p in output.iterdir())=={'COMPLETE','packed.json','model.safetensors'}
    assert manifest['report']['method']==method
    with (output/'model.safetensors').open('r+b') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError): artifact._load_owned(output,identity,builder=lambda spec:policy())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires real CUDA/NVRTC')
@pytest.mark.parametrize('shape',[(1,17),(3,33),(2,64)])
def test_fused_cuda_reads_packed_codes_on_current_stream(shape):
    device=torch.device('cuda')
    layer=nn.Linear(shape[1],6).to(device).bfloat16()
    packed=PackedLinear.from_linear(layer,[0,2,3,4,8,16],group_size=8)
    x=torch.randn(*shape,device=device).bfloat16()
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream): actual=packed(x)
    torch.cuda.current_stream().wait_stream(stream)
    packed.backend='portable'
    expected=packed(x)
    torch.testing.assert_close(actual,expected,rtol=.01,atol=.002)
