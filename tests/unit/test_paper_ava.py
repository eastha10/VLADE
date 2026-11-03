"""Small CPU proofs for the separate paper sequence/LoRA path."""
import copy
import json
import sys
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from vlade.training.paper_ava import (LoRALinear,add_lora,validate_recipe,sequence_count,
    sequence_stream,attention_penalty,backward_sequence,_save,_restore)
from vlade.training.loop import LoopState,make_scheduler


def test_lora_starts_identical_and_updates_adapters_without_base():
    base = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.GELU(),torch.nn.Linear(4,2))
    x = torch.randn(5,3)
    expected = base(x).detach().clone()
    original = copy.deepcopy(base.state_dict())
    assert add_lora(base,rank=2,alpha=2) == 2
    torch.testing.assert_close(base(x),expected,rtol=0,atol=0)
    optimizer = torch.optim.SGD([p for p in base.parameters() if p.requires_grad],lr=.1)
    base(x).square().mean().backward()
    assert base[0].base.weight.grad is None and base[0].lora_b.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(base(x),expected)
    torch.testing.assert_close(base[0].base.weight,original['0.weight'],rtol=0,atol=0)


def test_attention_penalty_uses_omega_mean_per_sample():
    omega = torch.tensor([[[.4],[.4]],[[.9],[.9]]],requires_grad=True)
    penalty = attention_penalty(omega,.6)
    torch.testing.assert_close(penalty,torch.tensor(.25))
    penalty.backward()
    assert omega.grad[0].lt(0).all() and omega.grad[1].gt(0).all()


def test_tbptt_keeps_two_frame_gradients_and_detaches_at_2_3():
    inputs = [torch.tensor(float(i+1),requires_grad=True) for i in range(4)]
    contexts = []
    def frame(value, context):
        contexts.append(context)
        hidden = value if context is None else value+context
        return hidden.square(),hidden,dict(mae=hidden.detach().item(),attention_penalty=0.)
    backward_sequence(inputs,frame,accumulation=2)
    assert contexts[0] is None
    assert contexts[1].requires_grad and not contexts[2].requires_grad and contexts[3].requires_grad
    # Two-frame pair graphs: h0=1,h1=3; detach h1; h2=6,h3=10. Loss / (4*2).
    expected = [1., .75, 4., 2.5]
    assert [x.grad.item() for x in inputs] == expected


class Data:
    def __init__(self):
        self.entries = [dict(episode_id='short',split='train',queries=3),
            dict(episode_id='a',split='train',queries=5),dict(episode_id='b',split='train',queries=6),
            dict(episode_id='test',split='test',queries=10)]
    def episodes(self,split,order):
        for identifier in order:
            entry = next(e for e in self.entries if e['episode_id']==identifier)
            assert entry['split']=='train'
            yield entry,[dict(episode_id=identifier,timestep=8*i) for i in range(entry['queries'])]


def test_sequences_are_causal_and_cursor_restores_across_epoch():
    data = Data()
    assert sequence_count(data) == 5
    keys = lambda stream: [[(q['episode_id'],q['timestep']) for q in w] for w in stream]
    full = keys(islice(sequence_stream(data,seed=42),13))
    assert full[7:12] == keys(islice(sequence_stream(data,seed=42,skip=7),5))
    assert len({tuple(w) for w in full[:5]})==5
    assert all(len({key[0] for key in window}) == 1 for window in full)
    assert all(window[-1][1]-window[0][1] == 24 for window in full)


@pytest.mark.parametrize('update',[dict(max_steps=40000),dict(effective_batch_size=64),dict(sequence_length=1),
    dict(initial_repository='LiAuto-DSR/avavla-libero-4in1'),dict(accumulation_steps=64)])
def test_best_recipe_rejects_silent_substitution(update):
    cfg = json.loads((ROOT/'configs/experiments/ava_paper_best.json').read_text())
    validate_recipe(cfg)
    with pytest.raises(ValueError):
        validate_recipe({**cfg,**update})


class Processor:
    def save_pretrained(self,path):
        (path/'processor.json').write_text('{}')


def test_lora_checkpoint_reloads_components_optimizer_and_cursor(tmp_path):
    model,head,proprio,generator = [torch.nn.Sequential(torch.nn.Linear(2,2)) for _ in range(4)]
    for module in (model,head,proprio):
        add_lora(module,2,2)
    components = (model,Processor(),head,proprio,generator)
    parameters = [p for m in (model,head,proprio,generator) for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=.01)
    scheduler = make_scheduler(optimizer,10,1,'cosine')
    sum(m(torch.ones(1,2)).square().mean() for m in (model,head,proprio,generator)).backward()
    optimizer.step(); scheduler.step()
    identity = dict(configuration='CPU test only')
    saved = _save(tmp_path,identity,components,optimizer,scheduler,LoopState(1,256))
    expected = [copy.deepcopy(m.state_dict()) for m in (model,head,proprio,generator)]
    with torch.no_grad():
        for p in parameters:
            p.add_(100)
    state = _restore(saved,identity,components,optimizer,scheduler)
    assert (state.step,state.microsteps)==(1,256)
    for module,weights in zip((model,head,proprio,generator),expected):
        for key,value in module.state_dict().items():
            torch.testing.assert_close(value,weights[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='mismatch'):
        _restore(saved,{'changed':True},components,optimizer,scheduler)
    with (saved/'state.pt').open('ab') as file:
        file.write(b'tamper')
    with pytest.raises(ValueError,match='integrity'):
        _restore(saved,identity,components,optimizer,scheduler)


def test_04_run_all_defaults_do_not_install_download_or_train():
    nb = json.loads((ROOT/'experiments/colab_04_ava_paper_best.ipynb').read_text(encoding='utf-8'))
    cells = {c['id']:''.join(c['source']) for c in nb['cells']}
    def forbidden(*args,**kwargs):
        pytest.fail('Default 04 must only prepare files')
    env = dict(RUN_INSTALL=False,RUN_DOWNLOAD=False,RUN_PILOT=False,RUN_PAPER_TRAINING=False,run=forbidden)
    for name in ('install-ava','paper-session-limit','download-pretrained','paper-pilot','paper-train'):
        exec(cells[name],env)
    for c in nb['cells']:
        if c['cell_type']=='code':
            compile(''.join(c['source']),c['id'],'exec')
