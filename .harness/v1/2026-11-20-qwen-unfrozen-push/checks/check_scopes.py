"""Exercise full/policy/top-layer selection and legacy defaults on real modules."""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(sys.argv[1]).resolve()/'src'))
import torch
from torch import nn
from vlade.training.scopes import configure_trainable_scope

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Module()
        self.backbone.visual=nn.Linear(4,4)
        self.backbone.language=nn.Module()
        self.backbone.language.layers=nn.ModuleList([nn.Linear(4,4) for _ in range(3)])
        self.backbone.language.norm=nn.LayerNorm(4)
        self.action_head=nn.Linear(4,7)
        self.ava=nn.Linear(4,4)
        self.proprio_projector=nn.Linear(8,4)
    @property
    def language_model(self):return self.backbone.language

model=Model()
settings={'trainable_scope':'full'}
before=copy.deepcopy(settings)
summary=configure_trainable_scope(model,settings,from_s0=True)
assert all(p.requires_grad for p in model.parameters())
assert summary['backbone_trainable_parameters']>0 and settings==before
summary=configure_trainable_scope(model,{},from_s0=True)
assert summary['trainable_scope']=='policy'
assert not any(p.requires_grad for p in model.backbone.parameters())
assert all(p.requires_grad for p in model.action_head.parameters())
summary=configure_trainable_scope(model,{},from_s0=False)
assert summary['trainable_parameters']==summary['total_parameters']
summary=configure_trainable_scope(model,{'trainable_scope':'top_layers','top_layer_count':1},from_s0=True)
assert not any(p.requires_grad for p in model.backbone.visual.parameters())
assert not any(p.requires_grad for p in model.language_model.layers[0].parameters())
assert all(p.requires_grad for p in model.language_model.layers[-1].parameters())
assert all(p.requires_grad for p in model.language_model.norm.parameters())
flags=[p.requires_grad for p in model.parameters()]
for settings in ({'trainable_scope':'bad'}, {'trainable_scope':'top_layers'},
                 {'trainable_scope':'top_layers','top_layer_count':4},
                 {'trainable_scope':'top_layers','top_layer_count':True},
                 {'trainable_scope':'full','top_layer_count':1}):
    try:configure_trainable_scope(model,settings,from_s0=True)
    except ValueError:pass
    else:raise AssertionError('Accepted invalid scope')
    assert [p.requires_grad for p in model.parameters()]==flags
print(json.dumps({'scope_checks':'passed','legacy_defaults_preserved':True,'invalid_scope_rejected':True}))
