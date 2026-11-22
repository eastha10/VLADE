"""Reject unregistered explicit S0 paths before reading model bytes or making outputs."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

source,scratch=map(Path,sys.argv[1:])
sys.path.insert(0,str(source.resolve()/'src'))
from vlade.training import qwen_vla_task as trainer
scratch.mkdir(parents=True,exist_ok=False)
raw=scratch/'unregistered'
raw.mkdir()
output=scratch/'output'
settings={'max_steps':1,'accumulation_steps':1,'save_steps':1,'validation_queries':1,
          'initialization':'student_checkpoint'}
with (patch.object(trainer,'source_identity',return_value={'git_commit':'synthetic'}),
      patch.object(trainer.torch.cuda,'is_available',return_value=True),
      patch.object(trainer.torch.cuda,'is_bf16_supported',return_value=True),
      patch.object(trainer.QwenVLA,'load') as load):
    try:trainer.run(settings,output,local_s0=raw,local_s0_manifest_sha256='a'*64)
    except ValueError as error:assert 'verified HF cache' in str(error)
    else:raise AssertionError('Unregistered local S0 was accepted')
    load.assert_not_called()
assert not output.exists()
print(json.dumps({'unregistered_local_s0':'rejected_before_model_read_and_output'}))
