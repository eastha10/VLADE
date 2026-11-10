"""CPU-only contract checks; package builds, installation and GPU calls are mocked."""
import os
import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(os.environ["VLADE_PROJECT_ROOT"]) / "src"))
from vlade.utils import colab_notebooks


BASE = b'''class LlamaModel:
    def forward(self):
        layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
'''
PATCHED = BASE.replace(b"                    cache_position,\n", b"                    cache_position,\n"
                      b"                    extra_attn_weights if extra_layer_ids and layer_idx in extra_layer_ids else None,\n")
BASE_SHA = hashlib.sha256(BASE).hexdigest()
PATCHED_SHA = hashlib.sha256(PATCHED).hexdigest()
REVISION = "a03eee5da1794c45f5a911b60a9a4545e6e0ed1f"
PATCH_REL = "configs/models/teacher/transformers-avavla-a03eee5-checkpoint-extra-attn.patch"


@pytest.mark.parametrize('failure',[None,'patch','revision','source','apply-check','apply','patched-source','build','wheel-content'])
def test_managed_build_without_prismatic_checkout(tmp_path,monkeypatch,failure):
    from vlade.runtime import ava_patch
    code=tmp_path/'code';patch=code/PATCH_REL;patch.parent.mkdir(parents=True)
    patch.write_bytes((Path(os.environ['VLADE_PROJECT_ROOT'])/PATCH_REL).read_bytes() if failure!='patch' else b'bad')
    monkeypatch.setattr(ava_patch,'BASE_SHA',BASE_SHA)
    monkeypatch.setattr(ava_patch,'INSTALLED_SHA',PATCHED_SHA)
    calls=[]
    def run(args,**kwargs):
        args=list(map(str,args));calls.append(args)
        if args[0]=='git':
            source=Path(args[args.index('-C')+1]);llama=source/'src/transformers/models/llama/modeling_llama.py'
            if 'checkout' in args:
                llama.parent.mkdir(parents=True);llama.write_bytes(BASE if failure!='source' else b'wrong')
            if 'apply' in args:
                stage='apply-check' if '--check' in args else 'apply'
                if failure==stage: raise RuntimeError(stage)
                if stage=='apply':llama.write_bytes(PATCHED if failure!='patched-source' else BASE)
        elif args[:2]==['uv','build']:
            if failure=='build':raise RuntimeError('build')
            wheels=Path(args[args.index('--out-dir')+1])
            with zipfile.ZipFile(wheels/'transformers-test.whl','w') as z:
                z.writestr('transformers/models/llama/modeling_llama.py',BASE if failure=='wheel-content' else PATCHED)
    monkeypatch.setattr(ava_patch.subprocess,'run',run)
    monkeypatch.setattr(ava_patch.subprocess,'check_output',lambda *a,**k: 'wrong' if failure=='revision' else REVISION)
    if failure:
        with pytest.raises((ValueError,RuntimeError)):
            ava_patch.build(code,tmp_path/'build','uv',{})
    else:
        wheel=ava_patch.build(code,tmp_path/'build','uv',{})
        assert wheel.is_file() and not (code/'third_party').exists()
        assert any('apply' in c and '--check' in c for c in calls)
        assert any('apply' in c and '--check' not in c for c in calls)


@pytest.mark.parametrize('failure',['install','check','verify','freeze',None])
def test_runtime_ready_only_after_every_check(tmp_path,monkeypatch,failure):
    from vlade.runtime import environment,ava_patch
    code=tmp_path/'code';folder=code/'configs/runtime';folder.mkdir(parents=True)
    spec={'schema_version':1,'name':'ava','python':'3.11','packages':['torch==2.2.0'],'managed_ava_patch':True}
    (folder/'ava.json').write_text(json.dumps(spec))
    monkeypatch.setattr(environment,'ensure_uv',lambda *a:'uv')
    monkeypatch.setattr(ava_patch,'build',lambda *a:tmp_path/'approved.whl')
    def run(args,**kwargs):
        stage='install' if 'install' in args else 'check' if 'check' in args else 'venv'
        if failure==stage:raise RuntimeError(stage)
        if stage=='venv':
            python=environment.interpreter(Path(args[-1]));python.parent.mkdir(parents=True);python.touch()
    monkeypatch.setattr(environment,'command',run)
    def verify(*a):
        if failure=='verify':raise RuntimeError('verify')
    monkeypatch.setattr(ava_patch,'verify',verify)
    def freeze(*a,**k):
        if failure=='freeze':raise RuntimeError('freeze')
        return 'torch==2.2.0\n'
    monkeypatch.setattr(environment.subprocess,'check_output',freeze)
    if failure:
        with pytest.raises(RuntimeError):environment.prepare(code,tmp_path/'work','ava')
        assert not list((tmp_path/'work').rglob('vlade-runtime.json'))
        with pytest.raises(FileNotFoundError):environment.locate(code,tmp_path/'work','ava')
    else:
        python=environment.prepare(code,tmp_path/'work','ava')
        assert environment.locate(code,tmp_path/'work','ava')==python


@pytest.mark.parametrize("failure", [None, "hash", "outside", "not-venv", "argument"])
def test_imported_source_verifier(tmp_path, monkeypatch, failure):
    venv = tmp_path / "venv"
    folder = tmp_path / "outside" if failure == "outside" else venv
    source = folder / "models/llama/modeling_llama.py"
    source.parent.mkdir(parents=True)
    payload = PATCHED if failure != "argument" else BASE
    source.write_bytes(payload)
    expected = "bad" if failure == "hash" else hashlib.sha256(payload).hexdigest()
    original_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace(__file__=str(source))
                        if name == "transformers.models.llama.modeling_llama" else original_import(name))
    monkeypatch.setattr(sys, "prefix", sys.base_prefix if failure == "not-venv" else str(venv))
    monkeypatch.setattr(sys, "argv", ["verify", expected, str(venv)])
    script = colab_notebooks._ava_installed_transformers_check()
    if failure:
        with pytest.raises(RuntimeError):
            exec(script, {})
    else:
        exec(script, {})
