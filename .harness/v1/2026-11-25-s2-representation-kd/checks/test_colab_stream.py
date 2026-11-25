"""Child failures must reach notebook output instead of a huge command repr."""
import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
assert ROOT == Path.cwd().resolve() and (ROOT / 'AGENTS.md').is_file()
sys.path.insert(0,str(ROOT / 'src'))
from vlade.utils.representation_kd_notebook import RUN_CELL


def wrapper():
    tree = ast.parse(RUN_CELL)
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == 'stream_s2')
    scope = {'os':os,'signal':signal,'subprocess':subprocess,'ACTION':'preflight'}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<notebook stream>','exec'),scope)
    return scope['stream_s2']


def test_real_child_traceback_is_printed_before_short_failure(tmp_path,capsys):
    with pytest.raises(RuntimeError,match='S2 preflight 실패') as caught:
        wrapper()([sys.executable,'-B','-u','-c',
            "import sys; print('child stdout',flush=True); print('child stderr',file=sys.stderr,flush=True); raise ValueError('actual diagnostic')"],
            cwd=tmp_path,env=os.environ.copy())
    output = capsys.readouterr().out
    assert 'child stdout' in output and 'child stderr' in output
    assert 'ValueError: actual diagnostic' in output and 'Traceback' in output
    assert '-c' not in str(caught.value)


def test_success_streams_both_outputs(tmp_path,capsys):
    wrapper()([sys.executable,'-B','-u','-c',
        "import sys; print('ok stdout',flush=True); print('ok stderr',file=sys.stderr,flush=True)"],
        cwd=tmp_path,env=os.environ.copy())
    output = capsys.readouterr().out
    assert 'ok stdout' in output and 'ok stderr' in output


def test_interrupt_signals_only_own_new_process_group(monkeypatch):
    calls = []
    class Pipe:
        closed = False
        def __iter__(self): raise KeyboardInterrupt
        def close(self): self.closed = True
    pipe = Pipe()
    process = SimpleNamespace(pid=12345,stdout=pipe,wait=lambda:calls.append('wait'))
    def popen(*args,**kwargs):
        assert kwargs['start_new_session'] is True
        assert kwargs['stderr'] is subprocess.STDOUT
        assert kwargs['env']['PYTHONUNBUFFERED'] == '1'
        return process
    monkeypatch.setattr(subprocess,'Popen',popen)
    monkeypatch.setattr(os,'killpg',lambda pid,sig:calls.append((pid,sig)),raising=False)
    with pytest.raises(KeyboardInterrupt):
        wrapper()(['synthetic'],cwd=ROOT,env={})
    assert calls == [(12345,signal.SIGINT),'wait'] and pipe.closed
