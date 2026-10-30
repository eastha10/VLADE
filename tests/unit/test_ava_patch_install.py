"""CPU-only contract checks; package builds, installation and GPU calls are mocked."""
import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
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


def installation_probe(tmp_path, failure=None):
    code = tmp_path / "code"
    content = tmp_path / "content"
    shared = tmp_path / "shared"
    for path in (code, content, shared):
        path.mkdir()
    patch = Path(__file__).resolve().parents[2] / PATCH_REL
    staged = code / PATCH_REL
    staged.parent.mkdir(parents=True)
    staged.write_bytes(patch.read_bytes() if failure != "patch" else b"damaged")
    manifest = code / "experiments/manifests/colab_source.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"files_sha256": {PATCH_REL:
        "bad" if failure == "manifest" else hashlib.sha256(patch.read_bytes()).hexdigest()}}))
    upstream = code / "third_party/ava_vla"
    upstream.mkdir(parents=True)
    (upstream / "pyproject.toml").write_text('[project]\ndependencies = ["torch==2.2.0", '
        '"torchvision==0.17.0", "transformers @ old", "dlimp @ old", "huggingface_hub", "accelerate"]')
    calls, dependencies = [], []
    venv = content / "vlade_ava_fft_py311"
    python = venv / "bin/python"

    def run(args, env=None):
        command = [str(value) for value in args]
        calls.append(command)
        if command[0] == "git":
            source = Path(command[2])
            llama = source / "src/transformers/models/llama/modeling_llama.py"
            if "checkout" in command:
                llama.parent.mkdir(parents=True)
                llama.write_bytes(BASE if failure != "source" else b"wrong source")
            elif "apply" in command:
                stage = "apply-check" if "--check" in command else "apply"
                if failure == stage:
                    raise RuntimeError(stage)
                if stage == "apply":
                    llama.write_bytes(PATCHED if failure != "patched-source" else BASE)
        elif command[:2] == ["uv", "build"]:
            if failure == "build":
                raise RuntimeError("build")
            wheels = Path(command[command.index("--out-dir") + 1])
            with zipfile.ZipFile(wheels / "transformers-4.40.1-py3-none-any.whl", "w") as archive:
                archive.writestr("transformers/models/llama/modeling_llama.py",
                                 BASE if failure == "wheel-content" else PATCHED)
        elif command[:3] == ["uv", "pip", "install"] and failure == "install":
            raise RuntimeError("install")
        elif command[0] == str(python) and command[1] == "-c":
            stage = "installed" if "importlib.import_module" in command[2] else "cuda-check"
            if failure == stage:
                raise RuntimeError(stage)

    def environment(name, version, packages):
        assert name == "vlade_ava_fft_py311" and version == "3.11"
        dependencies.extend(packages)
        if failure == "environment":
            raise RuntimeError("environment")
        python.parent.mkdir(parents=True)
        python.write_bytes(b"mock venv interpreter")
        if failure == "provenance":
            (venv / "vlade_transformers_patch.json").mkdir()
        return python

    def output(args, text):
        if args[0] == "git":
            return "wrong revision" if failure == "revision" else REVISION
        if failure == "freeze":
            raise RuntimeError("freeze")
        return "torch==2.2.0\n"

    namespace = {"Path": lambda value: content if str(value) == "/content" else Path(value),
                 "CODE": code, "ROOT": shared, "ENV": {}, "UV": "uv", "run": run,
                 "environment": environment, "subprocess": SimpleNamespace(check_output=output),
                 "json": json, "hashlib": hashlib}
    source = colab_notebooks.ava_install_source().replace(
        "62f22856fe7b8459d6d61d4676b0075b65724979f84859ac475dbe36c10a3d80", BASE_SHA).replace(
        "0a9c25300f3358f87af1719199608af3c27567dba0425690327fccd44cfb516b", PATCHED_SHA)
    return SimpleNamespace(namespace=namespace, source=source, calls=calls,
                           dependencies=dependencies, python=python, staged=staged)


def test_installation_chain_and_fft_gate(tmp_path):
    probe = installation_probe(tmp_path)
    exec(probe.source, probe.namespace)
    assert probe.namespace["AVA_ENV_READY"] is True
    assert "torch==2.2.0" in probe.dependencies and "torchvision==0.17.0" in probe.dependencies
    assert "bitsandbytes==0.43.3" in probe.dependencies and "accelerate==1.15.0" in probe.dependencies
    installs = [c for c in probe.calls if c[:3] == ["uv", "pip", "install"]]
    assert all(c[c.index("--python") + 1] == str(probe.python) for c in installs)
    assert any("--force-reinstall" in c and "--no-deps" in c for c in installs)
    probe.namespace["run"]([probe.python, "scripts/train/run_fft.py"])
    assert probe.calls[-1][-1] == "scripts/train/run_fft.py"
    probe.staged.write_bytes(b"damaged on rerun")
    with pytest.raises(RuntimeError, match="patch"):
        exec(probe.source, probe.namespace)
    before = len(probe.calls)
    with pytest.raises(RuntimeError, match="not verified"):
        probe.namespace["run"]([probe.python, "scripts/train/run_fft.py"])
    assert len(probe.calls) == before


@pytest.mark.parametrize("failure", ["manifest", "patch", "revision", "source", "apply-check", "apply",
                                     "patched-source", "build", "wheel-content", "environment", "install",
                                     "installed", "cuda-check", "freeze", "provenance"])
def test_each_failure_blocks_fft(tmp_path, failure):
    probe = installation_probe(tmp_path, failure)
    with pytest.raises((RuntimeError, OSError)):
        exec(probe.source, probe.namespace)
    assert probe.namespace["AVA_ENV_READY"] is False and probe.namespace["AVA_PY"] is None
    before = len(probe.calls)
    with pytest.raises(RuntimeError, match="not verified"):
        probe.namespace["require_ava_environment"]()
    with pytest.raises(RuntimeError, match="not verified"):
        probe.namespace["run"]([probe.python, "scripts/train/run_fft.py"])
    assert len(probe.calls) == before


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


def test_generated_ava_notebook_guards_fft_and_avoids_global_pip(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    for name in ("README.md", "STRUCTURE.md", "AGENTS.md"):
        (root / name).write_text(name)
    target = root / PATCH_REL
    target.parent.mkdir(parents=True)
    target.write_bytes((Path(__file__).resolve().parents[2] / PATCH_REL).read_bytes())
    monkeypatch.setattr(colab_notebooks, "ROOT", root)
    monkeypatch.setattr(colab_notebooks, "revision", lambda path: "fixture")
    generated = colab_notebooks.generate()
    assert len(generated) == 4
    notebooks = [json.loads(p.read_text(encoding="utf-8")) for p in generated[:-1]]
    for notebook in notebooks:
        for item in notebook["cells"]:
            if item["cell_type"] == "code":
                compile("".join(item["source"]), item["id"], "exec")
    cells = {item["id"]: "".join(item["source"]) for item in notebooks[1]["cells"]}
    assert 'EXPECTED_SOURCE_MANIFEST_SHA256 = "' in cells["drive-and-source"]
    assert "Path('/content'), EXPECTED_SOURCE_MANIFEST_SHA256)" in cells["drive-and-source"]
    assert "[sys.executable, '-m', 'pip'" not in cells["drive-and-source"]
    assert "[tool_python, '-m', 'pip', 'install', 'uv']" in cells["drive-and-source"]
    assert cells["install-ava"].startswith("AVA_ENV_READY = False")
    for name in ("fft-config", "fft-smoke", "fft-main"):
        assert cells[name].startswith("require_ava_environment()")
    for notebook in (notebooks[0], notebooks[2]):
        assert not any("AVA_ENV_READY" in "".join(item["source"]) for item in notebook["cells"])
