"""Managed AVA Transformers patch build and verification for research runs."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import uuid

FORK = "https://github.com/LiAuto-DSR/transformers-avavla.git"
REVISION = "a03eee5da1794c45f5a911b60a9a4545e6e0ed1f"
PATCH = "configs/models/teacher/transformers-avavla-a03eee5-checkpoint-extra-attn.patch"
PATCH_SHA = "9f0e46c9e93d62a92af950aeee712191cfcf314c1e636a9608fddcb57ff6087d"
BASE_SHA = "62f22856fe7b8459d6d61d4676b0075b65724979f84859ac475dbe36c10a3d80"
INSTALLED_SHA = "0a9c25300f3358f87af1719199608af3c27567dba0425690327fccd44cfb516b"
VERIFY_PROGRAM = "import ast, hashlib, importlib, json, sys\nfrom pathlib import Path\nexpected_sha, expected_prefix = sys.argv[1:3]\nprefix = Path(sys.prefix).resolve()\nif sys.prefix == sys.base_prefix or prefix != Path(expected_prefix).resolve():\n    raise RuntimeError('AVA verification requires the dedicated virtual environment')\nmodule = importlib.import_module('transformers.models.llama.modeling_llama')\nsource = Path(module.__file__).resolve()\nif not source.is_relative_to(prefix) or source.name != 'modeling_llama.py':\n    raise RuntimeError('Transformers was imported outside the AVA environment')\npayload = source.read_bytes()\nif hashlib.sha256(payload).hexdigest() != expected_sha:\n    raise RuntimeError('Installed AVA Transformers source SHA mismatch')\ntree = ast.parse(payload)\nmodel = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LlamaModel')\nforward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')\ncalls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)\n         and isinstance(node.func, ast.Attribute) and node.func.attr == '_gradient_checkpointing_func']\nexpected = 'extra_attn_weights if extra_layer_ids and layer_idx in extra_layer_ids else None'\nif len(calls) != 1 or len(calls[0].args) != 9 or ast.unparse(calls[0].args[-1]) != expected:\n    raise RuntimeError('Installed checkpoint attention argument was not preserved')\nprint(json.dumps({'source': str(source), 'sha256': expected_sha, 'venv': str(prefix)}))"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(python: Path, environment: Path) -> None:
    subprocess.run([str(python), "-c", VERIFY_PROGRAM, INSTALLED_SHA, str(environment)], check=True)


def build(code: Path, work: Path, uv: str, env: dict) -> Path:
    """Build the approved wheel without accessing a Prismatic source checkout."""
    patch = code / PATCH
    if not patch.is_file() or digest(patch) != PATCH_SHA:
        raise ValueError("Approved AVA Transformers patch is missing or differs")
    folder = work / "ava-transformers" / REVISION[:12]
    folder.mkdir(parents=True, exist_ok=True)
    wheel_dir = folder / "wheels"
    existing = list(wheel_dir.glob("transformers-*.whl"))
    if not existing:
        source = folder / ("source-" + uuid.uuid4().hex[:8])
        source.mkdir(exist_ok=True)
        run = lambda args: subprocess.run([str(x) for x in args], env=env, check=True)
        run(["git", "-C", source, "init"])
        run(["git", "-C", source, "fetch", "--depth", "1", FORK, REVISION])
        run(
            ["git", "-c", "core.autocrlf=false", "-C", source, "checkout", "--detach", "FETCH_HEAD"]
        )
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != REVISION:
            raise ValueError("AVA Transformers revision differs")
        llama = source / "src/transformers/models/llama/modeling_llama.py"
        if digest(llama) != BASE_SHA:
            raise ValueError("AVA Transformers base source differs")
        run(["git", "-C", source, "apply", "--check", patch])
        run(["git", "-C", source, "apply", patch])
        if digest(llama) != INSTALLED_SHA:
            raise ValueError("Managed AVA patch result differs")
        wheel_dir.mkdir(exist_ok=True)
        run([uv, "build", "--wheel", "--python", "3.11", "--out-dir", wheel_dir, source])
        existing = list(wheel_dir.glob("transformers-*.whl"))
    if len(existing) != 1:
        raise ValueError("Expected one managed AVA wheel")
    with zipfile.ZipFile(existing[0]) as archive:
        content = archive.read("transformers/models/llama/modeling_llama.py")
    if hashlib.sha256(content).hexdigest() != INSTALLED_SHA:
        raise ValueError("AVA wheel does not contain the approved patch")
    return existing[0]
