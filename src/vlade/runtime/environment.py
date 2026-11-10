"""Prepare the pinned research environments used by the maintained Colab jobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
import time


def interpreter(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def command(args: list, *, env: dict) -> None:
    deadline = float(env.get("VLADE_DEADLINE_UNIX", 0))
    remaining = deadline - time.time() if deadline else None
    if remaining is not None and remaining <= 0:
        raise TimeoutError("Research setup deadline expired")
    subprocess.run([str(value) for value in args], env=env, check=True, timeout=remaining)


def runtime_environment(work: Path) -> dict[str, str]:
    """Keep environment/cache files in the selected runtime workspace."""
    env = os.environ.copy()
    temporary = work / "temporary"
    temporary.mkdir(parents=True, exist_ok=True)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="disabled",
        TF_CPP_MIN_LOG_LEVEL="2",
        UV_CACHE_DIR=str(work / "cache/uv"),
        HF_HOME=str(work / "cache/huggingface"),
        VLADE_ASSET_CACHE=str(work / "assets"),
        TORCH_HOME=str(work / "cache/torch"),
        TEMP=str(temporary),
        TMP=str(temporary),
        TMPDIR=str(temporary),
        HF_TOKEN_PATH=str(work / "cache/hf_token"),
    )
    return env


def ensure_uv(work: Path, env: dict) -> str:
    """Install the environment manager in its own runtime tool environment."""
    existing = shutil.which("uv")
    if existing:
        return existing
    tools = work / "tools"
    python = interpreter(tools)
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(tools)
    command([python, "-m", "pip", "install", "uv"], env=env)
    return str(tools / ("Scripts/uv.exe" if os.name == "nt" else "bin/uv"))


def specification(code: Path, name: str) -> dict:
    if not name.replace("_", "").isalnum():
        raise ValueError("Invalid runtime name")
    value = json.loads((code / "configs/runtime" / (name + ".json")).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("name") != name:
        raise ValueError("Unsupported runtime specification")
    return value


def prepare(code: Path, work: Path, name: str) -> Path:
    """Build/install a research environment; this is called only by explicit setup."""
    spec = specification(code, name)
    work = work.resolve()
    env = runtime_environment(work)
    uv = ensure_uv(work, env)
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    directory = work / "environments" / (name + "-" + identity)
    python = interpreter(directory)
    ready = directory / "vlade-runtime.json"
    if ready.is_file():
        if json.loads(ready.read_text(encoding="utf-8")) != spec:
            raise ValueError("Existing environment differs from its specification")
        if spec.get("managed_ava_patch"):
            from .ava_patch import verify

            verify(python, directory)
        if spec.get("benchmark_source"):
            from .benchmark_source import verify as verify_benchmark

            verify_benchmark(spec, python)
        return python
    if spec.get("system_packages") and sys.platform == "linux":
        command(["apt-get", "update", "-qq"], env=env)
        command(["apt-get", "install", "-y", *spec["system_packages"]], env=env)
    if not python.is_file():
        directory.parent.mkdir(parents=True, exist_ok=True)
        command([uv, "venv", "--python", spec["python"], directory], env=env)
    packages = list(spec["packages"])
    if spec.get("managed_ava_patch"):
        from .ava_patch import build

        packages.append(str(build(code, work / "build", uv, env)))
    command([uv, "pip", "install", "--python", python, *packages], env=env)
    command([uv, "pip", "check", "--python", python], env=env)
    if spec.get("benchmark_source"):
        from .benchmark_source import install, verify as verify_benchmark

        install(spec, work, python, env)
        verify_benchmark(spec, python)
    if spec.get("managed_ava_patch"):
        from .ava_patch import verify

        verify(python, directory)
    freeze = subprocess.check_output(
        [uv, "pip", "freeze", "--python", str(python)], text=True, env=env
    )
    (directory / "environment-freeze.txt").write_text(freeze, encoding="utf-8")
    (directory / "vlade-runtime.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return python


def locate(code: Path, work: Path, name: str) -> Path:
    spec = specification(code, name)
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    directory = work / "environments" / (name + "-" + identity)
    python = interpreter(directory)
    if not python.is_file() or not (directory / "vlade-runtime.json").is_file():
        raise FileNotFoundError("Run the research setup command for this config first")
    if json.loads((directory / "vlade-runtime.json").read_text()) != spec:
        raise ValueError("Installed research environment differs from its config")
    if spec.get("managed_ava_patch"):
        from .ava_patch import verify

        verify(python, directory)
    if spec.get("benchmark_source"):
        from .benchmark_source import verify as verify_benchmark

        verify_benchmark(spec, python)
    return python
