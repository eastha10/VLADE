"""Generate short Colab clients for the config-driven, standalone research CLI."""

from __future__ import annotations
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[3]
PROFILES = {
    "colab_01_download_data_weights.ipynb": ("prepare_inputs", "01 · HF 입력 준비"),
    "colab_02_ava_fft_and_fixed_cache.ipynb": ("ava_fft", "02 · AVA FFT·T1 선택·cache"),
    "colab_03_qwen_task_only.ipynb": ("qwen_task", "03 · Qwen Task-only"),
    "colab_04_ava_paper_best.ipynb": ("ava_paper", "04 · AVA 논문 기준 LoRA/TBPTT"),
    "colab_04_openvla_oft_fft.ipynb": ("oft_fft", "04 · OpenVLA-OFT FFT"),
    "colab_05_ava_a100_one_step_reload.ipynb": (
        "ava_one_step",
        "05 · Canonical 1-Episode·1-step·별도 재로딩",
    ),
    "colab_05_qwen_action_distillation.ipynb": ("qwen_kd", "05 · Qwen Action KD"),
    "colab_ava_fft_suite_balanced_cu.ipynb": ("ava_suite", "AVA Suite-balanced FFT"),
    "colab_inference_smoke.ipynb": ("inference_ava", "실제 가중치 단일 추론"),
    "colab_06_evaluate_libero_plus.ipynb": ("evaluate_ava", "06 · LIBERO+ paired rollout 평가"),
}


def cell(kind: str, text: str, name: str) -> dict:
    source = textwrap.dedent(text).strip() + "\n"
    value = {
        "cell_type": kind,
        "id": name,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        compile(source, name, "exec")
        value.update(execution_count=None, outputs=[])
    return value


def bootstrap(source_ref: str = "main", **kwargs) -> str:
    return (
        textwrap.dedent(
            """
        from pathlib import Path
        import os, re, subprocess, sys, time
        os.environ.setdefault('VLADE_SESSION_STARTED_UNIX', str(time.time()))
        SOURCE_REF = __REF__  # 재개할 때는 이전 SOURCE_COMMIT을 지정하십시오.
        REPOSITORY = 'https://github.com/eastha10/VLADE.git'
        SOURCE_COMMIT = SOURCE_REF if re.fullmatch(r'[0-9a-f]{40}', SOURCE_REF) else subprocess.check_output(['git','ls-remote',REPOSITORY,SOURCE_REF], text=True).split()[0]
        if not re.fullmatch(r'[0-9a-f]{40}', SOURCE_COMMIT): raise ValueError('불변 Git commit이 필요합니다.')
        CODE = Path('/content/vlade_sources') / SOURCE_COMMIT
        if any(name.startswith('vlade.') and getattr(module,'__file__',None) and not Path(module.__file__).resolve().is_relative_to(CODE) for name,module in list(sys.modules.items())):
            raise RuntimeError('다른 소스가 로드되어 있습니다. Colab 런타임을 다시 시작하십시오.')
        if not CODE.exists():
            subprocess.run(['git','clone','--no-checkout','--filter=blob:none',REPOSITORY,str(CODE)],check=True)
            subprocess.run(['git','-C',str(CODE),'fetch','--depth','1','origin',SOURCE_COMMIT],check=True)
            subprocess.run(['git','-C',str(CODE),'checkout','--detach',SOURCE_COMMIT],check=True)
        sys.path[:0] = [str(CODE/'src'),str(CODE)]
        from vlade.utils.source_identity import source_identity
        assert source_identity(CODE,require_clean=True)['git_commit']==SOURCE_COMMIT
        print('SOURCE_COMMIT:',SOURCE_COMMIT)
    """
        )
        .replace("__REF__", repr(source_ref))
        .strip()
    )


def write_thin_notebook(path: Path, profile: str, title: str) -> Path:
    config = "configs/research/" + profile + ".json"
    intro = f"# {title}\n\n실행 로직은 프로젝트 모듈과 CLI에 있습니다. `CONFIG`의 훈련 step·평가 rollout·seed·학습률·주기·예산을 편집하거나 `OVERRIDES`로 지정하십시오. 재개는 같은 run_group과 고정 SOURCE_COMMIT 및 원래 설정을 사용합니다. 기본 실행 스위치는 모두 False입니다. 가중치/데이터는 고정 HF 입력이며 Drive는 자동 갱신하지 않습니다.\n\n실험 단계: prepare_inputs → 훈련/검증 → select_teacher → cache_teacher → qwen_kd → evaluate_ava/evaluate_oft. Suite-balanced는 pilot4/main을 구분합니다."
    controls = """
        import datetime, uuid
        from vlade.research.client import job
        WORKSPACE = '/content/vlade_work'
        CONFIG = str(CODE / __CONFIG__)
        RUN_GROUP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        OVERRIDES = {'run_group': RUN_GROUP}  # 예: 'train.max_steps': 1000 또는 'evaluation.rollouts_per_task': 10
        RUN_SETUP = False
        RUN_EXPERIMENT = False
    """.replace(
        "__CONFIG__", repr(config)
    )
    cells = [
        cell("markdown", intro, "purpose"),
        cell("code", bootstrap(), "source"),
        cell(
            "code",
            "from google.colab import userdata\nos.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')",
            "authentication",
        ),
        cell("code", controls, "config"),
        cell("code", "job(CONFIG,action='plan',workspace=WORKSPACE,overrides=OVERRIDES)", "plan"),
        cell(
            "code",
            "if RUN_SETUP:\n    job(CONFIG,action='setup',workspace=WORKSPACE,overrides=OVERRIDES)",
            "setup",
        ),
        cell(
            "code",
            "if RUN_EXPERIMENT:\n    job(CONFIG,action='run',workspace=WORKSPACE,overrides=OVERRIDES)",
            "run",
        ),
    ]
    if profile == "prepare_inputs":
        cells.append(
            cell(
                "code",
                "RUN_FULLFRAMES = False  # 승인된 별도 공개 D/seed42 전체 프레임 게시\nif RUN_FULLFRAMES:\n    job(CODE/'configs/research/fullframes.json',action='setup',workspace=WORKSPACE)\n    job(CODE/'configs/research/fullframes.json',action='run',workspace=WORKSPACE)",
                "export-fullframes-to-hf",
            )
        )
    if profile == "ava_one_step":
        cells.insert(
            -1,
            cell(
                "code",
                "RUN_PREPARE_ONE = False\nif RUN_PREPARE_ONE:\n    job(CODE/'configs/research/prepare_one_episode.json',action='setup',workspace=WORKSPACE)\n    job(CODE/'configs/research/prepare_one_episode.json',action='run',workspace=WORKSPACE)",
                "prepare-one-episode",
            ),
        )
    value = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": path.name, "provenance": []},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def generate() -> list[Path]:
    paths = [
        write_thin_notebook(ROOT / "experiments" / name, *profile)
        for name, profile in PROFILES.items()
    ]
    manifest = ROOT / "experiments/manifests/colab_source.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": 2,
        "distribution": "git_commit",
        "repository": "https://github.com/eastha10/VLADE.git",
        "default_ref": "main",
        "runtime_source": "project_owned",
        "submodules_required": False,
    }
    manifest.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return paths + [manifest]


def _ava_installed_transformers_check() -> str:
    from vlade.runtime.ava_patch import VERIFY_PROGRAM

    return VERIFY_PROGRAM


def ava_install_source() -> str:
    return "from vlade.runtime.environment import prepare\nAVA_PY = prepare(CODE, ROOT, 'ava')\nAVA_ENV_READY = True"


def main() -> None:
    """Regenerate maintained notebooks from the project CLI profiles."""
    for path in generate():
        print(path.relative_to(ROOT))
