"""Generate Colab notebooks that load verified ordinary project files from Drive."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import subprocess
import textwrap
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def source_state(source_revision: str | None, source_files: dict[str, bytes]) -> dict:
    """Compare the working tree and index to a pinned commit; unknown means dirty.

    Only the manifest and notebook 02 are derived outputs. Changes to other
    tracked files, untracked files or the pinned AVA submodule remain dirty.
    """
    state = {"contains_uncommitted_work": True,
             "source_revisions": {".": None, "third_party/ava_vla": None},
             "git_comparison_verified": False}

    def git(directory: Path, *args: str, payload: bytes | None = None) -> bytes:
        result = subprocess.run(["git", "-C", str(directory), *args], input=payload,
                                capture_output=True, check=False)
        if result.returncode:
            raise ValueError("Git source comparison unavailable")
        return result.stdout

    try:
        if source_revision is not None and not source_revision.strip():
            return state
        if Path(git(ROOT, "rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT.resolve():
            return state
        fixed = git(ROOT, "rev-parse", "--verify", "--end-of-options",
                    (source_revision or "HEAD") + "^{commit}").decode().strip()
        state["source_revisions"]["."] = fixed
        paths = ("--", ".", ":(top,exclude)experiments/manifests/colab_source.json",
                 ":(top,exclude)experiments/colab_02_ava_fft_and_fixed_cache.ipynb")
        changed = git(ROOT, "diff", "--no-ext-diff", "--no-textconv", "--name-only", "-z",
                      "--ignore-submodules=none", fixed, *paths)
        pending = git(ROOT, "status", "--porcelain=v1", "-z", "--untracked-files=all",
                      "--ignore-submodules=none", *paths)
        submodule = "third_party/ava_vla"
        fields = git(ROOT, "ls-tree", fixed, "--", submodule).decode().split()
        if fields[:2] != ["160000", "commit"]:
            return state
        expected = fields[2]
        state["source_revisions"][submodule] = expected
        directory = ROOT / submodule
        if Path(git(directory, "rev-parse", "--show-toplevel").decode().strip()).resolve() != directory.resolve():
            return state
        actual = git(directory, "rev-parse", "--verify", "HEAD").decode().strip()
        sub_pending = git(directory, "status", "--porcelain=v1", "-z", "--untracked-files=all")

        def blobs(repository: Path, commit: str) -> dict[str, str]:
            result = {}
            for entry in git(repository, "ls-tree", "-r", "-z", commit).split(b"\0"):
                if not entry:
                    continue
                header, name = entry.split(b"\t", 1)
                mode, kind, oid = header.split()
                if mode in (b"100644", b"100755") and kind == b"blob":
                    result[name.decode()] = oid.decode()
            return result

        root_blobs, sub_blobs = blobs(ROOT, fixed), blobs(directory, expected)
        snapshot_dirty = False
        for name, payload in source_files.items():
            if name.startswith(submodule + "/"):
                relative = name[len(submodule) + 1:]
                repository, tree = directory, sub_blobs
            else:
                relative, repository, tree = name, ROOT, root_blobs
            # Check captured bytes too: ignored files and assume-unchanged index
            # flags must not hide an uncommitted source from the snapshot.
            if relative not in tree:
                snapshot_dirty = True
            else:
                oid = git(repository, "hash-object", "--path=" + relative, "--stdin",
                          payload=payload).decode().strip()
                snapshot_dirty |= oid != tree[relative]
        state["contains_uncommitted_work"] = bool(
            changed or pending or sub_pending or actual != expected or snapshot_dirty)
        state["git_comparison_verified"] = True
    except (OSError, ValueError, UnicodeError):
        # Missing metadata, invalid refs and inaccessible submodules must never
        # turn a plain source copy into an apparently clean checkout.
        pass
    return state


def revision(path: Path) -> str | None:
    process = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    if process.returncode == 0:
        return process.stdout.strip()
    if path != ROOT:
        # Some managed workspaces expose the gitlink but deny submodule metadata.
        reference = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "HEAD", path.relative_to(ROOT).as_posix()],
                                   capture_output=True, text=True)
        fields = reference.stdout.split()
        if reference.returncode == 0 and fields[:2] == ["160000", "commit"]:
            return fields[2]
    return None


def cell(kind: str, text: str, name: str) -> dict:
    result = {"cell_type": kind, "id": name, "metadata": {},
              "source": textwrap.dedent(text).strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
        compile("".join(result["source"]), name, "exec")
    return result


def notebook(path: Path, cells: list[dict]) -> None:
    value = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
             "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                          "language_info": {"name": "python"}, "colab": {"name": path.name, "provenance": []}}}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def stage_source(project_root: Path, scratch_root: Path, expected_manifest_sha256: str) -> Path:
    """Verify Drive source files and stage them without copying data or weights."""
    project_root = project_root.resolve()
    manifest_relative = Path("experiments/manifests/colab_source.json")
    manifest_path = project_root / manifest_relative
    if not manifest_path.is_file():
        raise FileNotFoundError("Drive의 VLADE 폴더에 src/scripts/configs/experiments/third_party 일반 파일 구조를 먼저 복사하십시오: " + str(manifest_path))
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("노트북과 colab_source.json 버전이 다릅니다. 같은 버전의 일반 프로젝트 파일을 사용하십시오.")
    manifest = json.loads(manifest_bytes)
    code = scratch_root / ("vlade_colab_" + expected_manifest_sha256[:12])
    code = code.resolve()
    if code.is_relative_to(project_root) or project_root.is_relative_to(code):
        raise ValueError("실행 소스 임시 폴더와 Drive 프로젝트 폴더는 서로 겹칠 수 없습니다.")
    for relative, expected in manifest["files_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
            raise ValueError("잘못된 소스 상대 경로: " + relative)
        source = project_root / path
        target = code / path
        if not source.resolve().is_relative_to(project_root) or not target.resolve().is_relative_to(code):
            raise ValueError("프로젝트 밖의 소스 경로: " + relative)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Drive 소스가 없거나 변경되었습니다: " + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            shutil.copyfile(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("복사된 소스 무결성 불일치: " + relative)
    # Publish the manifest only after every listed source file has been verified.
    staged_manifest = code / manifest_relative
    staged_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = staged_manifest.with_suffix(".json.tmp")
    temporary_manifest.write_bytes(manifest_bytes)
    temporary_manifest.replace(staged_manifest)
    return code


def bootstrap(digest: str, *, isolated_uv: bool = False) -> str:
    source = '''
from pathlib import Path
from google.colab import drive
import hashlib, json, os, shutil, subprocess, sys
drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/colab/VLADE')  # 세 노트북에서 동일한 저장 폴더를 지정하십시오.
ROOT.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = ROOT  # src/scripts/configs/experiments/third_party가 바로 들어 있는 실제 프로젝트 루트
EXPECTED_MANIFEST_SHA256 = "''' + digest + '''"
''' + inspect.getsource(stage_source) + '''
CODE = stage_source(PROJECT_ROOT, Path('/content'), EXPECTED_MANIFEST_SHA256)
for module_name, module in list(sys.modules.items()):
    if module_name == 'vlade' or module_name.startswith('vlade.'):
        filename = getattr(module, '__file__', None)
        if filename and not Path(filename).resolve().is_relative_to(CODE.resolve()):
            raise RuntimeError('다른 버전의 VLADE가 메모리에 있습니다. Colab 런타임을 다시 시작하십시오.')
sys.path.insert(0, str(CODE / 'src'))
print('Drive:', ROOT, '\\n실행 소스:', CODE)
ENV = os.environ.copy()
ENV.update(TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled', TF_CPP_MIN_LOG_LEVEL='2',
           HF_HOME=str(ROOT / 'hf_cache'))
def run(args, env=None):
    process = subprocess.Popen([str(x) for x in args], cwd=CODE, env=env or ENV,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait():
        raise RuntimeError('실행에 실패했습니다. 위 오류를 확인하십시오. 후속 학습은 시작하지 않습니다.')
if shutil.which('uv') is None:
    run([sys.executable, '-m', 'pip', 'install', 'uv'])
UV = shutil.which('uv')
assert UV
def environment(name, version, packages):
    directory = Path('/content') / name
    python = directory / 'bin/python'
    if not python.is_file():
        run([UV, 'venv', '--python', version, directory])
    run([UV, 'pip', 'install', '--python', python, *packages])
    return python
def completed_checkpoints(directory):
    return sorted(path for path in directory.glob('step-*')
                  if (path / 'COMPLETE').is_file() and (path / 'trainer_state.pt').is_file())
'''
    if isolated_uv:
        source = source.replace("EXPECTED_MANIFEST_SHA256", "EXPECTED_SOURCE_MANIFEST_SHA256")
        source = source.replace(
            "if shutil.which('uv') is None:\n    run([sys.executable, '-m', 'pip', 'install', 'uv'])\nUV = shutil.which('uv')\nassert UV",
            "UV = shutil.which('uv')\nif UV is None:\n"
            "    tool_env = Path('/content/vlade_uv_tool')\n"
            "    tool_python = tool_env / 'bin/python'\n"
            "    if not tool_python.is_file():\n"
            "        run([sys.executable, '-m', 'venv', tool_env])\n"
            "    run([tool_python, '-m', 'pip', 'install', 'uv'])\n"
            "    UV = str(tool_env / 'bin/uv')\nassert UV")
    return source


def _ava_installed_transformers_check() -> str:
    """Child-process verification of the imported package in the AVA-only venv."""
    return textwrap.dedent('''
        import ast, hashlib, importlib, json, sys
        from pathlib import Path
        expected_sha, expected_prefix = sys.argv[1:3]
        prefix = Path(sys.prefix).resolve()
        if sys.prefix == sys.base_prefix or prefix != Path(expected_prefix).resolve():
            raise RuntimeError('AVA verification requires the dedicated virtual environment')
        module = importlib.import_module('transformers.models.llama.modeling_llama')
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(prefix) or source.name != 'modeling_llama.py':
            raise RuntimeError('Transformers was imported outside the AVA environment')
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise RuntimeError('Installed AVA Transformers source SHA mismatch')
        tree = ast.parse(payload)
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LlamaModel')
        forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == '_gradient_checkpointing_func']
        expected = 'extra_attn_weights if extra_layer_ids and layer_idx in extra_layer_ids else None'
        if len(calls) != 1 or len(calls[0].args) != 9 or ast.unparse(calls[0].args[-1]) != expected:
            raise RuntimeError('Installed checkpoint attention argument was not preserved')
        print(json.dumps({'source': str(source), 'sha256': expected_sha, 'venv': str(prefix)}))
    ''').strip()


def ava_install_source() -> str:
    """Build the verified managed patch into a wheel; fail closed before any AVA run."""
    return textwrap.dedent(r'''
        AVA_ENV_READY = False
        AVA_PY = None
        import tempfile, tomllib, zipfile
        AVA_VENV = Path('/content') / 'vlade_ava_fft_py311'
        AVA_PATCH_REL = 'configs/models/teacher/transformers-avavla-a03eee5-checkpoint-extra-attn.patch'
        AVA_PATCH_SHA = '9f0e46c9e93d62a92af950aeee712191cfcf314c1e636a9608fddcb57ff6087d'
        AVA_TRANSFORMERS_REV = 'a03eee5da1794c45f5a911b60a9a4545e6e0ed1f'
        AVA_BASE_SHA = '62f22856fe7b8459d6d61d4676b0075b65724979f84859ac475dbe36c10a3d80'
        AVA_INSTALLED_SHA = '0a9c25300f3358f87af1719199608af3c27567dba0425690327fccd44cfb516b'
        AVA_VERIFY_SOURCE = __AVA_VERIFY_SOURCE__
        _VLADE_AVA_BASE_RUN = getattr(run, '_vlade_base_run', run)

        def verify_ava_transformers(python):
            _VLADE_AVA_BASE_RUN([python, '-c', AVA_VERIFY_SOURCE, AVA_INSTALLED_SHA, AVA_VENV])

        def require_ava_environment():
            if not AVA_ENV_READY or AVA_PY is None:
                raise RuntimeError('AVA patched environment is not verified; rerun the installation cell')
            verify_ava_transformers(AVA_PY)

        def run(args, env=None):
            command = [str(value) for value in args]
            if command and (command[0] == str(AVA_VENV / 'bin/python')
                            or any(Path(value).name == 'run_fft.py' for value in command)):
                require_ava_environment()
            return _VLADE_AVA_BASE_RUN(args, env=env)

        run._vlade_base_run = _VLADE_AVA_BASE_RUN
        patch = CODE / AVA_PATCH_REL
        source_manifest = json.loads((CODE / 'experiments/manifests/colab_source.json').read_text())
        if source_manifest['files_sha256'].get(AVA_PATCH_REL) != AVA_PATCH_SHA:
            raise RuntimeError('Staged source manifest does not pin the approved AVA patch')
        if not patch.is_file() or hashlib.sha256(patch.read_bytes()).hexdigest() != AVA_PATCH_SHA:
            raise RuntimeError('Staged AVA patch is missing or has the wrong SHA')

        build_root = Path(tempfile.mkdtemp(prefix='vlade-ava-transformers-', dir=AVA_VENV.parent))
        source = build_root / 'source'
        source.mkdir()
        build_env = ENV.copy()
        build_env.update(UV_CACHE_DIR=str(build_root / 'uv-cache'), TMPDIR=str(build_root / 'tmp'))
        (build_root / 'tmp').mkdir()
        fork = 'https://github.com/LiAuto-DSR/transformers-avavla.git'
        _VLADE_AVA_BASE_RUN(['git', '-C', source, 'init'], env=build_env)
        _VLADE_AVA_BASE_RUN(['git', '-C', source, 'fetch', '--depth', '1', fork, AVA_TRANSFORMERS_REV], env=build_env)
        _VLADE_AVA_BASE_RUN(['git', '-C', source, 'checkout', '--detach', 'FETCH_HEAD'], env=build_env)
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != AVA_TRANSFORMERS_REV:
            raise RuntimeError('AVA Transformers checkout revision mismatch')
        llama = source / 'src/transformers/models/llama/modeling_llama.py'
        if hashlib.sha256(llama.read_bytes()).hexdigest() != AVA_BASE_SHA:
            raise RuntimeError('Unpatched AVA Transformers source SHA mismatch')
        _VLADE_AVA_BASE_RUN(['git', '-C', source, 'apply', '--check', patch], env=build_env)
        _VLADE_AVA_BASE_RUN(['git', '-C', source, 'apply', patch], env=build_env)
        if hashlib.sha256(llama.read_bytes()).hexdigest() != AVA_INSTALLED_SHA:
            raise RuntimeError('Patched AVA Transformers source SHA mismatch')
        wheels = build_root / 'wheels'
        wheels.mkdir()
        _VLADE_AVA_BASE_RUN([UV, 'build', '--wheel', '--python', '3.11', '--out-dir', wheels, source], env=build_env)
        wheel_files = list(wheels.glob('transformers-*.whl'))
        if len(wheel_files) != 1:
            raise RuntimeError('Expected exactly one patched Transformers wheel')
        wheel = wheel_files[0]
        with zipfile.ZipFile(wheel) as archive:
            wheel_source = archive.read('transformers/models/llama/modeling_llama.py')
        if hashlib.sha256(wheel_source).hexdigest() != AVA_INSTALLED_SHA:
            raise RuntimeError('Built Transformers wheel does not contain the approved patch')

        upstream = CODE / 'third_party/ava_vla'
        dependencies = tomllib.loads((upstream / 'pyproject.toml').read_text())['project']['dependencies']
        dependencies = [dep for dep in dependencies if not dep.startswith(('transformers @', 'dlimp @', 'huggingface_hub', 'accelerate'))]
        dependencies += [
            str(wheel),
            'dlimp @ git+https://github.com/moojink/dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46',
            'numpy==1.26.4', 'tensorflow-metadata==1.15.0', 'bitsandbytes==0.43.3',
            'huggingface_hub==0.36.2', 'accelerate==1.15.0']
        candidate_python = environment('vlade_ava_fft_py311', '3.11', dependencies)
        if str(candidate_python) != str(AVA_VENV / 'bin/python'):
            raise RuntimeError('Refusing to install outside the dedicated AVA environment')
        _VLADE_AVA_BASE_RUN([UV, 'pip', 'install', '--python', candidate_python,
                            '--no-deps', '--force-reinstall', wheel], env=build_env)
        _VLADE_AVA_BASE_RUN([UV, 'pip', 'install', '--python', candidate_python, '--no-deps', '-e', upstream])
        verify_ava_transformers(candidate_python)
        _VLADE_AVA_BASE_RUN([candidate_python, '-c', "import torch,transformers,tensorflow_datasets; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory/2**30)"])
        freeze = subprocess.check_output([str(UV), 'pip', 'freeze', '--python', str(candidate_python)], text=True)
        (ROOT / 'ava_environment_freeze.txt').write_text(freeze)
        (AVA_VENV / 'vlade_transformers_patch.json').write_text(json.dumps({
            'base_commit': AVA_TRANSFORMERS_REV, 'patch_sha256': AVA_PATCH_SHA,
            'installed_source_sha256': AVA_INSTALLED_SHA,
            'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}, indent=2))
        AVA_PY = candidate_python
        AVA_ENV_READY = True
        print('AVA managed Transformers patch verified; FFT environment ready')
    ''').replace('__AVA_VERIFY_SOURCE__', repr(_ava_installed_transformers_check())).strip()


def generate(*, source_revision: str | None = None, only_notebook: int | None = None) -> list[Path]:
    """Generate all notebooks, or only 02, against one immutable source commit."""
    if only_notebook not in (None, 2):
        raise ValueError("Only notebook 02 supports selective generation")
    paths = []
    manifest_path = ROOT / "experiments/manifests/colab_source.json"
    for folder in ("src", "scripts", "configs", "docs", "experiments/baselines", "experiments/manifests", "third_party/ava_vla"):
        for path in (ROOT / folder).rglob("*"):
            if path == manifest_path or not path.is_file() or any(part in {".git", "__pycache__", ".pytest_cache", "wandb", "node_modules"} for part in path.parts):
                continue
            if path.suffix.lower() in {".py", ".json", ".toml", ".txt", ".md", ".yaml", ".yml", ".patch"} or path.name in {"LICENSE", "NOTICE"}:
                paths.append(path)
    paths.extend(ROOT / name for name in ("README.md", "STRUCTURE.md", "AGENTS.md"))
    payloads = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(set(paths))}
    file_hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    state = source_state(source_revision, payloads)
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
           for name, expected in file_hashes.items()):
        state["contains_uncommitted_work"] = True
        state["git_comparison_verified"] = False
    manifest = {"schema_version": 1, "built_for": "2026-10-28_kanban_colab_preparation",
                "contains_uncommitted_work": state["contains_uncommitted_work"],
                "revision_note": "Source revision is pinned independently of derived manifest/notebook-02 commits; unverified Git state is dirty. files_sha256 records actual source bytes.",
                "source_revisions": state["source_revisions"],
                "git_comparison_verified": state["git_comparison_verified"],
                "files_sha256": file_hashes}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    common = cell("code", bootstrap(digest), "drive-and-source")
    notebooks = ROOT / "experiments"

    def write_selected(path: Path, cells: list[dict]) -> None:
        if only_notebook is None or path.name == "colab_02_ava_fft_and_fixed_cache.ipynb":
            notebook(path, cells)

    one = notebooks / "colab_01_download_data_weights.ipynb"
    write_selected(one, [
        cell("markdown", """
        # 01 · LIBERO Plus 데이터·AVA-VLA 가중치 → Google Drive
        CPU 런타임에서 시작할 수 있습니다. 저장소의 일반 코드 파일과 폴더를 Drive에 복사하고 공통 프로젝트 루트를 지정하십시오.
        `src/scripts/configs/docs/experiments/third_party/ava_vla`와 루트 문서 구조를 유지하십시오. 소스 목록과 해시는 `experiments/manifests/colab_source.json`으로 검사합니다.
        실행 순서: Drive 연결 → 환경 설치 → 공식 고정 revision 다운로드/해시 검증 → 분할 ZIP 결합/CRC 검사·해제 → episode 준비.
        [데이터 원본](https://huggingface.co/datasets/Sylvest/libero_plus_rlds/tree/fb0c7029b076030d5d57227229e4f7460def1f7c)
        은 **75.5 GB 분할 ZIP 한 개**입니다. `.z01/.z02/.zip`을 각각 해제하지 마십시오.
        [AVA 초기 T0](https://huggingface.co/LiAuto-DSR/avavla-libero-4in1/tree/090bf68d234ea13bc78476236f82458b25b797b8)
        는 backbone·tokenizer·processor·action head·proprio projector·attention generator 전체를 받습니다.
        Drive에는 압축 원본+추출본+query 이미지+가중치를 둘 공간이, `/content`에는 결합 ZIP 약 75.5 GB가 필요합니다.
        코드가 실제 파일 크기로 여유 공간을 검사합니다. Drive 계정 할당량은 Drive에서 별도로 확인하십시오.
        오늘 로컬에서 다운로드/GPU 학습을 실행한 결과가 아닙니다. 각 셀을 실행하면 해당 다운로드·처리가 시작됩니다.
        """, "purpose"), common,
        cell("code", """
        DATA_PY = environment('vlade_data_py311', '3.11', [
            'numpy==1.26.4', 'pillow==10.4.0', 'huggingface_hub==0.36.2',
            'tensorflow==2.15.0', 'tensorflow-datasets==4.9.3', 'tensorflow-metadata==1.15.0'])
        if shutil.which('zip') is None:
            run(['apt-get', 'update'])
            run(['apt-get', 'install', '-y', 'zip'])
        """, "install-data"),
        cell("markdown", "## 공식 파일 다운로드·검증\n중단 후 같은 셀을 실행하면 Hub 다운로드 캐시를 재사용하고 파일 해시를 다시 검사합니다. 대용량 데이터를 로컬 컴퓨터로 거치지 않고 Drive에 저장합니다.", "download-help"),
        cell("code", """
        run([DATA_PY, 'scripts/data/prepare_colab_assets.py', 'data', '--root', ROOT])
        run([DATA_PY, 'scripts/data/prepare_colab_assets.py', 'ava', '--root', ROOT])
        """, "download"),
        cell("code", """
        run([DATA_PY, 'scripts/data/prepare_colab_assets.py', 'extract', '--root', ROOT,
             '--scratch', '/content/libero_zip_scratch'])
        """, "extract"),
        cell("markdown", """
        ## 공통 episode 분할·정규화·관측 준비
        80/10/10은 동일 콘텐츠 그룹의 seed 42 해시로 배정합니다. 실제 episode 비율은 근사치입니다.
        다른 컴퓨터에서 만든 기존 split과 같은 배정이라고 주장하지 않습니다. 기존 manifest를 받으면 비교 후 한 버전으로 고정하십시오.
        Train의 **전체 timestep**에서 exact q01/q99를 계산합니다. Validation/test는 통계에 넣지 않습니다.
        Query 간격은 8이며 두 카메라 원본을 query 시점에만 보관합니다. 마지막 chunk의 padding은 loss에서 제외합니다.
        이 prepared-data 경로는 augmentation 없이 FFT/BC/cache의 입력·정규화를 공유합니다.
        중단된 `.incomplete`는 완료 데이터로 읽지 않습니다. 전처리 중단 시에는 새 output 경로로 다시 준비하십시오.
        """, "prepare-help"),
        cell("code", """
        PREPARED = ROOT / 'data/processed/libero_plus_episode80'
        if (PREPARED / 'COMPLETE').is_file():
            print('준비된 데이터 재사용:', PREPARED)
        else:
            run([DATA_PY, 'scripts/data/prepare_episode_data.py',
                 '--tfds-root', ROOT / 'data/processed/libero_plus_tfds', '--output', PREPARED])
        manifest = json.loads((PREPARED / 'manifest.json').read_text())
        print(json.dumps({key: manifest[key] for key in ['counts', 'split_policy', 'split_sha256',
                                                       'normalization_sha256', 'dataset_revision']}, indent=2))
        print('다음: 02 Teacher 노트북을 A100 런타임에서 실행하십시오.')
        """, "prepare")])
    two = notebooks / "colab_02_ava_fft_and_fixed_cache.ipynb"
    write_selected(two, [
        cell("markdown", """
        # 02 · AVA-VLA 전체 미세조정 → 검증 후 T1 선택 → 순차 캐시
        01을 완료한 뒤 A100 런타임에서 실행하십시오. LoRA를 사용하지 않으며 backbone/head/proprio/attention generator를 학습 대상으로 둡니다.
        A100 40 GB에서 FFT는 OOM이 날 수 있습니다. 먼저 1 step을 실행하고 실제 peak 메모리를 확인하십시오.
        **현재 FFT는 temporal zero-context입니다.** 이후 캐시는 이전 query의 layer -2 action hidden을 전달하지만,
        시간 문맥을 사용한 FFT/TBPTT를 구현하거나 원본 AVA 학습을 재현한 것은 아닙니다. 본 실험 채택 전 검토 항목입니다.
        체크포인트/optimizer 저장은 수십 GB가 될 수 있습니다. `COMPLETE`와 training state가 있는 저장만 재개합니다.
        """, "purpose"), cell("code", bootstrap(digest, isolated_uv=True), "drive-and-source"),
        cell("code", ava_install_source(), "install-ava"),
        cell("code", """
        require_ava_environment()
        PREPARED = ROOT / 'data/processed/libero_plus_episode80'
        from vlade.data.episode_data import EpisodeData
        from vlade.training.run_control import epoch_budget, require_reports, planned_checkpoint_space, pilot_context
        MAIN_EPOCHS = 1  # 최소 1회 전체 train query 순회. 최종 목표 epoch는 검토 후 조정하십시오.
        RUN_MAIN_FFT = False  # 1-step/reload 보고서 확인 후 본 학습을 실행할 때만 True
        SAVES_PER_EPOCH = 4
        BUDGET = epoch_budget(EpisodeData(PREPARED), MAIN_EPOCHS, batch_size=1, accumulation=8)
        RUN_NAME = 'ava-fft-epoch-seed42'  # 설정/코드가 바뀐 기존 실행은 별도 이름을 사용하십시오.
        OUTPUT = ROOT / 'runs/ava_fft' / RUN_NAME
        CONFIGS = ROOT / 'run-configs'
        CONFIGS.mkdir(exist_ok=True)
        config = json.loads((CODE / 'configs/experiments/libero_plus_fft.json').read_text())
        config['fft_defaults'].update(data_root=str(ROOT / 'data/processed/libero_plus_tfds'),
            prepared_data_root=str(PREPARED), image_aug=False, batch_size=1,
            grad_accumulation_steps=8, max_steps=BUDGET['optimizer_steps'],
            save_steps=max(1, (BUDGET['optimizer_steps'] + MAIN_EPOCHS*SAVES_PER_EPOCH - 1)//(MAIN_EPOCHS*SAVES_PER_EPOCH)),
            save_optimizer_state=True, eval_steps=100, max_eval_batches=32, debug_records=1)
        config['training']['colab_budget'] = BUDGET
        print('실제 데이터 기준 본 학습량:', json.dumps(BUDGET, indent=2))
        config['fft_runs']['ava_vla'].update(
            checkpoint=str(ROOT / 'checkpoints/teacher/avavla-libero-4in1'), output_dir=str(OUTPUT))
        MAIN_CONFIG = CONFIGS / (RUN_NAME + '.json')
        if MAIN_CONFIG.exists() and json.loads(MAIN_CONFIG.read_text()) != config:
            raise ValueError('기존 실행 설정이 다릅니다. RUN_NAME을 바꾸십시오.')
        MAIN_CONFIG.write_text(json.dumps(config, indent=2))
        run([AVA_PY, 'scripts/train/run_fft.py', '--model', 'ava_vla', '--config', MAIN_CONFIG, '--preflight'])
        """, "fft-config"),
        cell("markdown", "## 실제 1-step·저장·재로딩 검사\n본 학습과 별도 경로를 사용합니다. 이 1-step 체크포인트를 다른 목표량의 본 실험으로 재개하지 않습니다. 오류가 발생하면 본 학습 셀을 실행하지 마십시오.", "smoke-help"),
        cell("code", """
        require_ava_environment()
        smoke = json.loads(json.dumps(config))
        SMOKE_OUTPUT = ROOT / 'runs/ava_fft' / (RUN_NAME + '-one-step')
        smoke['fft_defaults'].update(max_steps=1, save_steps=1, grad_accumulation_steps=1)
        smoke['fft_runs']['ava_vla']['output_dir'] = str(SMOKE_OUTPUT)
        SMOKE_CONFIG = CONFIGS / (RUN_NAME + '-one-step.json')
        SMOKE_CONFIG.write_text(json.dumps(smoke, indent=2))
        if not (SMOKE_OUTPUT / 'step-0000001/COMPLETE').is_file():
            run([AVA_PY, 'scripts/train/run_fft.py', '--model', 'ava_vla', '--config', SMOKE_CONFIG])
        run([AVA_PY, 'scripts/train/check_fft_checkpoint.py', '--checkpoint', SMOKE_OUTPUT / 'step-0000001',
             '--data', PREPARED, '--config', SMOKE_CONFIG, '--report', SMOKE_OUTPUT / 'reload_report.json'])
        """, "fft-smoke"),
        cell("markdown", "## 본 학습·다음 날 재개\n고정 1000-step 대신 실제 train query 수 × MAIN_EPOCHS에서 optimizer step을 계산합니다. accumulation 경계로 올림한 여분 query 수를 표시합니다. episode 순서는 seed/epoch로 섞고 내부 시간 순서는 유지하며, prepared-data 재개는 query cursor를 복원합니다. GPU kernel의 bitwise 결정성은 별도 검증 사항입니다. RUN_MAIN_FFT 기본 False이므로 모두 실행만으로 본 학습이 시작되지 않습니다.", "main-help"),
        cell("code", """
        require_ava_environment()
        checkpoints = completed_checkpoints(OUTPUT)
        latest = checkpoints[-1] if checkpoints else None
        if not RUN_MAIN_FFT:
            print('본 학습 대기: budget·1-step/reload 결과 확인 후 RUN_MAIN_FFT=True로 실행하십시오.')
        elif latest and int(latest.name.split('-')[1]) >= config['fft_defaults']['max_steps']:
            print('목표 step의 완료 checkpoint가 이미 있습니다:', latest)
        else:
            require_reports([SMOKE_OUTPUT / 'reload_report.json'],
                [pilot_context(SMOKE_CONFIG, SMOKE_OUTPUT / 'step-0000001', PREPARED, 'fft')])
            print('저장 공간 계획:', planned_checkpoint_space(ROOT, SMOKE_OUTPUT / 'step-0000001',
                config['fft_defaults']['max_steps'], config['fft_defaults']['save_steps'],
                int(latest.name.split('-')[1]) if latest else 0))
            command = [AVA_PY, 'scripts/train/run_fft.py', '--model', 'ava_vla', '--config', MAIN_CONFIG]
            if latest:
                command += ['--resume', latest]
            run(command)
        """, "fft-main"),
        cell("markdown", """
        ## Validation 후 T1 선택
        마지막 step을 자동으로 T1으로 부르지 않습니다. 별도 validation/rollout 보고서를 확인한 뒤 아래 세 값을 지정하십시오.
        `validation_report`는 JSON이며 report_type=teacher_validation_comparison, selection_split=validation,
        used_final_test=false, 실제 evaluation_protocol_sha256, prepared manifest의 data_split_sha256,
        candidates 배열을 포함해야 합니다. 각 후보에는 checkpoint/role(T0 또는 T1),
        validation_success_rate/validation_episodes, original_libero_success_rate/original_libero_episodes가 필요합니다.
        실제 측정값과 선택 T1이 없으면 거부합니다. 한 번의 재로딩 보고서는 선정 근거가 아닙니다.
        선택 파일에 전체 bundle SHA256·통계/split SHA256·보고서 SHA256을 기록합니다.
        """, "selection-help"),
        cell("code", """
        T1_CHECKPOINT = ''  # 실제 validation으로 선택한 완료 checkpoint의 Drive 경로
        VALIDATION_REPORT = ''  # 실제 validation/rollout 보고서의 Drive 경로
        SELECTION_REASON = ''  # 선택 지표·비교 checkpoint·이유
        SELECTION = ROOT / 'selections/T1.json'
        if T1_CHECKPOINT and VALIDATION_REPORT and SELECTION_REASON:
            if SELECTION.is_file():
                previous = json.loads(SELECTION.read_text())
                if (Path(previous['checkpoint']).resolve() != Path(T1_CHECKPOINT).resolve() or
                    Path(previous['validation_report']).resolve() != Path(VALIDATION_REPORT).resolve() or
                    previous['selection_reason'] != SELECTION_REASON or
                    hashlib.sha256(Path(VALIDATION_REPORT).read_bytes()).hexdigest() != previous['validation_report_sha256']):
                    raise ValueError('기존 T1 선택과 새 입력이 다릅니다. SELECTION에 새 파일 경로를 지정하십시오.')
            else:
                run([AVA_PY, 'scripts/data/select_teacher.py', '--checkpoint', T1_CHECKPOINT,
                     '--validation-report', VALIDATION_REPORT, '--reason', SELECTION_REASON,
                     '--data', PREPARED, '--output', SELECTION])
        else:
            print('선행 대기: 실제 validation·T1 선정 후 cache를 생성하십시오.')
        """, "select-t1"),
        cell("markdown", """
        ## 고정 시퀀스 cache: 2-episode pilot → 전체 train
        T1은 frozen·비양자화, query 간격 8, episode별 reset과 이전 hidden 전달을 유지합니다.
        재로딩 무결성·샘플 수와 고르게 떨어진 최대 3개 episode의 전체 query 재실행을 검사하고 완료된 episode만 재사용합니다.
        기본은 action cache입니다. `SAVE_HIDDEN=True`이면 `[56,D]` layer -2 feature도 기록합니다.
        큰 hidden JSON은 전체 데이터에서 용량이 매우 커질 수 있으므로 pilot 실측으로 예산을 먼저 확인하십시오.
        `RUN_FULL_CACHE`는 pilot 결과를 검토한 후 켜십시오. action-only cache를 representation KD용으로 해석하지 않습니다.
        """, "cache-help"),
        cell("code", """
        SAVE_HIDDEN = False
        RUN_FULL_CACHE = False
        if SELECTION.is_file():
            suffix = 'hidden' if SAVE_HIDDEN else 'action'
            PILOT = ROOT / 'data/teacher_cache' / ('T1-pilot-' + suffix)
            command = [AVA_PY, 'scripts/data/generate_fixed_cache.py', '--selection', SELECTION,
                       '--data', PREPARED, '--output', PILOT, '--max-episodes', '2']
            run(command + (['--hidden'] if SAVE_HIDDEN else []))
            pilot = json.loads((PILOT / 'cache_manifest.json').read_text())
            total_queries = sum(item['queries'] for item in json.loads((PREPARED / 'manifest.json').read_text())['episodes'] if item['split'] == 'train')
            estimate = pilot['bytes'] * total_queries / pilot['queries']
            print('전체 cache 예상 GiB (pilot 비례 추정):', estimate / 2**30)
            if RUN_FULL_CACHE:
                if shutil.disk_usage(ROOT).free < estimate * 1.2:
                    raise OSError('pilot 기반 cache 예상 용량과 여유 공간이 부족합니다.')
                command = [AVA_PY, 'scripts/data/generate_fixed_cache.py', '--selection', SELECTION,
                           '--data', PREPARED, '--output', ROOT / 'data/teacher_cache' / ('T1-full-' + suffix)]
                run(command + (['--hidden'] if SAVE_HIDDEN else []))
        else:
            print('선행 대기: 선택된 T1이 없습니다. 공개 T0/zero-context debug를 대신 저장하지 않습니다.')
        """, "cache")])
    three = notebooks / "colab_03_qwen_task_only.ipynb"
    write_selected(three, [
        cell("markdown", """
        # 03 · Qwen3.5-0.8B Task-only 기준선
        01의 동일 episode split·정규화·query 간격 8을 사용합니다. Teacher cache 없이 demonstration L1으로 학습합니다.
        최근 2 query × 2 camera, 8차원 proprio, 연속 action `[1,8,7]` 계약입니다. Teacher와 별도 Python 3.13 환경을 만듭니다.
        실행 순서: 환경 → Qwen 가중치 다운로드 → 1-step → 작은 episode overfit → 결과 확인 후 본 학습/재개.
        실제 0.8B 가중치·GPU 학습은 아직 검증하지 않았습니다. 이 노트북을 실행하면서 검증 결과를 기록합니다.
        [고정 Qwen revision](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17).
        Task-only/KD의 공정 비교에는 같은 초기 revision·seed·head·sampler·최종 step·입력 budget을 유지하십시오.
        이 노트북에는 KD loss, CoTinyVLA Plan/Think 생성, 환경 rollout 성공률 평가가 포함되어 있지 않습니다.
        """, "purpose"), common,
        cell("code", """
        STUDENT_PY = environment('vlade_qwen_task_py313', '3.13', [
            'torch==2.12.0', 'torchvision==0.27.0', 'transformers==5.9.0',
            'accelerate', 'pillow', 'numpy', 'sentencepiece'])
        run([STUDENT_PY, '-c', "import torch; from transformers import Qwen3_5Model; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__,torch.cuda.get_device_name(0))"])
        (ROOT / 'student_environment_freeze.txt').write_text(subprocess.check_output(
            [str(UV), 'pip', 'freeze', '--python', str(STUDENT_PY)], text=True))
        """, "install-student"),
        cell("markdown", "## Qwen 초기 가중치 준비\n이 셀을 실행하면 공식 고정 revision의 모델 가중치 약 1.75 GB와 processor 자료를 Drive에 받습니다. 기존 로컬 폴더의 processor 자료만으로는 전체 학습을 실행할 수 없습니다. 오늘 다운로드한 것으로 기록하지 않습니다.", "weights-help"),
        cell("code", """
        QWEN = ROOT / 'checkpoints/students/qwen3_5_0_8b'
        program = "import sys; from pathlib import Path; sys.path.insert(0,'src'); from vlade.data.colab_assets import download; download('Qwen/Qwen3.5-0.8B','2fc06364715b967f1860aea9cf38778875588b17',Path(sys.argv[1]))"
        run([STUDENT_PY, '-c', program, QWEN])
        """, "download-student"),
        cell("code", """
        PREPARED = ROOT / 'data/processed/libero_plus_episode80'
        from vlade.data.episode_data import EpisodeData, write_json
        from vlade.training.run_control import epoch_budget, overfit_report, require_reports, planned_checkpoint_space, pilot_context
        MAIN_EPOCHS = 1
        RUN_MAIN_STUDENT = False
        OVERFIT_MAX_RATIO = .8  # 초기/마지막 loss 중앙값의 비율: 파일럿 감소 검사 기준
        SAVES_PER_EPOCH = 4
        BUDGET = epoch_budget(EpisodeData(PREPARED), MAIN_EPOCHS, batch_size=1, accumulation=8)
        RUN_NAME = 'qwen-task-epoch-seed42'
        CONFIGS = ROOT / 'run-configs'
        CONFIGS.mkdir(exist_ok=True)
        student = json.loads((CODE / 'configs/models/student/qwen3_5_0_8b.json').read_text())['student']
        base = dict(method='task_only', data=str(PREPARED), initial_checkpoint=str(QWEN), student=student,
                    seed=42, batch_size=1, accumulation_steps=8, max_steps=BUDGET['optimizer_steps'],
                    save_steps=max(1,(BUDGET['optimizer_steps']+MAIN_EPOCHS*SAVES_PER_EPOCH-1)//(MAIN_EPOCHS*SAVES_PER_EPOCH)),
                    learning_rate=2e-5, weight_decay=.01, validation_queries=32, eval_steps=100, episode_limit=0,
                    budget=BUDGET,
                    output=str(ROOT / 'runs/student_task' / RUN_NAME))
        def student_config(suffix, **updates):
            settings = {**base, **updates}
            settings['output'] = str(ROOT / 'runs/student_task' / (RUN_NAME + suffix))
            path = CONFIGS / (RUN_NAME + suffix + '.json')
            if path.exists() and json.loads(path.read_text()) != settings:
                raise ValueError('기존 실행 설정 불일치: RUN_NAME을 바꾸십시오.')
            path.write_text(json.dumps(settings, indent=2))
            return path, Path(settings['output'])
        def train_student(config_path, output):
            settings = json.loads(config_path.read_text())
            run([STUDENT_PY, 'scripts/train/run_student_task.py', '--config', config_path, '--preflight'])
            checkpoints = completed_checkpoints(output)
            latest = checkpoints[-1] if checkpoints else None
            if latest and int(latest.name.split('-')[1]) >= settings['max_steps']:
                print('이미 완료된 학습:', latest)
                return
            command = [STUDENT_PY, 'scripts/train/run_student_task.py', '--config', config_path]
            if latest:
                command += ['--resume', latest]
            run(command)
        """, "student-config"),
        cell("code", """
        smoke_config, smoke_output = student_config('-one-step', max_steps=1, save_steps=1,
            accumulation_steps=1, validation_queries=1)
        train_student(smoke_config, smoke_output)
        run([STUDENT_PY, 'scripts/train/check_student_checkpoint.py', '--checkpoint', smoke_output / 'step-0000001',
             '--data', PREPARED, '--config', smoke_config, '--report', smoke_output / 'reload_report.json'])
        """, "student-smoke"),
        cell("markdown", "## 작은 episode overfit 검사\n같은 train episode를 반복하며 처음/마지막 5개 loss 중앙값의 감소 비율을 자동 검사합니다. 이 파일럿 기준은 완전 수렴이나 rollout 성공률을 의미하지 않습니다. 실패 보고서가 있으면 본 학습을 거부합니다. 실제 가중치의 입력 모달리티 영향은 별도 확인 대상입니다.", "overfit-help"),
        cell("code", """
        overfit_config, overfit_output = student_config('-overfit', max_steps=50, save_steps=50,
            accumulation_steps=1, episode_limit=1, validation_queries=4)
        train_student(overfit_config, overfit_output)
        by_step = {}
        for log in sorted(overfit_output.glob('metrics-*.jsonl'), key=lambda path: path.stat().st_mtime):
            for line in log.read_text().splitlines():
                row = json.loads(line)
                by_step[row['step']] = row
        report = overfit_report(list(by_step.values()), max_ratio=OVERFIT_MAX_RATIO)
        overfit_steps = json.loads(overfit_config.read_text())['max_steps']
        report['context'] = pilot_context(overfit_config, overfit_output / f'step-{overfit_steps:07d}',
            PREPARED, 'student', policy={'max_ratio': OVERFIT_MAX_RATIO, 'window': 5})
        write_json(overfit_output / 'overfit_report.json', report)
        print('파일럿 loss 감소 검사:', report)
        import matplotlib.pyplot as plt
        for log in sorted(overfit_output.glob('metrics-*.jsonl')):
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            plt.plot([row['step'] for row in rows], [row['loss'] for row in rows], label=log.stem)
        plt.xlabel('Optimizer step'); plt.ylabel('Train L1'); plt.legend(); plt.show()
        """, "student-overfit"),
        cell("code", """
        MAIN_CONFIG, MAIN_OUTPUT = student_config('')
        if RUN_MAIN_STUDENT:
            overfit_steps = json.loads(overfit_config.read_text())['max_steps']
            require_reports([smoke_output / 'reload_report.json', overfit_output / 'overfit_report.json'],
                [pilot_context(smoke_config, smoke_output / 'step-0000001', PREPARED, 'student'),
                 pilot_context(overfit_config, overfit_output / f'step-{overfit_steps:07d}', PREPARED, 'student',
                    policy={'max_ratio': OVERFIT_MAX_RATIO, 'window': 5})])
            checkpoints = completed_checkpoints(MAIN_OUTPUT)
            print('저장 공간 계획:', planned_checkpoint_space(ROOT, smoke_output / 'step-0000001',
                base['max_steps'], base['save_steps'], int(checkpoints[-1].name.split('-')[1]) if checkpoints else 0))
            train_student(MAIN_CONFIG, MAIN_OUTPUT)
        else:
            print('본 학습 대기: budget·reload·overfit 결과 확인 후 RUN_MAIN_STUDENT=True로 실행하십시오.')
        reports = [json.loads(path.read_text()) for path in sorted(MAIN_OUTPUT.glob('run-*.json'), key=lambda path: path.stat().st_mtime_ns)]
        print(json.dumps(reports[-1] if reports else {}, indent=2, ensure_ascii=False))
        print('Validation L1은 rollout 성공률이 아닙니다. 환경 평가와 최종 기준 성능 기록은 별도로 진행하십시오.')
        """, "student-main")])
    return ([two] if only_notebook == 2 else [one, two, three]) + [manifest_path]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate verified Colab source snapshots")
    parser.add_argument("--source-revision", help="Pinned source commit, retained across derived-output commits")
    parser.add_argument("--only-notebook", type=int, choices=(2,), help="Update only notebook 02 and the manifest")
    args = parser.parse_args(argv)
    for path in generate(source_revision=args.source_revision, only_notebook=args.only_notebook):
        print(path)


if __name__ == "__main__":
    main()
