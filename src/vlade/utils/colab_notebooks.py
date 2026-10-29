"""Generate Colab notebooks that load verified ordinary project files from Drive."""
from __future__ import annotations
import hashlib
import inspect
import json
import subprocess
import textwrap
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANONICAL_NORMALIZATION = Path("experiments/manifests/libero_plus_rlds/normalization_train_d_seed42.json")


def source_sha256(path: Path) -> str:
    """Hash repository text as stored by Git, independent of Windows checkout EOLs."""
    payload = path.read_bytes()
    if path.relative_to(ROOT) == CANONICAL_NORMALIZATION:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


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


def bootstrap(digest: str) -> str:
    return '''
from pathlib import Path
from google.colab import drive
import hashlib, json, os, shutil, subprocess, sys
drive.mount('/content/real_drive')
ROOT = Path('/content/real_drive/MyDrive/VLADE')  # 네 노트북에서 동일한 저장 폴더를 지정하십시오.
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


def generate() -> list[Path]:
    paths = []
    manifest_path = ROOT / "experiments/manifests/colab_source.json"
    canonical_manifest = ROOT / "experiments/manifests/libero_plus_rlds/canonical_d_seed42.jsonl.gz"
    for folder in ("src", "scripts", "configs", "docs", "experiments/baselines", "experiments/manifests",
                   "third_party/ava_vla", "third_party/openvla_oft"):
        for path in (ROOT / folder).rglob("*"):
            if path == manifest_path or not path.is_file() or any(part in {".git", "__pycache__", ".pytest_cache", "wandb", "node_modules"} for part in path.parts):
                continue
            if (path.suffix.lower() in {".py", ".json", ".toml", ".txt", ".md", ".yaml", ".yml"} or
                    path.name in {"LICENSE", "NOTICE"} or path == canonical_manifest):
                paths.append(path)
    paths.extend(ROOT / name for name in ("README.md", "STRUCTURE.md", "AGENTS.md"))
    manifest = {"schema_version": 1, "built_for": "2026-10-28_kanban_colab_preparation",
                "contains_uncommitted_work": True,
                "revision_note": "Submodule reference may be the parent gitlink when metadata is inaccessible; exact ordinary source files are identified by files_sha256.",
                "source_revisions": {name: revision(ROOT if name == "." else ROOT / name)
                                     for name in (".", "third_party/ava_vla", "third_party/openvla_oft")},
                "files_sha256": {p.relative_to(ROOT).as_posix(): source_sha256(p)
                                  for p in sorted(set(paths))}}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    common = cell("code", bootstrap(digest), "drive-and-source")
    notebooks = ROOT / "experiments"
    one = notebooks / "colab_01_download_data_weights.ipynb"
    notebook(one, [
        cell("markdown", """
        # 01 · LIBERO Plus 데이터·AVA/OFT 가중치 → Google Drive
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
        run([DATA_PY, 'scripts/data/prepare_colab_assets.py', 'oft', '--root', ROOT])
        """, "download"),
        cell("code", """
        run([DATA_PY, 'scripts/data/prepare_colab_assets.py', 'extract', '--root', ROOT,
             '--scratch', '/content/libero_zip_scratch'])
        """, "extract"),
        cell("markdown", """
        ## 공통 episode 분할·정규화·관측 준비
        확정 정본은 D/content-group 80/10/10, seed 42입니다. Train/Validation/Test는
        11,477/1,428/1,442 Episode이고 Train 내부 Calibration은 728 Episode입니다.
        준비 과정은 각 trajectory의 exact action+state group을 canonical manifest와 대조하며 임의 재분할을 금지합니다.
        Train의 **전체 timestep**에서 exact q01/q99를 계산합니다. Validation/test는 통계에 넣지 않습니다.
        Query 간격은 8이며 두 카메라 원본을 query 시점에만 보관합니다. 마지막 chunk의 padding은 loss에서 제외합니다.
        이 prepared-data 경로는 augmentation 없이 FFT/BC/cache의 입력·정규화를 공유합니다.
        TFDS 1,024 shard는 Drive에서 순차 읽고 Episode 출력은 `/content/vlade-preprocess` 로컬 SSD에 기록합니다.
        100 Episode마다 동기식 tar archive를 Drive에 SHA-256 검증 후 게시합니다. 재시작 시 검증된 archive와
        기존 Drive `.incomplete`를 로컬로 복구한 뒤 `--resume`이 Episode 내용·checkpoint를 다시 검증합니다.
        """, "prepare-help"),
        cell("code", """
        PREPARED = Path('/content/vlade-preprocess/libero_plus_episode80')
        BACKUPS = ROOT / 'data/processed/libero_plus_episode80.backups'
        LEGACY_INCOMPLETE = ROOT / 'data/processed/libero_plus_episode80.incomplete'
        if (PREPARED / 'COMPLETE').is_file():
            print('준비된 데이터 재사용:', PREPARED)
        else:
            command = [DATA_PY, 'scripts/data/prepare_episode_data.py',
                       '--tfds-root', ROOT / 'data/processed/libero_plus_tfds', '--output', PREPARED,
                       '--resume', '--backup-dir', BACKUPS, '--backup-every', '100',
                       '--restore-backups', '--min-free-gib', '20',
                       '--canonical-manifest', CODE / 'experiments/manifests/libero_plus_rlds/canonical_d_seed42.jsonl.gz',
                       '--canonical-sha256', '487d2be1039ba6a4facc3ba44785ba5159245dfe53462a1d5a41ba8230446e80',
                       '--canonical-normalization', CODE / 'experiments/manifests/libero_plus_rlds/normalization_train_d_seed42.json',
                       '--canonical-normalization-sha256', 'eac95bcc81b6182b90a2bd030096df6e3e408cb88923d0a1d57dc6c81e297c67']
            if LEGACY_INCOMPLETE.is_dir():
                command += ['--legacy-incomplete', LEGACY_INCOMPLETE]
            run(command)
        manifest = json.loads((PREPARED / 'manifest.json').read_text())
        print(json.dumps({key: manifest[key] for key in ['counts', 'calibration_episodes', 'split_policy',
                                                       'canonical_manifest_sha256', 'canonical_normalization_sha256',
                                                       'split_sha256', 'normalization_sha256', 'dataset_revision']}, indent=2))
        print('다음: 02 Teacher 노트북을 A100 런타임에서 실행하십시오.')
        """, "prepare")])
    two = notebooks / "colab_02_ava_fft_and_fixed_cache.ipynb"
    notebook(two, [
        cell("markdown", """
        # 02 · AVA-VLA 전체 미세조정 → 검증 후 T1 선택 → 순차 캐시
        01을 완료한 뒤 A100 런타임에서 실행하십시오. LoRA를 사용하지 않으며 backbone/head/proprio/attention generator를 학습 대상으로 둡니다.
        A100 40 GB에서 FFT는 OOM이 날 수 있습니다. 먼저 1 step을 실행하고 실제 peak 메모리를 확인하십시오.
        **현재 FFT는 temporal zero-context입니다.** 이후 캐시는 이전 query의 layer -2 action hidden을 전달하지만,
        시간 문맥을 사용한 FFT/TBPTT를 구현하거나 원본 AVA 학습을 재현한 것은 아닙니다. 본 실험 채택 전 검토 항목입니다.
        체크포인트/optimizer 저장은 수십 GB가 될 수 있습니다. `COMPLETE`와 training state가 있는 저장만 재개합니다.
        """, "purpose"), common,
        cell("code", """
        import tomllib
        upstream = CODE / 'third_party/ava_vla'
        dependencies = tomllib.loads((upstream / 'pyproject.toml').read_text())['project']['dependencies']
        dependencies = [dep for dep in dependencies if not dep.startswith(('transformers @', 'dlimp @', 'huggingface_hub', 'accelerate'))]
        dependencies += [
            'transformers @ git+https://github.com/LiAuto-DSR/transformers-avavla.git@a03eee5da1794c45f5a911b60a9a4545e6e0ed1f',
            'dlimp @ git+https://github.com/moojink/dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46',
            'numpy==1.26.4', 'tensorflow-metadata==1.15.0', 'bitsandbytes==0.43.3',
            'huggingface_hub==0.36.2', 'accelerate==1.15.0']
        AVA_PY = environment('vlade_ava_fft_py311', '3.11', dependencies)
        run([UV, 'pip', 'install', '--python', AVA_PY, '--no-deps', '-e', upstream])
        run([AVA_PY, '-c', "import torch,transformers,tensorflow_datasets; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory/2**30)"])
        freeze = subprocess.check_output([str(UV), 'pip', 'freeze', '--python', str(AVA_PY)], text=True)
        (ROOT / 'ava_environment_freeze.txt').write_text(freeze)
        """, "install-ava"),
        cell("code", """
        PREPARED = Path('/content/vlade-preprocess/libero_plus_episode80')
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
    notebook(three, [
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
        PREPARED = Path('/content/vlade-preprocess/libero_plus_episode80')
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
    four = notebooks / "colab_04_openvla_oft_fft.ipynb"
    notebook(four, [
        cell("markdown", """
        # 04 · OpenVLA-OFT 독립 환경 1-step FFT·재로딩
        01에서 준비한 동일 canonical split/normalization을 사용합니다. AVA 환경과 Transformers fork를
        공유하지 않습니다. LoRA가 아닌 전체 파라미터 FFT이며, 본격 학습 전 1 optimizer step과 저장·재로딩만 검사합니다.
        A100 40 GB에서도 OOM 가능성이 있으므로 실제 peak VRAM과 Drive 여유 공간을 먼저 확인하십시오.
        """, "purpose"), common,
        cell("code", """
        import tomllib
        upstream = CODE / 'third_party/openvla_oft'
        dependencies = tomllib.loads((upstream / 'pyproject.toml').read_text())['project']['dependencies']
        dependencies = [dep for dep in dependencies if not dep.startswith(('transformers @', 'dlimp @', 'huggingface_hub', 'accelerate'))]
        dependencies += [
            'transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab',
            'dlimp @ git+https://github.com/moojink/dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46',
            'numpy==1.26.4', 'tensorflow-metadata==1.15.0', 'bitsandbytes==0.43.3',
            'huggingface_hub==0.36.2', 'accelerate==1.15.0']
        OFT_PY = environment('vlade_oft_fft_py311', '3.11', dependencies)
        run([UV, 'pip', 'install', '--python', OFT_PY, '--no-deps', '-e', upstream])
        run([OFT_PY, '-c', "import torch,transformers,tensorflow_datasets; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))"])
        (ROOT / 'oft_environment_freeze.txt').write_text(subprocess.check_output(
            [str(UV), 'pip', 'freeze', '--python', str(OFT_PY)], text=True))
        """, "install-oft"),
        cell("code", """
        PREPARED = Path('/content/vlade-preprocess/libero_plus_episode80')
        from vlade.data.episode_data import EpisodeData
        data = EpisodeData(PREPARED)
        assert data.manifest['counts'] == {'train': 11477, 'validation': 1428, 'test': 1442}
        assert data.manifest['canonical_manifest_sha256'] == '487d2be1039ba6a4facc3ba44785ba5159245dfe53462a1d5a41ba8230446e80'
        config = json.loads((CODE / 'configs/experiments/libero_plus_fft.json').read_text())
        output = ROOT / 'runs/openvla_oft_fft/oft-one-step-seed42'
        config['fft_defaults'].update(
            data_root=str(ROOT / 'data/processed/libero_plus_tfds'), prepared_data_root=str(PREPARED),
            image_aug=False, batch_size=1, grad_accumulation_steps=1, max_steps=1, save_steps=1,
            save_optimizer_state=True, eval_steps=0)
        config['fft_runs']['openvla_oft'].update(
            checkpoint=str(ROOT / 'checkpoints/baselines/openvla_oft/openvla-7b-oft-finetuned-libero-spatial-object-goal-10'),
            output_dir=str(output))
        configs = ROOT / 'run-configs'; configs.mkdir(exist_ok=True)
        config_path = configs / 'openvla-oft-one-step-seed42.json'
        config_path.write_text(json.dumps(config, indent=2))
        run([OFT_PY, 'scripts/train/run_fft.py', '--model', 'openvla_oft', '--config', config_path, '--preflight'])
        if not (output / 'step-0000001/COMPLETE').is_file():
            run([OFT_PY, 'scripts/train/run_fft.py', '--model', 'openvla_oft', '--config', config_path])
        run([OFT_PY, 'scripts/train/check_fft_checkpoint.py', '--checkpoint', output / 'step-0000001',
             '--data', PREPARED, '--config', config_path, '--report', output / 'reload_report.json'])
        """, "oft-smoke")])
    five = notebooks / "colab_05_ava_a100_one_step_reload.ipynb"
    notebook(five, [
        cell("markdown", """
        # 05 · AVA-VLA A100 1-step FFT · checkpoint 별도 프로세스 재로딩
        이 노트북은 Canonical D/seed 42의 **Train Episode 1개만** 원본 TFDS에서 준비하고
        AVA-VLA full-parameter forward/backward/update 1-step, 전체 checkpoint 저장과 별도 Python
        프로세스 재로딩을 검사합니다. 14,347 Episode 전체 전처리·복사나 본 학습은 실행하지 않습니다.

        기본 Colab Python 3.13은 사용하지 않고 `uv`로 Python 3.11 환경을 만듭니다.
        `bitsandbytes`를 포함한 AVA 고정 의존성을 설치하므로 네트워크 설치 시간은 필요하지만
        모델·데이터를 새로 다운로드하지 않습니다. 입력은 Drive의 기존 TFDS와 AVA 가중치입니다.

        **AVA temporal 한계:** FFT 샘플에는 이전 query의 action feature가 없어
        `temporal_context=None`인 learned zero-context 경로만 학습합니다. 따라서 원본 AVA의
        순차 temporal-context 학습/TBPTT 또는 rollout 성능 검증과 동등하지 않습니다.
        """, "purpose"), common,
        cell("code", """
        import tomllib
        upstream = CODE / 'third_party/ava_vla'
        dependencies = tomllib.loads((upstream / 'pyproject.toml').read_text())['project']['dependencies']
        dependencies = [dep for dep in dependencies if not dep.startswith(
            ('transformers @', 'dlimp @', 'huggingface_hub', 'accelerate'))]
        dependencies += [
            'transformers @ git+https://github.com/LiAuto-DSR/transformers-avavla.git@a03eee5da1794c45f5a911b60a9a4545e6e0ed1f',
            'dlimp @ git+https://github.com/moojink/dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46',
            'numpy==1.26.4', 'tensorflow-metadata==1.15.0', 'bitsandbytes==0.43.3',
            'huggingface_hub==0.36.2', 'accelerate==1.15.0']
        AVA_PY = environment('vlade_ava_fft_py311', '3.11', dependencies)
        run([UV, 'pip', 'install', '--python', AVA_PY, '--no-deps', '-e', upstream])
        run([AVA_PY, '-c', "import bitsandbytes,torch,transformers,tensorflow_datasets; "
            "assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); "
            "p=torch.cuda.get_device_properties(0); "
            "print(torch.__version__,transformers.__version__,bitsandbytes.__version__,p.name,p.total_memory/2**30)"])
        run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])
        freeze = subprocess.check_output([str(UV), 'pip', 'freeze', '--python', str(AVA_PY)], text=True)
        (ROOT / 'ava_environment_freeze.txt').write_text(freeze)
        """, "install-ava"),
        cell("code", """
        TFDS_ROOT = ROOT / 'data/processed/libero_plus_tfds'
        PREPARED = Path('/content/vlade-preprocess/libero_plus_canonical_train1')
        if not (PREPARED / 'COMPLETE').is_file():
            run([AVA_PY, 'scripts/data/prepare_single_canonical_episode.py',
                 '--tfds-root', TFDS_ROOT, '--output', PREPARED,
                 '--canonical-manifest', CODE / 'experiments/manifests/libero_plus_rlds/canonical_d_seed42.jsonl.gz',
                 '--canonical-sha256', '487d2be1039ba6a4facc3ba44785ba5159245dfe53462a1d5a41ba8230446e80',
                 '--canonical-normalization', CODE / 'experiments/manifests/libero_plus_rlds/normalization_train_d_seed42.json',
                 '--canonical-normalization-sha256', 'eac95bcc81b6182b90a2bd030096df6e3e408cb88923d0a1d57dc6c81e297c67'])
        manifest = json.loads((PREPARED / 'manifest.json').read_text())
        assert manifest['dataset_scope'] == 'canonical_train_episode_smoke'
        assert manifest['counts'] == {'train': 1}
        assert manifest['canonical_manifest_sha256'] == '487d2be1039ba6a4facc3ba44785ba5159245dfe53462a1d5a41ba8230446e80'
        assert manifest['canonical_normalization_sha256'] == 'eac95bcc81b6182b90a2bd030096df6e3e408cb88923d0a1d57dc6c81e297c67'
        print(json.dumps({key: manifest[key] for key in (
            'dataset_scope', 'counts', 'selected_source_episode_index',
            'canonical_manifest_sha256', 'canonical_normalization_sha256')}, indent=2))
        print('Episode:', manifest['episodes'][0]['episode_id'],
              'queries:', manifest['episodes'][0]['queries'])
        """, "prepare-one-episode"),
        cell("code", """
        OUTPUT = ROOT / 'runs/ava_fft/ava-a100-one-step-seed42'
        CONFIGS = ROOT / 'run-configs'; CONFIGS.mkdir(exist_ok=True)
        CONFIG = CONFIGS / 'ava-a100-one-step-seed42.json'
        config = json.loads((CODE / 'configs/experiments/libero_plus_fft.json').read_text())
        config['fft_defaults'].update(
            data_root=str(TFDS_ROOT), prepared_data_root=str(PREPARED), image_aug=False,
            batch_size=1, grad_accumulation_steps=1, max_steps=1, save_steps=1,
            save_optimizer_state=True, eval_steps=0, debug_records=0)
        config['fft_runs']['ava_vla'].update(
            checkpoint=str(ROOT / 'checkpoints/teacher/avavla-libero-4in1'),
            output_dir=str(OUTPUT))
        if CONFIG.exists() and json.loads(CONFIG.read_text()) != config:
            raise ValueError('기존 1-step 설정이 다릅니다. OUTPUT/CONFIG 이름을 바꾸십시오.')
        CONFIG.write_text(json.dumps(config, indent=2))
        run([AVA_PY, 'scripts/train/run_fft.py', '--model', 'ava_vla',
             '--config', CONFIG, '--preflight'])
        """, "configure-one-step"),
        cell("code", """
        CHECKPOINT = OUTPUT / 'step-0000001'
        if not (CHECKPOINT / 'COMPLETE').is_file():
            run([AVA_PY, 'scripts/train/run_fft.py', '--model', 'ava_vla', '--config', CONFIG])
        metrics = sorted(OUTPUT.glob('metrics-*.jsonl'))
        runs = sorted(OUTPUT.glob('run-*.json'))
        if len(metrics) != 1 or len(runs) != 1:
            raise ValueError('격리된 새 OUTPUT에는 metrics/run manifest가 각각 하나여야 합니다.')
        run([AVA_PY, 'scripts/train/check_fft_checkpoint.py',
             '--checkpoint', CHECKPOINT, '--data', PREPARED, '--config', CONFIG,
             '--metrics', metrics[0], '--run-manifest', runs[0],
             '--report', OUTPUT / 'reload_report.json'])
        """, "run-and-reload"),
        cell("code", """
        report = json.loads((OUTPUT / 'reload_report.json').read_text())
        summary = {
            'status': report['status'], 'gpu': report['gpu'],
            'step_time_seconds': report['training_metrics']['step_time_seconds'],
            'training_peak_allocated_gib': report['training_metrics']['peak_allocated_bytes'] / 2**30,
            'training_peak_reserved_gib': report['training_metrics']['peak_reserved_bytes'] / 2**30,
            'training_peak_process_ram_gib': report['training_metrics']['peak_process_rss_bytes'] / 2**30,
            'reload_peak_allocated_gib': report['reload_peak_allocated_bytes'] / 2**30,
            'reload_peak_process_ram_gib': report['reload_peak_process_rss_bytes'] / 2**30,
            'checkpoint_gib': report['checkpoint_bytes'] / 2**30,
            'loss': report['training_metrics']['loss'],
            'grad_norm': report['training_metrics']['grad_norm'],
            'gradient_diagnostics': report['gradient_diagnostics'],
            'action_finite': report['action_finite'],
            'training_finite_checks': report['training_finite_checks'],
            'scope': report['scope'],
            'ava_temporal_training': 'zero_context',
        }
        print(json.dumps(summary, indent=2))
        print('원본 보고서:', OUTPUT / 'reload_report.json')
        """, "show-report")])
    return [one, two, three, four, five, manifest_path]


def main() -> None:
    for path in generate():
        print(path)


if __name__ == "__main__":
    main()
