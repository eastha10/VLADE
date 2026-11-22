"""Self-contained safe-tensor deployment artifacts with strict topology loading."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import torch
from .linear import PackedLinear


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def qwen_architecture(model: torch.nn.Module) -> dict:
    return {'kind': 'qwen_vla_schema2', 'student': asdict(model.student_config),
            'backbone': model.backbone.config.to_dict()}


def build_qwen(spec: dict) -> torch.nn.Module:
    from transformers import Qwen3_5Config, Qwen3_5Model
    from vlade.models.students.qwen_vla import QwenVLA
    from vlade.models.students.configuration_qwen_vla import QwenVLAConfig
    if spec.get('kind') != 'qwen_vla_schema2':
        raise ValueError('Unknown packed model architecture')
    student = QwenVLAConfig(**spec['student'])
    student.validate()
    config = Qwen3_5Config.from_dict(spec['backbone'])
    config._attn_implementation = 'eager'
    return QwenVLA(Qwen3_5Model(config), student).to(dtype=torch.bfloat16)


def save(model: torch.nn.Module, destination: Path, *, architecture: dict, provenance: dict, report: dict,
         processor_source: Path | None = None, normalization: dict | None = None) -> str:
    """Save all unquantized state AND packed weights; no original weight file needed.

    Return the manifest SHA for the exporting process's immediate round-trip check.
    External consumers must load through the pinned HF input resolver below.
    """
    from safetensors.torch import save_file
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError('Packed artifact output already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / ('.packed-' + uuid.uuid4().hex)
    stage.mkdir()
    specs = {}
    for name, module in model.named_modules():
        if isinstance(module, PackedLinear):
            module.validate()
            specs[name] = {'in_features': module.in_features, 'out_features': module.out_features,
                           'group_size': module.group_size}
    if not specs:
        raise ValueError('No packed modules to export')
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}
    save_file(tensors, str(stage / 'model.safetensors'))
    files = ['model.safetensors']
    if processor_source is not None:
        (stage/'processor').mkdir()
        for name in ('chat_template.jinja', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json'):
            source = processor_source/name
            if not source.is_file() or source.is_symlink():
                raise ValueError('Saved processor file is missing or unsafe: ' + name)
            shutil.copyfile(source, stage/'processor'/name)
            files.append('processor/'+name)
    manifest = {'schema_version': 1, 'format': 'vlade_packed_vla', 'architecture': architecture,
                'modules': specs, 'provenance': provenance, 'report': report,
                'normalization': normalization,
                'files': {name: {'bytes': (stage/name).stat().st_size, 'sha256': digest(stage/name)} for name in files}}
    (stage / 'packed.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    result = digest(stage / 'packed.json')
    (stage / 'COMPLETE').write_text('ok\n', encoding='ascii')
    from vlade.utils.filesystem import publish_directory
    publish_directory(stage, destination)
    return result


def _load_owned(root: Path, manifest_sha256: str, *, builder=build_qwen, backend: str = 'auto'):
    """Internal round-trip of an artifact just produced by this exporter."""
    from safetensors.torch import load_file
    root = root.resolve(strict=True)
    if not (root/'COMPLETE').is_file() or digest(root/'packed.json') != manifest_sha256:
        raise ValueError('Packed artifact completion/manifest identity differs')
    manifest = json.loads((root/'packed.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('format') != 'vlade_packed_vla':
        raise ValueError('Unsupported packed artifact format')
    allowed = {'model.safetensors', 'processor/chat_template.jinja', 'processor/processor_config.json',
               'processor/tokenizer.json', 'processor/tokenizer_config.json'}
    if 'model.safetensors' not in manifest['files'] or not set(manifest['files']) <= allowed:
        raise ValueError('Unexpected packed artifact payload list')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'packed.json', 'COMPLETE'}:
        raise ValueError('Packed artifact file set differs')
    for name, fingerprint in manifest['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or path.stat().st_size != fingerprint['bytes'] or digest(path) != fingerprint['sha256']:
            raise ValueError('Packed payload integrity mismatch: ' + name)
    state = load_file(str(root/'model.safetensors'), device='cpu')
    model = builder(manifest['architecture'])
    replacements = {}
    for name, spec in manifest['modules'].items():
        original = model.get_submodule(name)
        if not isinstance(original, torch.nn.Linear) or (original.in_features, original.out_features) != (spec['in_features'], spec['out_features']):
            raise ValueError('Packed artifact module topology differs: ' + name)
        module = PackedLinear(**spec, backend=backend)
        for key in module._buffers:
            module._buffers[key] = state[name+'.'+key]
        module.validate()
        replacements[name] = module
    from .calibration import replace
    replace(model, replacements)
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False).eval()
    return model, manifest


def load(value: str | Path, *, registry: Path | None = None, backend: str = 'auto') -> tuple[torch.nn.Module, dict]:
    """Load external deployment inputs only from an immutable, verified HF record."""
    from vlade.storage.assets import resolve_input
    root = resolve_input(value, registry=registry)
    return _load_owned(root, digest(root/'packed.json'), backend=backend)
