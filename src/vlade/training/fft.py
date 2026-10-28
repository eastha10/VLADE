"""Full-parameter LIBERO+ fine-tuning for the pinned OpenVLA-OFT/AVA-VLA sources.

This module deliberately does not modify either upstream submodule. Each model
must run in its own Python environment because AVA uses a transformers fork.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCES = {"openvla_oft": "third_party/openvla_oft", "ava_vla": "third_party/ava_vla"}


@dataclass(frozen=True)
class FFTConfig:
    model: str
    checkpoint: Path
    data_root: Path
    output_dir: Path
    dataset_name: str = "libero_plus_mixdata"
    max_steps: int = 1000
    batch_size: int = 1
    grad_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    save_steps: int = 500
    seed: int = 42
    shuffle_buffer_size: int = 1000
    num_images_in_input: int = 2
    optimizer: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    image_aug: bool = True
    save_optimizer_state: bool = False
    config_path: Path | None = None
    max_grad_norm: float = 1.0
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    log_steps: int = 10
    eval_steps: int = 0
    max_eval_batches: int = 10
    device: str = "cuda:0"
    resume: Path | None = None
    prepared_data_root: Path | None = None
    canonical_split_manifest: Path | None = None
    canonical_split_sha256: str | None = None
    canonical_normalization: Path | None = None
    canonical_normalization_sha256: str | None = None
    debug_records: int = 0

    @classmethod
    def from_json(cls, path: Path, model: str) -> "FFTConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if model not in SOURCES:
            raise ValueError(f"Unknown model: {model}")
        settings = raw["fft_runs"][model]
        common = raw["fft_defaults"]
        merged = {**common, **settings, "model": model, "config_path": path.resolve()}
        for key in ("checkpoint", "data_root", "output_dir"):
            value = Path(merged[key]).expanduser()
            merged[key] = value if value.is_absolute() else REPO_ROOT / value
        for key in ("prepared_data_root", "canonical_split_manifest", "canonical_normalization"):
            if merged.get(key) is not None:
                value = Path(merged[key]).expanduser()
                merged[key] = value if value.is_absolute() else REPO_ROOT / value
        cfg = cls(**merged)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.model not in SOURCES:
            raise ValueError(f"Unknown model: {self.model}")
        if min(self.max_steps, self.batch_size, self.grad_accumulation_steps, self.save_steps) < 1:
            raise ValueError("Steps, batch size, accumulation and save interval must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer hyperparameters")
        if self.optimizer not in {"paged_adamw_8bit", "adamw"}:
            raise ValueError("optimizer must be paged_adamw_8bit or adamw")
        if self.num_images_in_input != 2:
            raise ValueError("These LIBERO checkpoints expect primary and wrist images")
        if not all(math.isfinite(value) for value in (self.learning_rate, self.weight_decay, self.max_grad_norm)):
            raise ValueError("Optimizer values must be finite")
        if self.max_grad_norm <= 0 or self.log_steps < 1 or self.eval_steps < 0 or self.max_eval_batches < 1:
            raise ValueError("Invalid clipping, logging or evaluation settings")
        if self.lr_schedule not in {"constant", "cosine"} or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("Invalid LR schedule or warmup_steps")
        if self.debug_records < 0:
            raise ValueError("debug_records must be nonnegative")
        canonical = (self.canonical_split_manifest, self.canonical_split_sha256,
                     self.canonical_normalization, self.canonical_normalization_sha256)
        if any(value is not None for value in canonical) and not all(value is not None for value in canonical):
            raise ValueError("Canonical split manifest and normalization paths/hashes must be configured together")
        if self.dataset_name == "libero_plus_mixdata" and (
                self.prepared_data_root is None or not all(value is not None for value in canonical)):
            raise ValueError(
                "LIBERO+ FFT requires canonical prepared data plus the sealed split and normalization identities"
            )
        for value in (self.canonical_split_sha256, self.canonical_normalization_sha256):
            if value is not None and (len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value)):
                raise ValueError("Canonical SHA-256 values must contain 64 hexadecimal characters")
        if not self.device.startswith("cuda"):
            raise ValueError("The pinned FFT adapters require a BF16 CUDA device")


def _tfds_info_files(data_root: Path, dataset_name: str) -> list[Path]:
    base = data_root / dataset_name
    return [path for path in [base / "dataset_info.json", *base.glob("*/dataset_info.json")] if path.is_file()]


def register_libero_plus_dataset(dataset_name: str) -> None:
    """Add LIBERO+ to upstream Open-X and resolve its disk-only TFDS builder."""
    if dataset_name != "libero_plus_mixdata":
        return
    import tensorflow_datasets as tfds
    from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
    from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS

    template = "libero_4_task_suites_no_noops"
    OXE_DATASET_CONFIGS[dataset_name] = OXE_DATASET_CONFIGS[template].copy()
    OXE_STANDARDIZATION_TRANSFORMS[dataset_name] = OXE_STANDARDIZATION_TRANSFORMS[template]
    if getattr(tfds.builder, "_vlade_libero_plus", False):
        return
    original_builder = tfds.builder

    def disk_builder(name: str, *args: Any, **kwargs: Any) -> Any:
        if name != dataset_name:
            return original_builder(name, *args, **kwargs)
        data_dir = kwargs.get("data_dir", args[0] if args else None)
        if data_dir is None:
            raise ValueError("LIBERO+ TFDS builder needs data_dir")
        infos = _tfds_info_files(Path(data_dir), dataset_name)
        if len(infos) != 1:
            raise FileNotFoundError(f"Expected one dataset_info.json under {Path(data_dir) / dataset_name}")
        return tfds.builder_from_directory(str(infos[0].parent))

    disk_builder._vlade_libero_plus = True  # type: ignore[attr-defined]
    tfds.builder = disk_builder


def find_component(checkpoint: Path, name: str) -> Path:
    matches = sorted(checkpoint.glob(f"{name}--*_checkpoint.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name} checkpoint in {checkpoint}, found {len(matches)}")
    return matches[0]


def load_component(module: Any, checkpoint: Path, name: str) -> None:
    import torch

    state = torch.load(find_component(checkpoint, name), map_location="cpu", weights_only=True)
    module.load_state_dict({key.removeprefix("module."): val for key, val in state.items()}, strict=True)


def preflight(cfg: FFTConfig) -> None:
    cfg.validate()
    source = REPO_ROOT / SOURCES[cfg.model]
    if not (source / "prismatic").is_dir():
        raise FileNotFoundError(f"Missing upstream submodule: {source}")
    if cfg.resume is not None:
        validate_resume(cfg)
        cfg = replace(cfg, checkpoint=cfg.resume)
    if not (cfg.checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {cfg.checkpoint / 'config.json'}")
    index = cfg.checkpoint / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        missing = [name for name in sorted(shards) if not (cfg.checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing model weight shard(s): {', '.join(missing)}")
    elif not any(cfg.checkpoint.glob("*.safetensors")):
        raise FileNotFoundError(f"No full model weights in {cfg.checkpoint}")
    for filename in ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"):
        if not (cfg.checkpoint / filename).is_file():
            raise FileNotFoundError(f"Missing checkpoint model code: {cfg.checkpoint / filename}")
    if cfg.prepared_data_root is not None:
        from vlade.data.episode_data import EpisodeData
        from vlade.data.canonical_split import load_canonical_split
        from vlade.data.colab_assets import sha256

        assert cfg.canonical_split_manifest is not None and cfg.canonical_split_sha256 is not None
        assert cfg.canonical_normalization is not None and cfg.canonical_normalization_sha256 is not None
        canonical = load_canonical_split(cfg.canonical_split_manifest, cfg.canonical_split_sha256)
        if sha256(cfg.canonical_normalization) != cfg.canonical_normalization_sha256.lower():
            raise ValueError("Canonical normalization SHA-256 mismatch")
        data = EpisodeData(cfg.prepared_data_root)
        if (data.manifest.get("canonical_manifest_sha256") != canonical.sha256 or
                data.manifest.get("canonical_normalization_sha256") != cfg.canonical_normalization_sha256.lower()):
            raise ValueError("Prepared data was built from a different canonical split or normalization")
    elif not cfg.data_root.is_dir():
        raise FileNotFoundError(f"Missing extracted RLDS data root: {cfg.data_root}")
    if cfg.prepared_data_root is None and not (cfg.data_root / cfg.dataset_name).is_dir():
        raise FileNotFoundError(
            f"Expected extracted TFDS directory {cfg.data_root / cfg.dataset_name}; "
            "the split ZIP archives must be extracted first"
        )
    if cfg.prepared_data_root is None and cfg.dataset_name == "libero_plus_mixdata" and len(_tfds_info_files(cfg.data_root, cfg.dataset_name)) != 1:
        raise FileNotFoundError(f"Expected one TFDS dataset_info.json under {cfg.data_root / cfg.dataset_name}")
    for name in ("action_head", "proprio_projector"):
        find_component(cfg.checkpoint, name)
    if cfg.model == "ava_vla":
        find_component(cfg.checkpoint, "vision_attn_weight_generator")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _activate_upstream(model: str) -> None:
    source = str(REPO_ROOT / SOURCES[model])
    if "prismatic" in sys.modules:
        raise RuntimeError("Run each upstream model in a fresh Python process")
    sys.path.insert(0, source)


def _make_model_and_components(cfg: FFTConfig, device: Any) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(cfg.checkpoint, trust_remote_code=True)
    # Use the pinned source implementation; do not rewrite the original Hub snapshot.
    model_config = OpenVLAConfig.from_pretrained(cfg.checkpoint, local_files_only=True)
    model = OpenVLAForActionPrediction.from_pretrained(
        cfg.checkpoint, config=model_config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="eager" if cfg.model == "ava_vla" else "sdpa",
        local_files_only=True,
    )
    model.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device)
    head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=ACTION_DIM)
    proprio = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM)
    load_component(head, cfg.checkpoint, "action_head")
    load_component(proprio, cfg.checkpoint, "proprio_projector")
    head.to(device=device, dtype=torch.bfloat16)
    proprio.to(device=device, dtype=torch.bfloat16)
    generator = None
    if cfg.model == "ava_vla":
        from prismatic.models.attn_weight_generator import AttentionWeightGenerator
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK

        patch_count = model.vision_backbone.get_num_patches()
        patch_hw = math.isqrt(patch_count)
        if patch_hw * patch_hw != patch_count:
            raise ValueError(f"AVA attention requires square image patch grid, got {patch_count}")
        generator = AttentionWeightGenerator(
            embed_dim=model.llm_dim // 4,
            image_shape=(patch_hw, patch_hw, model.llm_dim),
            action_shape=(NUM_ACTIONS_CHUNK, ACTION_DIM, model.llm_dim),
            text_dim=model.llm_dim,
            num_images=cfg.num_images_in_input,
            max_steps=cfg.max_steps,
            head_type="softmaxscore",
            score_config=(1.9, 0.1, 0.0),
            sink_ids=[68, 75, 180, 187, 324, 331, 436, 443],
            sink_weight=1.0,
        )
        load_component(generator, cfg.checkpoint, "vision_attn_weight_generator")
        generator.to(device=device, dtype=torch.bfloat16)
    return model, processor, head, proprio, generator


def _make_loader(cfg: FFTConfig, model: Any, processor: Any, *, train: bool = True) -> tuple[Any, Any]:
    if cfg.prepared_data_root is not None:
        from vlade.training.prepared_fft import make_loader
        return make_loader(cfg, processor, train=train)
    from torch.utils.data import DataLoader

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

    register_libero_plus_dataset(cfg.dataset_name)
    tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(
        tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder, use_wrist_image=True, use_proprio=True,
    )
    dataset = RLDSDataset(
        cfg.data_root, cfg.dataset_name, transform,
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size, image_aug=cfg.image_aug if train else False,
        train=train,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    return DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collator,
                      num_workers=0, drop_last=True), dataset


def compute_loss(model: Any, head: Any, proprio: Any, generator: Any, batch: dict, device: Any,
                 num_patches: int, step: int, *, capture: Any = None) -> Any:
    """Continuous-action L1 objective, with AVA's attention module in the graph."""
    import torch
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    labels = batch["labels"].to(device)
    batch_size = labels.shape[0]
    target = batch["actions"].to(device=device, dtype=torch.float32)
    if target.shape != (batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM):
        raise ValueError(f"Expected actions [B, {NUM_ACTIONS_CHUNK}, {ACTION_DIM}], got {target.shape}")
    # Upstream collator squeezes the batch dimension for B=1.
    state = batch["proprio"].to(device=device, dtype=torch.bfloat16).reshape(batch_size, -1)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        kwargs = {}
        if generator is not None:
            # RLDSBatchTransform samples individual timesteps; it has no previous
            # action feature. The upstream generator uses its learned zero buffer.
            kwargs = {"temporal_context": None, "vision_attn_weight_generator": generator,
                      "batch_idx": step if model.training else None}
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device=device, dtype=torch.bfloat16),
            labels=labels, output_hidden_states=True,
            proprio=state,
            proprio_projector=proprio,
            use_cache=False,
            **kwargs,
        )
        action_mask = get_current_action_mask(labels[:, 1:]) | get_next_actions_mask(labels[:, 1:])
        hidden = output.hidden_states[-1][:, num_patches:-1]
        expected_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM
        if hidden.shape[:2] != action_mask.shape or not (action_mask.sum(dim=1) == expected_tokens).all():
            raise ValueError("Action token/hidden alignment failed; check truncation and tokenizer")
        action_hidden = hidden[action_mask].reshape(labels.shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        prediction = head.predict_action(action_hidden.to(torch.bfloat16))
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
        if capture is not None:
            capture({"predicted_action": prediction.detach().float().cpu(), "gt_action": target.detach().cpu(),
                     "proprio": state.detach().float().cpu(),
                     "gt_token_ids": labels[:, 1:][action_mask].reshape(batch_size, 56).detach().cpu()})
        if "action_valid_mask" in batch:
            valid = batch["action_valid_mask"].to(device)
            if valid.shape != target.shape or valid.dtype != torch.bool or not valid.any():
                raise ValueError("Invalid action tail mask")
            return (prediction.float()[valid] - target[valid]).abs().mean()
        return torch.nn.functional.l1_loss(prediction.float(), target)


def _make_optimizer(cfg: FFTConfig, params: list[Any]) -> Any:
    if cfg.optimizer == "paged_adamw_8bit":
        from bitsandbytes.optim import PagedAdamW8bit

        return PagedAdamW8bit(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    from torch.optim import AdamW

    return AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    source_manifest = REPO_ROOT / "experiments/manifests/colab_source.json"
    if source_manifest.is_file():
        relative = "." if path == REPO_ROOT else path.relative_to(REPO_ROOT).as_posix()
        return json.loads(source_manifest.read_text(encoding="utf-8"))["source_revisions"].get(relative)
    return None


def _code_fingerprint() -> dict[str, str]:
    paths = [Path(__file__).with_name(name) for name in ("fft.py", "loop.py", "prepared_fft.py", "run_control.py", "debug_records.py")]
    paths.extend(Path(__file__).parents[1] / "data" / name for name in
                 ("episode_data.py", "canonical_split.py", "canonical_normalization.py"))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _prepared_data_identity(cfg: FFTConfig) -> dict | None:
    if cfg.prepared_data_root is None:
        return None
    from vlade.data.episode_data import EpisodeData
    data = EpisodeData(cfg.prepared_data_root)
    return {**{key: data.manifest[key] for key in ("normalization_sha256", "split_sha256",
                                                    "canonical_manifest_sha256",
                                                    "canonical_normalization_sha256")},
            "manifest_sha256": hashlib.sha256((data.root / "manifest.json").read_bytes()).hexdigest()}


def _software_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "tensorflow", "transformers", "bitsandbytes", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def save_checkpoint(cfg: FFTConfig, step: int, model: Any, processor: Any, head: Any,
                    proprio: Any, generator: Any, optimizer: Any, dataset: Any,
                    *, scheduler: Any = None, loop_state: Any = None) -> Path:
    """Save a reloadable full model and separate continuous-action components."""
    import torch
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

    destination = cfg.output_dir / f"step-{step:07d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    path = cfg.output_dir / f".step-{step:07d}-{uuid.uuid4().hex}.incomplete"
    path.mkdir(parents=True)
    model.save_pretrained(path, safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(path)
    for filename in ("configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"):
        source_file = cfg.checkpoint / filename
        if source_file.is_file():
            shutil.copy2(source_file, path / filename)
    torch.save(head.state_dict(), path / f"action_head--{step}_checkpoint.pt")
    torch.save(proprio.state_dict(), path / f"proprio_projector--{step}_checkpoint.pt")
    if generator is not None:
        torch.save(generator.state_dict(), path / f"vision_attn_weight_generator--{step}_checkpoint.pt")
    if cfg.save_optimizer_state:
        if scheduler is not None and loop_state is not None:
            from vlade.training.loop import training_state

            torch.save(training_state(loop_state, optimizer, scheduler), path / "trainer_state.pt")
        else:
            torch.save(optimizer.state_dict(), path / "optimizer.pt")
    save_dataset_statistics(dataset.dataset_statistics, path)
    experiment_path = cfg.config_path or REPO_ROOT / "configs/experiments/libero_plus_fft.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    (path / "vlade_fft.json").write_text(json.dumps({
        "step": step, "model": cfg.model, "dataset_name": cfg.dataset_name,
        "seed": cfg.seed, "optimizer": cfg.optimizer,
        "checkpoint_source": str(cfg.checkpoint),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
        "dataset_revision": experiment["training"]["dataset_revision"],
        "checkpoint_revision": experiment["initial_checkpoints"][cfg.model]["revision"],
        "source_revision": _git_revision(REPO_ROOT),
        "upstream_revision": _git_revision(REPO_ROOT / SOURCES[cfg.model]),
        "software": {**_software_versions(), "cuda": torch.version.cuda},
        "training_code_sha256": _code_fingerprint(),
        "prepared_data_identity": _prepared_data_identity(cfg),
        "hardware": {"gpu": torch.cuda.get_device_name(torch.device(cfg.device)), "vram_bytes": torch.cuda.get_device_properties(torch.device(cfg.device)).total_memory},
        "ava_temporal_training": "zero_context" if generator is not None else None,
    }, indent=2), encoding="utf-8")
    from vlade.training.run_control import seal_checkpoint
    seal_checkpoint(path, step, loop_state.microsteps if loop_state is not None else step * cfg.grad_accumulation_steps)
    (path / "COMPLETE").write_text("ok\n", encoding="utf-8")
    from vlade.utils.filesystem import publish_directory
    publish_directory(path, destination)
    return destination


def validate_resume(cfg: FFTConfig) -> dict[str, Any]:
    """Reject weight-only/incomplete checkpoints and changes to training semantics."""
    assert cfg.resume is not None
    for name in ("COMPLETE", "vlade_fft.json", "trainer_state.pt"):
        if not (cfg.resume / name).is_file():
            raise FileNotFoundError(f"Resume requires {cfg.resume / name}; enable save_optimizer_state")
    from vlade.training.run_control import verify_checkpoint
    sealed = verify_checkpoint(cfg.resume, accumulation=cfg.grad_accumulation_steps)
    saved = json.loads((cfg.resume / "vlade_fft.json").read_text(encoding="utf-8"))
    if sealed["step"] != saved["step"]:
        raise ValueError("Checkpoint seal/metadata step mismatch")
    current = asdict(cfg)
    compatible = ("model", "checkpoint", "data_root", "dataset_name", "max_steps", "batch_size",
                  "grad_accumulation_steps", "learning_rate", "weight_decay", "seed",
                  "shuffle_buffer_size", "num_images_in_input", "optimizer", "gradient_checkpointing",
                  "image_aug", "max_grad_norm", "lr_schedule", "warmup_steps", "prepared_data_root",
                  "canonical_split_manifest", "canonical_split_sha256", "canonical_normalization",
                  "canonical_normalization_sha256")
    for name in compatible:
        value = str(current[name]) if isinstance(current[name], Path) else current[name]
        if saved["configuration"].get(name) != value:
            raise ValueError(f"Resume configuration mismatch: {name}")
    experiment_path = cfg.config_path or REPO_ROOT / "configs/experiments/libero_plus_fft.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if saved["dataset_revision"] != experiment["training"]["dataset_revision"]:
        raise ValueError("Resume dataset revision mismatch")
    if saved["checkpoint_revision"] != experiment["initial_checkpoints"][cfg.model]["revision"]:
        raise ValueError("Resume checkpoint revision mismatch")
    upstream = _git_revision(REPO_ROOT / SOURCES[cfg.model])
    if upstream is not None and saved["upstream_revision"] != upstream:
        raise ValueError("Resume upstream revision mismatch")
    if saved.get("training_code_sha256") != _code_fingerprint():
        raise ValueError("Resume training code mismatch")
    if saved.get("prepared_data_identity") != _prepared_data_identity(cfg):
        raise ValueError("Resume prepared split/normalization mismatch")
    return saved


def train(cfg: FFTConfig) -> None:
    import torch
    from vlade.training.loop import LoopState, evaluate, make_scheduler, restore_training_state, run_loop

    preflight(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("FFT requires a CUDA GPU with BF16 support")
    device = torch.device(cfg.device)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("FFT requires a CUDA GPU with BF16 support")
    # RLDS preprocessing must not reserve CUDA memory needed by the VLA.
    from vlade.training.run_control import configure_tensorflow_cpu
    configure_tensorflow_cpu(cfg.seed)
    _activate_upstream(cfg.model)
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.resume is None and any(cfg.output_dir.glob("step-*")):
        raise FileExistsError("Output already contains checkpoints; choose a new output_dir or --resume")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    manifest_path = cfg.output_dir / f"run-{run_id}.json"
    metrics_path = cfg.output_dir / f"metrics-{run_id}.jsonl"
    experiment_path = cfg.config_path or REPO_ROOT / "configs/experiments/libero_plus_fft.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    manifest = {
        "status": "running", "run_id": run_id,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "source_revision": _git_revision(REPO_ROOT),
        "upstream_revision": _git_revision(REPO_ROOT / SOURCES[cfg.model]),
        "dataset_revision": experiment["training"]["dataset_revision"],
        "checkpoint_revision": experiment["initial_checkpoints"][cfg.model]["revision"],
        "software": {**_software_versions(), "cuda": torch.version.cuda},
        "training_code_sha256": _code_fingerprint(),
        "prepared_data_identity": _prepared_data_identity(cfg),
        "hardware": {"gpu": torch.cuda.get_device_name(device),
                     "vram_bytes": torch.cuda.get_device_properties(device).total_memory},
        "ava_temporal_training": "zero_context" if cfg.model == "ava_vla" else None,
        "resume_data_stream": ("deterministic_query_cursor" if cfg.prepared_data_root else "restarted_not_bitwise_exact") if cfg.resume else None,
        "sampler": "seeded_episode_shuffle_causal_queries_v1" if cfg.prepared_data_root else "upstream_rlds",
    }

    def record_manifest() -> None:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)

    record_manifest()
    state = LoopState()
    try:
        load_cfg = replace(cfg, checkpoint=cfg.resume) if cfg.resume else cfg
        model, processor, head, proprio, generator = _make_model_and_components(load_cfg, device)
        loader, dataset = _make_loader(cfg, model, processor)
        validation = _make_loader(cfg, model, processor, train=False)[0] if cfg.eval_steps else None
        modules = [model, head, proprio] + ([generator] if generator is not None else [])
        for module in modules:
            if not all(param.requires_grad for param in module.parameters()):
                raise RuntimeError("FFT invariant failed: some model/component parameters are frozen")
        params = [p for module in modules for p in module.parameters() if p.requires_grad]
        optimizer = _make_optimizer(cfg, params)
        scheduler = make_scheduler(optimizer, cfg.max_steps, cfg.warmup_steps, cfg.lr_schedule)
        if cfg.resume:
            payload = torch.load(cfg.resume / "trainer_state.pt", map_location="cpu", weights_only=True)
            state = restore_training_state(payload, optimizer, scheduler)
            saved = validate_resume(cfg)
            if state.step != saved["step"]:
                raise ValueError("Trainer state step does not match checkpoint metadata")
            print("Resumed optimizer/scheduler/RNG; prepared queries restore their cursor; raw RLDS restarts.", flush=True)
        num_patches = model.vision_backbone.get_num_patches() * cfg.num_images_in_input + 1
        torch.cuda.reset_peak_memory_stats(device)

        last_training_batch = None
        def loss_fn(batch: Any, step: int) -> Any:
            nonlocal last_training_batch
            if model.training:
                last_training_batch = batch
            return compute_loss(model, head, proprio, generator, batch, device, num_patches, step)

        def before_step(current: Any) -> None:
            if "gradient_diagnostics" not in manifest:
                from vlade.training.run_control import gradient_diagnostics
                groups = {"vision": model.vision_backbone, "language": model.language_model,
                          "projector": model.projector, "action": head, "proprio": proprio}
                if generator is not None:
                    groups["attention_generator"] = generator
                manifest["gradient_diagnostics"] = gradient_diagnostics(groups)

        with metrics_path.open("x", encoding="utf-8") as metrics_file:
            def on_step(current: Any, metrics: dict[str, float]) -> None:
                metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                if validation is not None and (current.step % cfg.eval_steps == 0 or current.step == cfg.max_steps):
                    metrics["val_l1"] = evaluate(modules, validation, loss_fn, current.step, cfg.max_eval_batches)
                metrics_file.write(json.dumps({"step": current.step, "microsteps": current.microsteps,
                                               **metrics}) + "\n")
                metrics_file.flush()
                if current.step == 1 or current.step % cfg.log_steps == 0:
                    print(f"step={current.step} loss={metrics['loss']:.5f} lr={metrics['lr']:.3g}", flush=True)
                if current.step % cfg.save_steps == 0 or current.step == cfg.max_steps:
                    from vlade.training.run_control import checkpoint_space
                    manifest["checkpoint_estimated_bytes"] = checkpoint_space(cfg.output_dir, modules, optimizer)
                    saved_path = save_checkpoint(cfg, current.step, model, processor, head, proprio,
                                                 generator, optimizer, dataset,
                                                 scheduler=scheduler, loop_state=current)
                    print(f"Saved {saved_path}", flush=True)
                    if cfg.debug_records and last_training_batch is not None:
                        from vlade.training.debug_records import emit_checkpoint_debug
                        emit_checkpoint_debug(cfg, saved_path, current.step, modules, last_training_batch,
                                              lambda capture: compute_loss(model, head, proprio, generator,
                                                  last_training_batch, device, num_patches, current.step, capture=capture))
                manifest["step"] = current.step
                record_manifest()

            run_loop(modules=modules, optimizer=optimizer, scheduler=scheduler, batches=loader,
                     loss_fn=loss_fn, max_steps=cfg.max_steps,
                     accumulation_steps=cfg.grad_accumulation_steps, max_grad_norm=cfg.max_grad_norm,
                     state=state, on_before_step=before_step, on_step=on_step)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        step=state.step, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record_manifest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VLADE full-parameter LIBERO+ fine-tuning")
    parser.add_argument("--model", choices=tuple(SOURCES), required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/experiments/libero_plus_fft.json")
    parser.add_argument("--preflight", action="store_true", help="Validate paths and files without loading CUDA")
    parser.add_argument("--max-steps", type=int, help="Override max steps, useful for a one-step smoke test")
    parser.add_argument("--save-steps", type=int, help="Override checkpoint interval")
    parser.add_argument("--resume", type=Path, help="Completed checkpoint containing trainer_state.pt")
    parser.add_argument("--output-dir", type=Path, help="Use a separate output for smoke runs")
    parser.add_argument("--save-training-state", action="store_true", help="Save optimizer/scheduler/RNG for resume")
    args = parser.parse_args(argv)
    cfg = FFTConfig.from_json(args.config, args.model)
    cfg = replace(cfg, **{key: value for key, value in {
        "max_steps": args.max_steps, "save_steps": args.save_steps,
    }.items() if value is not None})
    cfg = replace(cfg, resume=args.resume.resolve() if args.resume else None,
                  output_dir=args.output_dir.resolve() if args.output_dir else cfg.output_dir,
                  save_optimizer_state=cfg.save_optimizer_state or args.save_training_state)
    cfg.validate()
    if args.preflight:
        preflight(cfg)
        print(f"Preflight OK: {cfg.model}")
    else:
        train(cfg)
    return 0
