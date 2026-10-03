"""Full-parameter LIBERO+ fine-tuning for the pinned OpenVLA-OFT/AVA-VLA sources.

This module deliberately does not modify either upstream submodule. Each model
must run in its own Python environment because AVA uses a transformers fork.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import subprocess
import sys
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
    if not cfg.data_root.is_dir():
        raise FileNotFoundError(f"Missing extracted RLDS data root: {cfg.data_root}")
    if not (cfg.data_root / cfg.dataset_name).is_dir():
        raise FileNotFoundError(
            f"Expected extracted TFDS directory {cfg.data_root / cfg.dataset_name}; "
            "the split ZIP archives must be extracted first"
        )
    if cfg.dataset_name == "libero_plus_mixdata" and len(_tfds_info_files(cfg.data_root, cfg.dataset_name)) != 1:
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
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
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


def _make_loader(cfg: FFTConfig, model: Any, processor: Any) -> tuple[Any, Any]:
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
        shuffle_buffer_size=cfg.shuffle_buffer_size, image_aug=cfg.image_aug,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    return DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0), dataset


def compute_loss(model: Any, head: Any, proprio: Any, generator: Any, batch: dict, device: Any,
                 num_patches: int, step: int) -> Any:
    """Continuous-action L1 objective, with AVA's attention module in the graph."""
    import torch
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    labels = batch["labels"].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        kwargs = {}
        if generator is not None:
            # RLDSBatchTransform samples individual timesteps; it has no previous
            # action feature. The upstream generator uses its learned zero buffer.
            kwargs = {"temporal_context": None, "vision_attn_weight_generator": generator, "batch_idx": step}
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device=device, dtype=torch.bfloat16),
            labels=labels, output_hidden_states=True,
            proprio=batch["proprio"].to(device=device, dtype=torch.bfloat16),
            proprio_projector=proprio,
            use_cache=False,
            **kwargs,
        )
        action_mask = get_current_action_mask(labels[:, 1:]) | get_next_actions_mask(labels[:, 1:])
        hidden = output.hidden_states[-1][:, num_patches:-1]
        action_hidden = hidden[action_mask].reshape(labels.shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        prediction = head.predict_action(action_hidden.to(torch.bfloat16))
        target = batch["actions"].to(device=device, dtype=torch.bfloat16)
        return torch.nn.functional.l1_loss(prediction, target)


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
    return result.stdout.strip() if result.returncode == 0 else None


def save_checkpoint(cfg: FFTConfig, step: int, model: Any, processor: Any, head: Any,
                    proprio: Any, generator: Any, optimizer: Any, dataset: Any) -> Path:
    """Save a reloadable full model and separate continuous-action components."""
    import torch
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

    path = cfg.output_dir / f"step-{step:07d}"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
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
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory},
        "ava_temporal_training": "zero_context" if generator is not None else None,
    }, indent=2), encoding="utf-8")
    (path / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return path


def train(cfg: FFTConfig) -> None:
    import torch

    preflight(cfg)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("FFT requires a CUDA GPU with BF16 support")
    _activate_upstream(cfg.model)
    set_seed(cfg.seed)
    device = torch.device("cuda:0")
    model, processor, head, proprio, generator = _make_model_and_components(cfg, device)
    loader, dataset = _make_loader(cfg, model, processor)
    modules = [model, head, proprio] + ([generator] if generator is not None else [])
    for module in modules:
        module.train()
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    for module in modules:
        if not all(param.requires_grad for param in module.parameters()):
            raise RuntimeError("FFT invariant failed: some model/component parameters are frozen")
    optimizer = _make_optimizer(cfg, params)
    num_patches = model.vision_backbone.get_num_patches() * cfg.num_images_in_input + 1
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for microstep, batch in enumerate(loader, start=1):
        loss = compute_loss(model, head, proprio, generator, batch, device, num_patches, step)
        (loss / cfg.grad_accumulation_steps).backward()
        if microstep % cfg.grad_accumulation_steps:
            continue
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step == 1 or step % 10 == 0:
            print(f"step={step} loss={loss.detach().float().item():.5f} "
                  f"allocated_gb={torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)
        if step % cfg.save_steps == 0 or step == cfg.max_steps:
            saved = save_checkpoint(cfg, step, model, processor, head, proprio, generator, optimizer, dataset)
            print(f"Saved {saved}", flush=True)
        if step >= cfg.max_steps:
            break
    if step != cfg.max_steps:
        raise RuntimeError(f"RLDS stream ended at step {step}, before requested step {cfg.max_steps}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VLADE full-parameter LIBERO+ fine-tuning")
    parser.add_argument("--model", choices=tuple(SOURCES), required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/experiments/libero_plus_fft.json")
    parser.add_argument("--preflight", action="store_true", help="Validate paths and files without loading CUDA")
    parser.add_argument("--max-steps", type=int, help="Override max steps, useful for a one-step smoke test")
    parser.add_argument("--save-steps", type=int, help="Override checkpoint interval")
    args = parser.parse_args(argv)
    cfg = FFTConfig.from_json(args.config, args.model)
    cfg = replace(cfg, **{key: value for key, value in {
        "max_steps": args.max_steps, "save_steps": args.save_steps,
    }.items() if value is not None})
    cfg.validate()
    if args.preflight:
        preflight(cfg)
        print(f"Preflight OK: {cfg.model}")
    else:
        train(cfg)
    return 0
