"""Verified S0 initialization, distinct from resuming a KD optimizer run."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vlade.data.episode_data import EpisodeData

from vlade.data.colab_assets import sha256
from vlade.models.students.qwen import StudentConfig, resolve_pretrained_directory
from vlade.training.run_control import verify_checkpoint


def initial_checkpoint(settings: dict, data: EpisodeData, config: StudentConfig) -> tuple[Path, dict]:
    """Require the complete schema-1 Student, including its trained action head."""
    from vlade.storage.assets import resolve_input

    root = resolve_input(settings["initial_checkpoint"])
    kind = settings.get("initialization", "pretrained_backbone")
    if kind == "pretrained_backbone":
        root = resolve_pretrained_directory(config, root)
        if not any((root / name).is_file() for name in
                   ("model.safetensors", "model.safetensors.index.json")):
            raise FileNotFoundError("Download the pinned Qwen model before Student training")
        return root, {"initial_download_manifest_sha256": sha256(root / "download_manifest.json")}
    if kind != "student_checkpoint":
        raise ValueError("initialization must be pretrained_backbone or student_checkpoint")
    seal = verify_checkpoint(root)
    saved = json.loads((root / "model/student.json").read_text(encoding="utf-8"))
    if saved != asdict(config):
        raise ValueError("S0 architecture/config differs; schema-2 QwenVLA is not schema-1 QwenStudent")
    identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    if identity.get("data_manifest_sha256") != sha256(data.root / "manifest.json"):
        raise ValueError("S0 was trained with a different prepared data/split/normalization contract")
    if not (root / "model/action_head.pt").is_file() or not (root / "processor").is_dir():
        raise ValueError("S0 must include its trained action head and saved processor")
    backbone = root / "model/backbone"
    if not (backbone / "config.json").is_file() or not any((backbone / name).is_file() for name in
            ("model.safetensors", "model.safetensors.index.json")):
        raise ValueError("S0 must include its complete native backbone weights and config")
    return root, {"initial_student_checkpoint": {
        "checkpoint_manifest_sha256": sha256(root / "checkpoint_manifest.json"),
        "identity_sha256": sha256(root / "identity.json"), "step": seal["step"]}}


def saved_processor(config: StudentConfig, directory: Path) -> Any:
    """Load processor bytes inside an already verified generated checkpoint."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(directory), local_files_only=True,
        trust_remote_code=False, size={"shortest_edge": min(65536, config.max_pixels_per_image),
                                       "longest_edge": config.max_pixels_per_image})
