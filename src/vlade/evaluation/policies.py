"""Research policy adapters using project-owned model code and verified HF inputs."""

from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def normalize_proprio(proprio: np.ndarray, stats: dict) -> np.ndarray:
    mask = np.asarray(stats.get("mask", [True] * 8))
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    return np.clip(
        np.where(mask, 2 * (proprio - low) / (high - low + 1e-8) - 1, proprio), -1.0, 1.0
    )


def select_stats(stats: dict, suite: str, requested: str | None) -> str:
    if requested:
        if requested not in stats:
            raise ValueError(f"Unknown normalization key: {requested}")
        return requested
    for key in (suite, suite + "_no_noops", "libero_plus_mixdata"):
        if key in stats:
            return key
    if len(stats) == 1:
        return next(iter(stats))
    raise ValueError("Set an explicit unnorm_key for this checkpoint/suite")


class PrismaticPolicy:
    """AVA/OFT continuous action chunks; episode temporal state is explicit."""

    def __init__(self, model: str, checkpoint: str, settings: dict):
        from vlade.storage.assets import resolve_input
        from vlade.training.fft import FFTConfig, _make_model_and_components

        path = resolve_input(checkpoint)
        self.settings = settings
        self.device = torch.device(settings.get("device", "cuda:0"))
        if (path / "paper_ava.json").is_file():
            if model != "ava_vla":
                raise ValueError("Native AVA paper adapters require the AVA policy")
            from .paper import load_components

            components = load_components(path, self.device)
        else:
            cfg = FFTConfig(model, path, Path("."), Path("."), gradient_checkpointing=False)
            components = _make_model_and_components(cfg, self.device)
        self.model, self.processor, self.head, self.proprio, self.generator = components
        for module in (self.model, self.head, self.proprio, self.generator):
            if module is not None:
                module.eval().requires_grad_(False)
        stats_file = path / "dataset_statistics.json"
        if stats_file.is_file():
            self.model.norm_stats.update(json.loads(stats_file.read_text(encoding="utf-8")))
        self.state = None

    def reset(self, suite: str) -> None:
        self.state = None
        self.key = select_stats(self.model.norm_stats, suite, self.settings.get("unnorm_key"))

    @torch.inference_mode()
    def predict(self, observation: dict, instruction: str, timestep: int) -> np.ndarray:
        from benchmarks.libero.images import prepare_images_for_vla

        images = prepare_images_for_vla(
            [observation["full_image"], observation["wrist_image"]],
            SimpleNamespace(center_crop=self.settings.get("center_crop", True)),
        )
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, images[0], return_tensors="pt")
        wrist = self.processor(
            self.processor.tokenizer.decode(inputs["input_ids"][0]), images[1], return_tensors="pt"
        )
        inputs["pixel_values"] = torch.cat((inputs["pixel_values"], wrist["pixel_values"]), dim=1)
        inputs = {
            key: value.to(
                self.device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype
            )
            for key, value in inputs.items()
        }
        proprio = normalize_proprio(
            observation["state"], self.model.norm_stats[self.key]["proprio"]
        )
        result = self.model.predict_action(
            **inputs,
            unnorm_key=self.key,
            proprio=proprio,
            proprio_projector=self.proprio,
            action_head=self.head,
            temporal_context=self.state,
            temporal_feature_layer=self.settings.get("temporal_feature_layer", -2),
            vision_attn_weight_generator=self.generator,
            do_sample=False,
        )
        self.state = result[1].detach() if self.generator is not None else None
        return np.asarray(result[0])


class StudentPolicy:
    """The maintained Qwen task/KD policy with the same causal query history."""

    def __init__(self, checkpoint: str, settings: dict):
        from vlade.storage.assets import resolve_input
        from vlade.models.students.qwen import QwenStudent, load_processor

        path = resolve_input(checkpoint)
        folder = path / "model" if (path / "model/student.json").is_file() else path
        self.model = (
            QwenStudent.load(folder)
            .to(settings.get("device", "cuda:0"))
            .eval()
            .requires_grad_(False)
        )
        self.settings = settings
        self.processor = load_processor(
            self.model.student_config, directory=resolve_input("hfasset://qwen_initial")
        )
        data = resolve_input(settings.get("normalization_data", "hfasset://prepared_d_seed42"))
        self.stats = json.loads((data / "normalization.json").read_text(encoding="utf-8"))
        self.history = []

    def reset(self, suite: str) -> None:
        self.history = []

    @torch.inference_mode()
    def predict(self, observation: dict, instruction: str, timestep: int) -> np.ndarray:
        from vlade.models.students.qwen import HistoryFrame, prepare_inputs
        from PIL import Image

        self.history.append(
            HistoryFrame(
                timestep,
                Image.fromarray(observation["full_image"].copy()),
                Image.fromarray(observation["wrist_image"].copy()),
            )
        )
        self.history = self.history[-2:]
        inputs = prepare_inputs(
            self.processor,
            self.model.student_config,
            instruction=instruction,
            frames=self.history,
            current_timestep=timestep,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        proprio = torch.as_tensor(
            normalize_proprio(observation["state"], self.stats["proprio"]),
            dtype=torch.float32,
            device=device,
        )[None]
        normalized = self.model(inputs, proprio).actions[0].float().cpu().numpy()
        stats = self.stats["action"]
        low = np.asarray(stats["q01"])
        high = np.asarray(stats["q99"])
        return np.where(
            np.asarray(stats.get("mask", [True] * 7)),
            0.5 * (normalized + 1) * (high - low + 1e-8) + low,
            normalized,
        )


def load_policy(kind: str, checkpoint: str, settings: dict):
    if kind in {"ava_vla", "openvla_oft"}:
        return PrismaticPolicy(kind, checkpoint, settings)
    if kind == "qwen":
        return StudentPolicy(checkpoint, settings)
    raise ValueError(f"Unknown evaluation policy: {kind}")
