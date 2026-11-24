"""Sequential, resumable action cache from a deliberately selected FFT Teacher T1."""
from __future__ import annotations
import argparse
import hashlib
import json
import importlib.metadata
import time
import uuid
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vlade.data.colab_assets import sha256
from vlade.data.episode_data import CONVENTION, EpisodeData, write_json
from vlade.data.teacher_records import TeacherProvenance, TeacherRecordWriter, load_teacher_records
from vlade.training.run_control import code_hashes, validate_teacher_report, verify_checkpoint, configure_tensorflow_cpu
from vlade.utils.filesystem import publish_directory


def bundle_hash(directory: Path) -> str:
    """Identity of model/config/tokenizer/component bytes, excluding logs and optimizer."""
    files = sorted(path for path in directory.iterdir() if path.is_file() and
                   path.name not in {"trainer_state.pt", "optimizer.pt", "COMPLETE"} and
                   not path.name.startswith("download_manifest"))
    if not files:
        raise ValueError("Empty checkpoint")
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.name}\0{sha256(path)}\n".encode())
    return digest.hexdigest()


def validate_selection(selection: dict, data: EpisodeData) -> Path:
    from vlade.storage.assets import resolve_path
    checkpoint = resolve_path(selection["checkpoint"])
    if not (checkpoint / "COMPLETE").is_file() or not (checkpoint / "vlade_fft.json").is_file():
        raise ValueError("T1 must be a completed VLADE FFT checkpoint; public T0 is not T1")
    verify_checkpoint(checkpoint)
    if selection.get("quantized") is not False or selection.get("frozen") is not True:
        raise ValueError("Base cache requires frozen, unquantized T1")
    if not selection.get("selection_reason", "").strip():
        raise ValueError("A validation-based selection reason is required")
    report = Path(selection["validation_report"])
    if sha256(report) != selection["validation_report_sha256"]:
        raise ValueError("Validation report changed")
    validate_teacher_report(json.loads(report.read_text(encoding="utf-8")), checkpoint, data.manifest["split_sha256"])
    for key in ("normalization_sha256", "split_sha256"):
        if selection[key] != data.manifest[key]:
            raise ValueError(f"Selection/data mismatch: {key}")
    if bundle_hash(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint bytes changed")
    saved = json.loads((checkpoint / "vlade_fft.json").read_text(encoding="utf-8"))
    if saved["step"] < 1 or saved["model"] != "ava_vla":
        raise ValueError("Expected a trained AVA-VLA checkpoint")
    if selection["teacher_step"] != saved["step"]:
        raise ValueError("Selection/checkpoint step mismatch")
    expected = {**{key: data.manifest[key] for key in ("normalization_sha256", "split_sha256")},
                "manifest_sha256": sha256(data.root / "manifest.json")}
    recorded = saved.get("prepared_data_identity")
    for key in ("canonical_manifest_sha256", "canonical_normalization_sha256"):
        if saved.get("plan_supervision") is not None or isinstance(recorded, dict) and key in recorded:
            expected[key] = data.manifest[key]
    if saved.get("prepared_data_identity") != expected:
        raise ValueError("Teacher was trained with a different data contract")
    return checkpoint


class SequentialTeacher:
    def __init__(self, checkpoint: Path, *, device: str = "cuda:0", temporal_feature_layer: int = -2,
                 timings=None):
        from vlade.storage.assets import resolve_path
        checkpoint = resolve_path(checkpoint)
        metadata_path = checkpoint / "vlade_fft.json"
        self.checkpoint_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        from vlade.training.fft import FFTConfig, _activate_upstream, _make_model_and_components
        configure_tensorflow_cpu()
        _activate_upstream("ava_vla")
        cfg = FFTConfig("ava_vla", checkpoint, Path("."), Path("."), gradient_checkpointing=False)
        self.device = torch.device(device)
        self.timings = timings
        self._timing_predictions = 0
        self.temporal_feature_layer = temporal_feature_layer
        self.model, self.processor, self.head, self.proprio, self.generator = _make_model_and_components(cfg, self.device)
        for module in (self.model, self.head, self.proprio, self.generator):
            module.eval()
            module.requires_grad_(False)
        # Disable only output unnormalization: data/records already use normalized coordinates.
        stats = {"q01": [-1.] * 7, "q99": [1.] * 7, "min": [-1.] * 7,
                 "max": [1.] * 7, "mask": [False] * 7}
        self.model.norm_stats = {"vlade_normalized": {"action": stats}}
        self.context = None
        self.feature_dimension = self.model.llm_dim

    def reset(self) -> None:
        self.context = None

    def _observation_inputs(self, sample: dict, data: EpisodeData, prompt: str) -> dict:
        inputs = self.processor(prompt, data.image(sample, "primary"), return_tensors="pt")
        wrist = self.processor(prompt, data.image(sample, "wrist"), return_tensors="pt")
        inputs["pixel_values"] = torch.cat((inputs["pixel_values"], wrist["pixel_values"]), dim=1)
        return {key: value.to(self.device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                for key, value in inputs.items()}

    @torch.inference_mode()
    def plan(self, sample: dict, data: EpisodeData, *, max_new_tokens: int) -> str:
        """Use T1's own visual prefix and language decoder for the current observation."""
        from vlade.distillation.teacher_plan import (
            InvalidTeacherPlan, decode_plan_tokens, plan_generation_kwargs, plan_prompt, require_plan_supervision,
        )

        require_plan_supervision(self.checkpoint_metadata)
        prompt = plan_prompt(sample["instruction"], sample["timestep"])
        inputs = self._observation_inputs(sample, data, prompt)
        model = self.model
        tokens = model.get_input_embeddings()(inputs["input_ids"])
        vision = model._process_vision_features(inputs["pixel_values"])
        proprio = torch.as_tensor(sample["proprio"], device=vision.device, dtype=vision.dtype)
        vision = model._process_proprio_features(vision, proprio, self.proprio)
        prefix, mask = model._build_multimodal_attention(tokens, vision, inputs["attention_mask"])
        tokenizer = self.processor.tokenizer
        generation = plan_generation_kwargs(tokenizer, model.get_output_embeddings().weight.shape[0], max_new_tokens)
        with (self.timings.measure("t1_plan_generation", phase="inference",
                                  episode_id=sample["episode_id"], timestep=sample["timestep"])
              if self.timings is not None else nullcontext()):
            generated = model.language_model.generate(
                inputs_embeds=prefix, attention_mask=mask, **generation,
            )
        token_ids = generated[0].tolist()
        try:
            return decode_plan_tokens(token_ids, tokenizer)
        except InvalidTeacherPlan as error:
            raise InvalidTeacherPlan(
                f"{error}; token_ids={token_ids[:128]}; episode_id={sample['episode_id']!r}; "
                f"timestep={sample['timestep']}"
            ) from error

    @torch.inference_mode()
    def predict(self, sample: dict, data: EpisodeData) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = f"In: What action should the robot take to {sample['instruction'].lower()}?\nOut:"
        inputs = self._observation_inputs(sample, data, prompt)
        with (self.timings.measure("t1_model_forward", phase="inference",
                                  episode_id=sample["episode_id"], timestep=sample["timestep"],
                                  is_warmup=self._timing_predictions < 2)
              if self.timings is not None else nullcontext()):
            action, hidden, *_ = self.model.predict_action(
                **inputs, unnorm_key="vlade_normalized", proprio=sample["proprio"],
                proprio_projector=self.proprio, action_head=self.head,
                temporal_context=self.context, temporal_feature_layer=self.temporal_feature_layer,
                vision_attn_weight_generator=self.generator, do_sample=False)
        self._timing_predictions += 1
        if hidden.shape != (1, 56, self.model.llm_dim):
            raise ValueError(f"Unexpected temporal feature shape: {hidden.shape}")
        self.context = hidden.detach()
        return torch.as_tensor(np.asarray(action), dtype=torch.float32), hidden[0].float().cpu()


def generate(data: EpisodeData, teacher: Any, selection: dict, output: Path, *, hidden: bool = False,
             max_episodes: int = 0, max_bytes_per_episode: int = 512 * 1024**2) -> dict:
    """Resume only sealed episodes, and replay the first episode before publishing."""
    provenance = TeacherProvenance(
        purpose="fixed_kd", checkpoint_id=selection["checkpoint"], checkpoint_sha256=selection["checkpoint_sha256"],
        dataset_revision=data.manifest["dataset_revision"], split_sha256=data.manifest["split_sha256"],
        normalization_sha256=data.manifest["normalization_sha256"],
        tokenizer_id="LiAuto-DSR/avavla-libero-4in1", tokenizer_revision="090bf68d234ea13bc78476236f82458b25b797b8",
        action_convention=CONVENTION, temporal_mode="episode_context", query_interval_steps=8,
        frozen=True, quantized=False)
    provenance.validate()
    identity = {"selection": selection, "hidden": hidden, "max_episodes": max_episodes,
                "data_manifest_sha256": sha256(data.root / "manifest.json"),
                "code_sha256": code_hashes([Path(__file__), Path(__file__).with_name("fft.py"),
                    Path(__file__).with_name("run_control.py"), Path(__file__).parents[1] / "data/episode_data.py",
                    Path(__file__).parents[1] / "data/teacher_records.py"]),
                "software": {"torch": str(torch.__version__), "transformers": importlib.metadata.version("transformers")}}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "cache_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Cache output belongs to a different Teacher/data/budget")
    else:
        write_json(config_path, identity)
    (output / "COMPLETE").unlink(missing_ok=True)
    started = time.monotonic()
    parts, queries, replay_error, replayed = [], 0, None, []
    seen_episodes = set()
    count = sum(entry["split"] == "train" for entry in data.entries)
    count = min(count, max_episodes) if max_episodes else count
    replay_indices = {index * (count - 1) // max(1, min(3, count) - 1) for index in range(min(3, count))}
    step = selection["teacher_step"]
    for entry, samples in data.episodes("train"):
        if max_episodes and len(parts) >= max_episodes:
            break
        if entry["episode_id"] in seen_episodes:
            raise ValueError("Duplicate cache episode identity")
        seen_episodes.add(entry["episode_id"])
        destination = output / entry["episode_id"]
        if not (destination / "manifest.json").is_file():
            teacher.reset()
            stage = output / f".{entry['episode_id']}-{uuid.uuid4().hex}.incomplete"
            with TeacherRecordWriter(stage, provenance, max_records=len(samples),
                                     max_bytes=max_bytes_per_episode) as writer:
                for index, sample in enumerate(samples):
                    action, features = teacher.predict(sample, data)
                    record = {"episode_id": entry["episode_id"], "timestep": sample["timestep"],
                              "teacher_step": step, "instruction": sample["instruction"],
                              "episode_start": index == 0, "episode_end": index == len(samples) - 1,
                              "observation_refs": {key: f"episodes/{entry['episode_id']}/{sample[key]}" for key in ("primary", "wrist")},
                              "checkpoint_sha256": provenance.checkpoint_sha256,
                              "normalization_sha256": provenance.normalization_sha256,
                              "proprio": torch.from_numpy(sample["proprio"]), "gt_action": torch.from_numpy(sample["gt_action"]),
                              "predicted_action": action, "action_valid_mask": torch.from_numpy(sample["action_valid_mask"])}
                    if hidden:
                        record.update(action_hidden_states=features, hidden_layer="language_model.hidden_states[-2]",
                                      hidden_layout="horizon_major_action_dimension_minor")
                    writer.write(record)
            publish_directory(stage, destination)
        rows = list(load_teacher_records(destination, expected_provenance=provenance))
        if len(rows) != len(samples):
            raise ValueError("Cache query count differs from source episode")
        for index, (row, sample) in enumerate(zip(rows, samples)):
            expected_refs = {key: f"episodes/{entry['episode_id']}/{sample[key]}" for key in ("primary", "wrist")}
            if (row["episode_id"] != entry["episode_id"] or row["timestep"] != sample["timestep"] or
                    row["teacher_step"] != step or row["instruction"] != sample["instruction"] or
                    row["observation_refs"] != expected_refs or row["episode_start"] != (index == 0) or
                    row["episode_end"] != (index == len(samples) - 1) or
                    not torch.equal(row["proprio"], torch.from_numpy(sample["proprio"])) or
                    not torch.equal(row["action_valid_mask"], torch.from_numpy(sample["action_valid_mask"])) or
                    not torch.equal(row["gt_action"], torch.from_numpy(sample["gt_action"]))):
                raise ValueError("Cache target/source mismatch")
            features = row.get("action_hidden_states")
            if (features is not None) != hidden:
                raise ValueError("Cache feature storage mode differs from the request")
            dimension = getattr(teacher, "feature_dimension", None)
            if hidden and (row.get("hidden_layer") != "language_model.hidden_states[-2]" or
                           (dimension is not None and features.shape != (56, dimension))):
                raise ValueError("Cache feature layer/shape differs from the Teacher")
        if len(parts) in replay_indices:
            teacher.reset()
            replay_error = 0.0 if replay_error is None else replay_error
            for row, sample in zip(rows, samples):
                action, features = teacher.predict(sample, data)
                error = float((action - row["predicted_action"]).abs().max())
                replay_error = max(replay_error, error)
                if not torch.allclose(action, row["predicted_action"], atol=1e-5, rtol=1e-4):
                    raise ValueError("Teacher replay/action comparison failed")
                if hidden and not torch.allclose(features, row["action_hidden_states"], atol=1e-5, rtol=1e-4):
                    raise ValueError("Teacher replay/feature comparison failed")
            replayed.append(entry["episode_id"])
        parts.append({"episode_id": entry["episode_id"], "manifest_sha256": sha256(destination / "manifest.json"),
                      "queries": len(rows)})
        queries += len(rows)
        print(f"Sealed {len(parts)} episodes / {queries} queries", flush=True)
    if not parts:
        raise ValueError("No cache episodes")
    result = {"status": "completed", "scope": "pilot" if max_episodes else "full_train_split",
              "episodes": parts, "queries": queries, "sample_replay_max_abs_error": replay_error,
              "invocation_seconds": time.monotonic() - started,
              "bytes": sum((output / part["episode_id"] / "records.jsonl").stat().st_size for part in parts),
              "incomplete_bytes": sum(path.stat().st_size for path in output.glob(".*.incomplete/records.jsonl")),
              "representation_features_saved": hidden,
              "replay_scope": "up_to_three_uniformly_spaced_complete_episodes_all_queries",
              "replayed_episodes": replayed,
              "teacher_training_temporal_policy": selection.get("training_temporal_policy", "unknown")}
    write_json(output / "cache_manifest.json", result)
    (output / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--hidden", action="store_true")
    args = parser.parse_args()
    if args.max_episodes < 0:
        raise ValueError("max_episodes must be nonnegative")
    data = EpisodeData(args.data)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    checkpoint = validate_selection(selection, data)
    generate(data, SequentialTeacher(checkpoint), selection, args.output,
             hidden=args.hidden, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
