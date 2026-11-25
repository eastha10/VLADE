"""Optional Plan CE for the existing AVA FFT loop; never synthesize Plan targets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vlade.distillation.teacher_plan import PLAN_PROMPT_VERSION, normalize_plan, plan_prompt


class PlanAnnotations:
    """Read immutable HF annotations bound to the exact prepared episode contract."""

    def __init__(self, reference: str | Path, data: Any) -> None:
        from vlade.storage.assets import resolve_input_record, sha256

        # Local cache paths must pass the same HF verification as hfasset:// keys.
        root, record = resolve_input_record(reference)
        if (record.get("format") not in {"directory", "upstream_files"}
                or not {"manifest.json", "plans.jsonl"} <= record.get("files", {}).keys()):
            raise ValueError("Plan manifest and annotations must both be pinned HF files")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        expected = {"manifest_sha256": sha256(data.root / "manifest.json"),
                    **{key: data.manifest[key] for key in ("split_sha256", "normalization_sha256")}}
        if (manifest.get("schema_version") != 1 or manifest.get("prompt_version") != PLAN_PROMPT_VERSION
                or manifest.get("prepared_data_identity") != expected
                or manifest.get("annotation_source") not in {"human_reviewed", "externally_reviewed"}):
            raise ValueError("Plan annotation schema, review source or prepared data identity mismatch")
        self.identity = {"prompt_version": PLAN_PROMPT_VERSION, "objective": "plan_ce",
                         "annotations_manifest_sha256": sha256(root / "manifest.json"),
                         "annotations_sha256": sha256(root / "plans.jsonl"),
                         "annotations_revision": record["revision"], "annotations_repo_id": record["repo_id"]}
        self.rows = {}
        train_episodes = {entry["episode_id"]: entry["queries"] for entry in data.entries if entry["split"] == "train"}
        for line in (root / "plans.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = row["episode_id"], row["timestep"]
            if (row.get("split") != "train" or key[0] not in train_episodes
                    or type(key[1]) is not int or key[1] < 0 or key[1] % 8 or key in self.rows
                    or key[1] // 8 >= train_episodes[key[0]]):
                raise ValueError("Plan annotations must be unique canonical Train queries")
            self.rows[key] = {"instruction": row["instruction"], "plan": normalize_plan(row["plan"])}
        if not self.rows:
            raise ValueError("Plan annotations are empty")
        if len(self.rows) != sum(train_episodes.values()):
            raise ValueError("Plan annotations must cover every query of the selected prepared Train input")

    def target(self, sample: dict) -> str:
        """A missing target is an error, never an instruction echo or pseudo-label."""
        row = self.rows.get((sample["episode_id"], sample["timestep"]))
        if row is None or row["instruction"] != sample["instruction"]:
            raise ValueError("Missing Plan target or instruction mismatch for sampled Train query")
        return row["plan"]


def annotation_loader(cfg: Any, data: Any, *, train: bool) -> PlanAnnotations | None:
    reference = getattr(cfg, "plan_annotations", None)
    return PlanAnnotations(reference, data) if train and reference is not None else None


def attach_plan_batch(batch: dict, annotations: PlanAnnotations | None, weight: float,
                      tokenizer: Any) -> None:
    """Attach explicit targets to an action batch without changing action-token labels."""
    if annotations is not None:
        batch["plan_supervision"] = {
            "targets": [annotations.target(row) for row in batch["sample_metadata"]],
            "weight": weight, "tokenizer": tokenizer,
        }


def plan_ce_loss(model: Any, proprio_projector: Any, batch: dict, device: Any) -> Any:
    """Separate causal LM forward: only Plan tokens + EOS carry CE labels.

    The action forward masks action embeddings and has different label semantics;
    its output.loss cannot supervise natural language. No chat template or new
    special tokens are introduced for this PurePrompt-trained Llama backbone.
    """
    import torch
    from vlade.distillation.teacher_plan import plan_generation_kwargs

    supervision = batch["plan_supervision"]
    tokenizer = supervision["tokenizer"]
    eos = tokenizer.eos_token_id
    vocab = model.get_output_embeddings().weight.shape[0]
    forbidden = set(plan_generation_kwargs(tokenizer, vocab, 128)["suppress_tokens"])
    if len(supervision["targets"]) != len(batch["sample_metadata"]):
        raise ValueError("Plan batch/metadata size mismatch")
    losses = []
    for index, (row, target) in enumerate(zip(batch["sample_metadata"], supervision["targets"])):
        prefix = tokenizer(plan_prompt(row["instruction"], row["timestep"]), add_special_tokens=True).input_ids
        body = tokenizer(normalize_plan(target), add_special_tokens=False).input_ids
        if not body or any(token in forbidden or token == eos or not 0 <= token < vocab for token in body):
            raise ValueError("Plan target contains non-text/action/special tokens")
        if len(body) + 1 > 128:
            raise ValueError("Plan target exceeds the supported 128-token generation budget")
        ids = torch.tensor([prefix + body + [eos]], device=device)
        text_labels = ids.clone()
        text_labels[:, :len(prefix)] = -100
        placement = getattr(model, "_vlade_teacher_placement", None)
        dtype = getattr(torch, placement.options["dtype"]) if placement else torch.bfloat16
        images = batch["pixel_values"][index:index + 1].to(device=device, dtype=dtype)
        vision = model._process_vision_features(images)
        state = batch["proprio"].reshape(len(batch["sample_metadata"]), -1)[index:index + 1]
        vision = model._process_proprio_features(vision, state.to(device=device, dtype=vision.dtype), proprio_projector)
        embeddings, mask = model._build_multimodal_attention(model.get_input_embeddings()(ids), vision,
                                                             torch.ones_like(ids))
        if embeddings.shape[1] > min(tokenizer.model_max_length, model.language_model.config.max_position_embeddings):
            raise ValueError("Plan multimodal sequence would be truncated")
        labels = model._build_multimodal_labels(text_labels, vision)
        # HF shifts causal labels once. Prompt, camera and proprio labels stay -100.
        output = model.language_model(inputs_embeds=embeddings, attention_mask=mask,
                                      labels=labels, use_cache=False, return_dict=True)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite supervised Plan CE")
        losses.append(output.loss)
    return torch.stack(losses).mean() * supervision["weight"]


def checkpoint_plan_contract(cfg: Any, step: int) -> dict | None:
    """Record actual supervised optimizer steps inside the sealed checkpoint metadata."""
    if getattr(cfg, "plan_annotations", None) is None:
        return None
    from vlade.data.episode_data import EpisodeData
    from vlade.storage.assets import resolve_input

    annotations = PlanAnnotations(cfg.plan_annotations, EpisodeData(resolve_input(cfg.prepared_data_root)))
    return {**annotations.identity, "weight": cfg.plan_loss_weight, "trained_steps": step}
