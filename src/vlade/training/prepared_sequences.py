"""Prepared episode windows adapted to the existing Prismatic frame interface."""

from __future__ import annotations

import json
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


def make_loader(cfg: Any, processor: Any, *, train: bool, suite_map: Any = None) -> tuple:
    """Batch complete K-frame windows; validation never samples train/test episodes."""
    from vlade.data.episode_data import EpisodeData
    from vlade.data.sequences import windows
    from vlade.data.prismatic.prompts import PurePromptBuilder
    from vlade.data.prismatic.transforms import RLDSBatchTransform
    from vlade.data.prismatic.action_tokenizer import ActionTokenizer
    from vlade.data.prismatic.collation import PaddedCollatorForActionPrediction

    if cfg.image_aug:
        raise ValueError("Sequence training requires the shared unaugmented prepared inputs")
    data = EpisodeData(cfg.prepared_data_root)
    transform = RLDSBatchTransform(
        ActionTokenizer(processor.tokenizer),
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    collate = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    class Sequences(IterableDataset):
        dataset_statistics = {cfg.dataset_name: data.stats}
        sampling_statistics = {
            "sequence_length": cfg.sequence_length,
            "query_interval_steps": 8,
            "cursor_unit": "sequence",
            "reset_context": "sequence_start",
            "eligible_episodes": sum(
                e["split"] == ("train" if train else "validation")
                and e["queries"] >= cfg.sequence_length
                for e in data.entries
            ),
            "excluded_short_episodes": sum(
                e["split"] == ("train" if train else "validation")
                and e["queries"] < cfg.sequence_length
                for e in data.entries
            ),
        }

        def __iter__(self):
            skip = 0
            if train and cfg.resume:
                saved = json.loads((cfg.resume / "vlade_fft.json").read_text(encoding="utf-8"))
                skip = saved["step"] * cfg.grad_accumulation_steps * cfg.batch_size
            for samples in windows(
                data,
                split="train" if train else "validation",
                length=cfg.sequence_length,
                seed=cfg.seed,
                skip=skip,
                limit=0 if train else cfg.max_eval_batches * cfg.batch_size,
                sampling=cfg.sequence_sampling,
                balanced=train and suite_map is not None,
                suite_map=suite_map,
                update_size=cfg.grad_accumulation_steps * cfg.batch_size,
            ):
                frames = []
                for sample in samples:
                    item = transform(
                        dict(
                            dataset_name=cfg.dataset_name.encode(),
                            action=sample["gt_action"],
                            observation=dict(
                                image_primary=np.asarray(data.image(sample, "primary"))[None],
                                image_wrist=np.asarray(data.image(sample, "wrist"))[None],
                                proprio=sample["proprio"][None],
                            ),
                            task=dict(language_instruction=sample["instruction"].encode()),
                        )
                    )
                    frame = collate([item])
                    frame["action_valid_mask"] = torch.from_numpy(sample["action_valid_mask"])[None]
                    frame["sample_metadata"] = [
                        {
                            key: sample[key]
                            for key in ("episode_id", "timestep", "instruction", "primary", "wrist")
                        }
                    ]
                    frames.append(frame)
                yield frames

    dataset = Sequences()
    return (
        DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=lambda items: {"sequences": items},
            num_workers=0,
            drop_last=train,
            generator=torch.Generator().manual_seed(cfg.seed),
        ),
        dataset,
    )
