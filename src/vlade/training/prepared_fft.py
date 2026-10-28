"""Adapt the common episode split to the unchanged upstream action-token interface."""
from __future__ import annotations
import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from vlade.data.episode_data import EpisodeData, query_stream


def make_loader(cfg: Any, processor: Any, *, train: bool = True) -> tuple[Any, Any]:
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform

    if cfg.image_aug:
        raise ValueError("The shared prepared-data path requires image_aug=false for explicit common inputs")
    data = EpisodeData(cfg.prepared_data_root)
    transform = RLDSBatchTransform(ActionTokenizer(processor.tokenizer), processor.tokenizer,
                                   image_transform=processor.image_processor.apply_transform,
                                   prompt_builder_fn=PurePromptBuilder, use_wrist_image=True, use_proprio=True)

    class Queries(IterableDataset):
        dataset_statistics = {cfg.dataset_name: data.stats}

        def __iter__(self):
            skip = 0
            if train and cfg.resume:
                saved = json.loads((cfg.resume / "vlade_fft.json").read_text(encoding="utf-8"))
                skip = saved["step"] * cfg.grad_accumulation_steps * cfg.batch_size
            for sample, _ in query_stream(data, "train" if train else "validation", seed=cfg.seed,
                                         repeat=train, skip=skip,
                                         limit=0 if train else cfg.max_eval_batches * cfg.batch_size):
                item = transform({"dataset_name": cfg.dataset_name.encode(),
                                  "action": sample["gt_action"],
                                  "observation": {"image_primary": np.asarray(data.image(sample, "primary"))[None],
                                                  "image_wrist": np.asarray(data.image(sample, "wrist"))[None],
                                                  "proprio": sample["proprio"][None]},
                                  "task": {"language_instruction": sample["instruction"].encode()}})
                item["action_valid_mask"] = torch.from_numpy(sample["action_valid_mask"])
                item["sample_metadata"] = {key: sample[key] for key in
                    ("episode_id", "timestep", "instruction", "primary", "wrist")}
                yield item

    base = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,
                                            processor.tokenizer.pad_token_id, padding_side="right")

    def collate(items):
        result = base(items)
        result["action_valid_mask"] = torch.stack([item["action_valid_mask"] for item in items])
        result["sample_metadata"] = [item["sample_metadata"] for item in items]
        return result

    dataset = Queries()
    return DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate, num_workers=0, drop_last=True,
                      generator=torch.Generator().manual_seed(cfg.seed)), dataset
