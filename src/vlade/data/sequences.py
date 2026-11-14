"""Same-episode contiguous policy-query windows, shuffled as complete sequences."""

from __future__ import annotations

from functools import lru_cache
import random
from typing import Any, Iterator


def windows(
    data: Any,
    *,
    split: str = "train",
    length: int = 4,
    seed: int = 42,
    skip: int = 0,
    limit: int = 0,
    sampling: str = "all_windows",
    suite_map: Any = None,
    balanced: bool = False,
    update_size: int = 1,
) -> Iterator[list[dict]]:
    """Sample windows without crossing episodes/splits; skip counts sequences.

    episode_uniform draws an eligible episode and start with replacement.
    all_windows visits every start once per shuffled epoch. Validation is finite
    and deterministic. Images are read later by the existing verified adapter.
    """
    from .episode_data import SUITES, _read_suite_mapping, _episode_suite

    if split not in {"train", "validation"} or type(length) is not int or length < 1:
        raise ValueError("Sequence sampling requires train/validation and a positive length")
    if (
        any(type(x) is not int for x in (skip, limit, seed))
        or min(skip, limit) < 0
        or sampling not in {"all_windows", "episode_uniform"}
    ):
        raise ValueError("Invalid sequence cursor/sampling")
    if type(update_size) is not int or update_size < 1 or (balanced and split != "train"):
        raise ValueError("Invalid balanced sequence update/split")
    entries = [e for e in data.entries if e["split"] == split and e["queries"] >= length]
    if not entries:
        raise ValueError(f"No {split} episode contains {length} consecutive policy queries")

    @lru_cache(maxsize=128)
    def samples(identifier: str) -> list[dict]:
        return next(data.episodes(split, order=[identifier]))[1]

    def window(entry: dict, start: int) -> list[dict]:
        result = samples(entry["episode_id"])[start : start + length]
        if (
            len(result) != length
            or any(q["episode_id"] != entry["episode_id"] for q in result)
            or [q["timestep"] for q in result]
            != list(range(result[0]["timestep"], result[0]["timestep"] + 8 * length, 8))
        ):
            raise ValueError("Sequence must contain contiguous same-episode policy queries")
        return result

    if split == "validation" or sampling == "all_windows":
        refs = [
            (entry, start) for entry in entries for start in range(entry["queries"] - length + 1)
        ]
        if split == "validation":
            selected = (
                range(len(refs))
                if not limit
                else [i * len(refs) // min(limit, len(refs)) for i in range(min(limit, len(refs)))]
            )
            for index in list(selected)[skip:]:
                yield window(*refs[index])
            return
        if balanced:
            raise ValueError("Balanced sequences require episode_uniform sampling")
        epoch, offset = divmod(skip, len(refs))
        emitted = 0
        while True:
            order = list(refs)
            random.Random(seed + epoch * 1000003).shuffle(order)
            for entry, start in order[offset:]:
                yield window(entry, start)
                emitted += 1
                if limit and emitted >= limit:
                    return
            epoch, offset = epoch + 1, 0

    pools = {"all": entries}
    if balanced:
        by_episode, by_instruction = _read_suite_mapping(data, suite_map)
        pools = {suite: [] for suite in SUITES}
        for entry in entries:
            suite = _episode_suite(entry, samples(entry["episode_id"]), by_episode, by_instruction)
            pools[suite].append(entry)
        if any(not pool for pool in pools.values()):
            raise ValueError("Every balanced suite needs an episode with a full sequence")
    rng = random.Random(seed)
    used_groups: set[str] = set()
    position, emitted = 0, 0
    while True:
        order = list(pools)
        rng.shuffle(order)
        for suite in order:
            if position % update_size == 0:
                used_groups.clear()
            pool = pools[suite]
            entry = rng.choice(pool)
            group = entry.get("content_group_id", entry["episode_id"])
            if balanced and group in used_groups:
                alternatives = [
                    e for e in pool if e.get("content_group_id", e["episode_id"]) not in used_groups
                ]
                if alternatives:
                    entry = rng.choice(alternatives)
                    group = entry.get("content_group_id", entry["episode_id"])
            start = rng.randrange(entry["queries"] - length + 1)
            used_groups.add(group)
            if position >= skip:
                yield window(entry, start)
                emitted += 1
                if limit and emitted >= limit:
                    return
            position += 1
