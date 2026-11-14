"""Preserve fish13031e suite quotas, independent queries and deterministic resume."""
from collections import Counter
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'src'))
from vlade.data.episode_data import query_stream,query_count

def keys(rows):
    return [(sample['episode_id'],sample['timestep']) for sample,_ in rows]

@pytest.mark.parametrize("one_group_per_suite", [False, True])
def test_independent_suite_shuffle_preserves_selections_and_resume(one_group_per_suite):
    entries, queries = [], {}
    for suite, count in zip(("spatial", "goal", "object", "libero_10"), (10, 9, 8, 6), strict=True):
        for index in range(count):
            episode_id = f"{suite}-{index}"
            group = suite if one_group_per_suite else f"{suite}-group-{index // 2}"
            entries.append({"episode_id": episode_id, "suite": suite, "content_group_id": group,
                            "split": "train", "queries": 2})
            queries[episode_id] = [{"episode_id": episode_id, "timestep": t,
                                    "instruction": suite, "group": group} for t in (0, 8)]

    def episodes(split, *, order):
        assert split == "train"
        lookup = {entry["episode_id"]: entry for entry in entries}
        for episode_id in order:
            yield lookup[episode_id], [sample.copy() for sample in queries[episode_id]]

    data = SimpleNamespace(entries=entries, episodes=episodes)
    options = dict(seed=17, repeat=True, sampling="suite_balanced", suite_map={})
    total = query_count(data)
    legacy = list(islice(query_stream(data, "train", **options), 2 * total))
    current = list(islice(query_stream(data, "train", fft_update_size=8, **options), 2 * total))
    assert [sample["suite"] for sample, _ in current] == [sample["suite"] for sample, _ in legacy]
    for start in (0, total):
        assert Counter(keys(current[start:start + total])) == Counter(keys(legacy[start:start + total]))
    assert all(history == [sample] for sample, history in current)
    # Each full group of four still contains one Query from each suite.
    for start in (0, total):
        for i in range(start, start + total - 3, 4):
            assert len({sample["suite"] for sample, _ in current[i:i + 4]}) == 4
    def repeated_groups(rows):
        return sum(len(set(sample["group"] for sample, _ in rows[i:i + 8])) < 8
                   for i in range(0, len(rows) - 7, 8))
    if one_group_per_suite:
        assert repeated_groups(current) > 0  # Keep the quota when no alternative exists.
    else:
        assert repeated_groups(current) < repeated_groups(legacy)
    for cursor in (3, total - 1, total + 3):
        resumed = islice(query_stream(data, "train", skip=cursor, fft_update_size=8, **options), 11)
        assert keys(resumed) == keys(current[cursor:cursor + 11])
