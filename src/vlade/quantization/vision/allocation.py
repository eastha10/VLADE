"""Parameter-weighted, budget-constrained greedy W4/W8 allocation."""
from __future__ import annotations
import math
import random


def allocate(sizes: dict[str, int], scores: dict, target_avg_bits: float,
             *, random_seed: int | None = None) -> dict[str, int]:
    """Greedy approximation; negative improvements never receive upgrades."""
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes.values()):
        raise ValueError("Require positive group sizes")
    if not math.isfinite(target_avg_bits) or not 4 <= target_avg_bits <= 8:
        raise ValueError("Average bit target must be in [4,8]")
    budget = math.floor(sum(sizes.values()) * target_avg_bits)
    used = 4 * sum(sizes.values())
    result = dict.fromkeys(sizes, 4)
    if random_seed is not None:
        order = sorted(sizes)
        random.Random(random_seed).shuffle(order)
    else:
        gains = {}
        for name, count in sizes.items():
            s4, s8 = scores[name]["4"]["combined"], scores[name]["8"]["combined"]
            if not all(math.isfinite(s) and s >= 0 for s in (s4, s8)):
                raise ValueError("Invalid sensitivity")
            gains[name] = max(0, s4 - s8) / (4 * count)
        order = sorted((n for n in sizes if gains[n] > 0), key=lambda n: (-gains[n], n))
    for name in order:
        cost = 4 * sizes[name]
        if used + cost <= budget:
            result[name] = 8
            used += cost
    return result
