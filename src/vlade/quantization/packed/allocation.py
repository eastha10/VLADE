"""ActQuant layer-balanced upgrades and QVLA guarded channel demotions."""
from __future__ import annotations
import math
import re
import torch


def actquant(sensitivity: dict[str, float], sizes: dict[str, int], target: float,
             choices: tuple[int, ...] = (2, 3, 4, 8, 16)) -> dict[str, int]:
    """ActQuant v4 Eq. 6: minimize sum of squared transformer-layer errors."""
    if (not sizes or set(sensitivity) != set(sizes) or any(n <= 0 for n in sizes.values())
            or not math.isfinite(target) or not choices or tuple(sorted(set(choices))) != choices
            or any(b not in (2, 3, 4, 8, 16) for b in choices) or not choices[0] <= target <= choices[-1]
            or any(not math.isfinite(s) or s < 0 for s in sensitivity.values())):
        raise ValueError('Invalid ActQuant allocation problem')
    def group(name):
        match = re.match(r'^(.*?(?:layers|blocks)\.\d+)(?:\.|$)', name)
        return match.group(1) if match else name.rpartition('.')[0]
    groups = {n: group(n) for n in sizes}
    selected = dict.fromkeys(sizes, 0)
    used = choices[0] * sum(sizes.values())
    budget = math.floor(target * sum(sizes.values()))
    while True:
        errors = {}
        for n in sizes:
            g = groups[n]
            errors[g] = errors.get(g, 0.) + sensitivity[n] * 2. ** (-2 * choices[selected[n]])
        candidates = []
        for n, size in sizes.items():
            j = selected[n]
            if j + 1 == len(choices):
                continue
            cost = (choices[j+1] - choices[j]) * size
            if used + cost > budget:
                continue
            e = errors[groups[n]]
            improvement = sensitivity[n] * (2. ** (-2*choices[j]) - 2. ** (-2*choices[j+1]))
            gain = (e*e - (e-improvement)**2) / cost
            candidates.append((gain, -size, n, cost))
        if not candidates:
            break
        # Zero-sensitivity ties may consume spare budget without changing Eq. 6.
        _, _, name, cost = max(candidates)
        selected[name] += 1
        used += cost
    return {name: choices[j] for name, j in selected.items()}


def qvla(costs: torch.Tensor, sizes: list[int], target: float, *, metric: str = 'parameters',
         max_prune_fraction: float = .1, prune_threshold: float = 1e-5) -> list[int]:
    """QVLA Eq. 7–8, with explicit L0 cap and an absolute 0-bit action-cost gate.

    Cost columns are [0,2,4,8,16]. 'channels' matches the paper's Eq. 7;
    'parameters' makes heterogeneous matrix widths share a weight-bit budget.
    """
    if (costs.shape != (len(sizes), 5) or not sizes or any(n <= 0 for n in sizes)
            or not torch.isfinite(costs).all() or (costs < 0).any() or not math.isfinite(target)
            or not 0 <= target <= 16 or metric not in ('parameters', 'channels')
            or not 0 <= max_prune_fraction <= 1 or not math.isfinite(prune_threshold) or prune_threshold < 0):
        raise ValueError('Invalid QVLA allocation problem')
    widths = sizes if metric == 'parameters' else [1] * len(sizes)
    selected = [16] * len(sizes)
    current, budget = 16 * sum(widths), math.floor(target * sum(widths))
    count, cap = 0, math.floor(len(sizes) * max_prune_fraction)
    columns = {b: i for i, b in enumerate((0, 2, 4, 8, 16))}
    for high, low in ((16, 8), (8, 4), (4, 2), (2, 0)):
        order = sorted(range(len(sizes)), key=lambda i: (
            float(costs[i, columns[low]] - costs[i, columns[high]]) / ((high-low) * widths[i]), i))
        for i in order:
            if current <= budget:
                return selected
            if selected[i] != high:
                continue
            if low == 0 and (count >= cap or float(costs[i, 0]) > prune_threshold):
                continue
            selected[i] = low
            current -= (high-low) * widths[i]
            count += low == 0
    if current > budget:
        raise ValueError('Requested QVLA budget is infeasible under the pruning guards')
    return selected
