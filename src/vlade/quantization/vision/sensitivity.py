"""One-group perturbations against immutable FP query contexts."""
from __future__ import annotations
from typing import Callable, Sequence, Any
import math
import torch
from .quantizer import VisionQuantizer


def losses(fp: tuple, quantized: tuple, epsilon: float = 1e-8) -> tuple[float, float]:
    """Normalized feature error and mean squared log importance error."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")
    z, w = [t.double() for t in fp]
    q, v = [t.double() for t in quantized]
    if z.shape != q.shape or w.shape != v.shape or not z.numel() or not w.numel():
        raise ValueError("Mismatched/empty sensitivity tensors")
    if not all(torch.isfinite(t).all() for t in (z, w, q, v)) or (w < 0).any() or (v < 0).any():
        raise ValueError("Nonfinite features or invalid importance")
    visual = ((z - q).square().sum() / (z.square().sum() + epsilon)).item()
    ava = (w.clamp_min(epsilon).log() - v.clamp_min(epsilon).log()).square().mean().item()
    return visual, ava


@torch.no_grad()
def measure(quantizer: VisionQuantizer, queries: Sequence[Any], evaluate: Callable,
            *, alpha: float, beta: float, batch_size: int = 1) -> dict:
    """Every perturbation starts from FP; contexts never advance during probes.

    batch_size bounds reference tensors held in memory. Queries retain their
    original text lengths and run separately within each reference batch.
    """
    if not queries or batch_size < 1 or any(not math.isfinite(x) or x < 0 for x in (alpha, beta)):
        raise ValueError("Invalid sensitivity inputs")
    result = {name: {str(b): dict(visual=0., ava=0., combined=0.) for b in (4, 8)}
              for name in quantizer.groups}
    quantizer.restore()
    try:
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            reference = [tuple(t.detach().cpu().clone() for t in evaluate(q)) for q in batch]
            for name in result:
                for bits in (4, 8):
                    with quantizer.temporary({name: bits}):
                        for query, fp in zip(batch, reference):
                            visual, ava = losses(fp, tuple(t.detach().cpu() for t in evaluate(query)))
                            row = result[name][str(bits)]
                            row["visual"] += visual / len(queries)
                            row["ava"] += ava / len(queries)
                            row["combined"] += (alpha * visual + beta * ava) / len(queries)
    finally:
        quantizer.restore()
    return result
