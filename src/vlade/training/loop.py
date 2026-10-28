"""Single-device optimizer loop, independent of the Teacher/Student architecture.

Loss callbacks return a scalar mean for equal-sized microbatches. Checkpoints are
called only at optimizer boundaries, so no partial gradients need serializing.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch


@dataclass
class LoopState:
    step: int = 0
    microsteps: int = 0


def capture_rng() -> dict[str, Any]:
    """Use only tensors and primitive containers for weights_only loading."""
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def make_scheduler(optimizer: Any, max_steps: int, warmup_steps: int = 0,
                   schedule: str = "constant") -> Any:
    """LR for update i is configured before that update, in optimizer steps."""
    if max_steps < 1 or not 0 <= warmup_steps < max_steps:
        raise ValueError("Require max_steps > warmup_steps >= 0")
    if schedule not in {"constant", "cosine"}:
        raise ValueError("schedule must be constant or cosine")

    def factor(index: int) -> float:
        if index < warmup_steps:
            return (index + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        progress = (index - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluate(modules: Sequence[Any], batches: Iterable[Any],
             loss_fn: Callable[[Any, int], Any], step: int, max_batches: int) -> float:
    """Bounded mean loss; restore individual modes and host/PyTorch RNG."""
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    modes = [(child, child.training) for module in modules for child in module.modules()]
    rng = capture_rng()
    total, count = 0.0, 0
    try:
        for module in modules:
            module.eval()
        iterator = iter(batches)
        with torch.no_grad():
            for _ in range(max_batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                loss = loss_fn(batch, step)
                if loss.ndim != 0 or not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite or non-scalar validation loss")
                total += loss.item()
                count += 1
        if count == 0:
            raise RuntimeError("Validation loader is empty")
        return total / count
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)


def run_loop(*, modules: Sequence[Any], optimizer: Any, scheduler: Any,
             batches: Iterable[Any], loss_fn: Callable[[Any, int], Any],
             max_steps: int, accumulation_steps: int = 1, max_grad_norm: float = 1.0,
             state: LoopState | None = None,
             on_before_step: Callable[[LoopState], None] | None = None,
             on_step: Callable[[LoopState, dict[str, float]], None] | None = None) -> LoopState:
    """Run complete accumulation windows; reject truncated streams and NaN/Inf.

    The caller supplies the stream position. RNG restoration alone cannot resume
    a shuffled RLDS/TFDS iterator exactly.
    """
    state = state if state is not None else LoopState()
    if max_steps < 1 or accumulation_steps < 1 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("Invalid loop settings")
    if not 0 <= state.step <= max_steps or state.microsteps != state.step * accumulation_steps:
        raise ValueError("Resume state is not an optimizer boundary")
    params = list({id(p): p for module in modules for p in module.parameters() if p.requires_grad}.values())
    if not params:
        raise ValueError("No trainable parameters")
    for module in modules:
        module.train()
    optimizer.zero_grad(set_to_none=True)
    if state.step == max_steps:
        return state
    iterator = iter(batches)
    while state.step < max_steps:
        total = 0.0
        try:
            for _ in range(accumulation_steps):
                batch = next(iterator)
                loss = loss_fn(batch, state.step)
                if loss.ndim != 0 or not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Non-finite or non-scalar loss at step {state.step}")
                (loss / accumulation_steps).backward()
                total += loss.detach().float().item()
            # error_if_nonfinite guards gradients even when the scalar loss is finite.
            norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
            if on_before_step is not None:
                on_before_step(state)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        except StopIteration as exc:
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"Data stream ended before optimizer step {state.step + 1}") from exc
        except BaseException:
            optimizer.zero_grad(set_to_none=True)
            raise
        state.step += 1
        state.microsteps += accumulation_steps
        if on_step is not None:
            on_step(state, {"loss": total / accumulation_steps, "lr": float(lr),
                            "grad_norm": float(norm)})
    return state


def training_state(state: LoopState, optimizer: Any, scheduler: Any) -> dict[str, Any]:
    """Model/component weights are stored separately by the model adapter."""
    return {"schema_version": 1, "loop": asdict(state), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": capture_rng(),
            "data_stream_resume": "restart_not_bitwise_exact"}


def restore_training_state(payload: dict[str, Any], optimizer: Any, scheduler: Any) -> LoopState:
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported trainer state schema")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return LoopState(**payload["loop"])
