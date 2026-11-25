"""Bounded CPU gradient accumulation with unchanged per-parameter optimizer updates."""
from __future__ import annotations

from contextlib import nullcontext
from collections import defaultdict
from copy import deepcopy
import math
from typing import Any, Iterable

import torch


def restore_paged_optimizer(optimizer, saved: dict) -> None:
    """Restore BNB 0.43.3 moments into paged buffers, one tensor at a time.

    Its ordinary loader moves serialized moments to dense CUDA tensors, losing
    the paging needed by this profile. Keep the original tensor values/dtypes,
    parameter order, quantization maps and per-parameter update counters.
    """
    if not getattr(optimizer, "is_paged", False) or not hasattr(optimizer, "get_state_buffer"):
        raise ValueError("Teacher paging restore requires the pinned paged optimizer")
    current, groups = optimizer.param_groups, saved["param_groups"]
    if len(current) != len(groups) or any(len(a["params"]) != len(b["params"]) for a, b in zip(current, groups)):
        raise ValueError("Saved Teacher optimizer parameter groups differ")
    mapping = {index: parameter for a, b in zip(current, groups) for parameter, index in zip(a["params"], b["params"])}
    if set(saved["state"]) - set(mapping):
        raise ValueError("Unexpected Teacher optimizer state keys")
    restored = defaultdict(dict)
    for index, state in saved["state"].items():
        parameter = mapping[index]
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if key in {"state1", "state2"}:
                    if value.shape != parameter.shape:
                        raise ValueError("Teacher paged moment shape mismatch")
                    buffer = optimizer.get_state_buffer(parameter, dtype=value.dtype)
                    if parameter.device.type == "cuda":
                        torch.cuda.synchronize(parameter.device)
                    buffer.copy_(value)
                    restored[parameter][key] = buffer
                else:
                    restored[parameter][key] = value.to(parameter.device)
            else:
                restored[parameter][key] = deepcopy(value)
    optimizer.state = restored
    for actual, previous in zip(current, groups):
        parameters = actual["params"]
        actual.clear()
        actual.update(deepcopy({key: value for key, value in previous.items() if key != "params"}))
        actual["params"] = parameters
    optimizer.initialized = False


class GradientOverflow(FloatingPointError):
    """Retry the same accumulation window before any optimizer mutation."""


class OffloadedGradients:
    """Retain low-precision accumulated gradients on CPU, stream one update at a time.

    The existing AdamW/PagedAdamW8bit object, parameter ordering, group indexes,
    optimizer state and schedule remain intact. No per-parameter step is taken
    until the entire accumulation window is finite and globally clipped.
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], *, loss_scale: float, max_retries: int = 8) -> None:
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("loss_scale must be finite and positive")
        if type(max_retries) is not int or not 0 <= max_retries <= 16:
            raise ValueError("max_retries must be an integer from 0 to 16")
        self.parameters = list(parameters)
        self.loss_scale = float(loss_scale)
        self.max_retries = max_retries
        self.successful_steps = 0
        self.buffers = {}
        self.ready = False
        self.handles = [p.register_post_accumulate_grad_hook(self._capture) for p in self.parameters]

    def _capture(self, parameter) -> None:
        if parameter.grad is None:
            return
        value = parameter.grad.detach().to("cpu", copy=True)
        if parameter in self.buffers:
            self.buffers[parameter].add_(value)
        else:
            self.buffers[parameter] = value
        parameter.grad = None

    def clear(self) -> None:
        self.buffers.clear()
        self.ready = False
        for parameter in self.parameters:
            parameter.grad = None

    def prepare(self, max_norm: float) -> float:
        if not math.isfinite(max_norm) or max_norm <= 0:
            raise ValueError("Teacher max_norm must be finite and positive")
        if not self.buffers:
            raise RuntimeError("No offloaded gradients reached the Teacher")
        squared = 0.0
        for value in self.buffers.values():
            # One tensor at a time: avoid a full FP32 CPU copy of a 7B gradient.
            unscaled = value.float() / self.loss_scale
            if not bool(torch.isfinite(unscaled).all()):
                raise GradientOverflow("Non-finite scaled Teacher gradient; optimizer was not updated")
            squared += float(torch.linalg.vector_norm(unscaled, dtype=torch.float64)) ** 2
        norm = math.sqrt(squared)
        if not math.isfinite(norm):
            raise GradientOverflow("Non-finite Teacher gradient norm")
        multiplier = min(1.0, max_norm / (norm + 1e-6)) / self.loss_scale
        for value in self.buffers.values():
            # Round only when retaining the original low-precision gradient.
            value.copy_(value.float() * multiplier)
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Clipped Teacher gradient is outside parameter precision range")
        self.ready = True
        return norm

    def reduce_scale(self) -> None:
        self.loss_scale /= 2
        self.successful_steps = 0
        self.clear()
        if self.loss_scale < 2**-24:
            raise FloatingPointError("Teacher FP16 loss scale exhausted")

    def diagnostics(self, groups: dict[str, Any]) -> dict:
        result = {}
        for name, module in groups.items():
            parameters = [p for p in module.parameters() if p.requires_grad]
            grads = [self.buffers[p] for p in parameters if p in self.buffers]
            if (not grads or not any(bool(torch.count_nonzero(g)) for g in grads)
                    or not all(bool(torch.isfinite(g).all()) for g in grads)):
                raise RuntimeError(f"Missing/nonfinite/zero gradient path: {name}")
            result[name] = {"trainable_tensors": len(parameters), "gradient_tensors": len(grads),
                            "finite_nonzero": True}
        return result

    def step(self, optimizer) -> None:
        if not self.ready:
            raise RuntimeError("Validate and clip all Teacher gradients before updating")
        try:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter not in self.buffers:
                        continue
                    parameter.grad = self.buffers.pop(parameter).to(parameter.device)
                    context = torch.cuda.device(parameter.device) if parameter.device.type == "cuda" else nullcontext()
                    with context:
                        # Keep every group and index intact. Only this parameter has
                        # .grad, so the original optimizer updates it exactly once.
                        optimizer.step()
                        if parameter.device.type == "cuda":
                            torch.cuda.synchronize(parameter.device)
                    parameter.grad = None
            self.successful_steps += 1
            # Growth belongs to the saved scaler state, not the experiment config.
            if self.successful_steps % 2000 == 0:
                self.loss_scale *= 2
        finally:
            self.clear()

    def state_dict(self) -> dict:
        if self.buffers or self.ready:
            raise ValueError("Teacher gradient state may only be saved at an optimizer boundary")
        return {"schema_version": 1, "loss_scale": self.loss_scale, "successful_steps": self.successful_steps}

    def load_state_dict(self, value: dict) -> None:
        if (value.get("schema_version") != 1 or type(value.get("loss_scale")) not in (int, float)
                or not math.isfinite(value["loss_scale"]) or value["loss_scale"] <= 0
                or type(value.get("successful_steps")) is not int or value["successful_steps"] < 0):
            raise ValueError("Invalid saved Teacher loss-scaling state")
        self.loss_scale, self.successful_steps = value["loss_scale"], value["successful_steps"]

    def close(self) -> None:
        self.clear()
        for handle in self.handles:
            handle.remove()


def offloaded_window(manager: OffloadedGradients, window: list, loss_fn, backward_fn,
                     step: int, max_norm: float) -> tuple[float, float]:
    """Retry overflow with identical queries/RNG; advance neither cursor nor schedule."""
    from vlade.training.loop import capture_rng, restore_rng

    rng = capture_rng()
    for attempt in range(manager.max_retries + 1):
        if attempt:
            restore_rng(rng)
        manager.clear()
        total = 0.0
        scale = manager.loss_scale / len(window)
        for batch in window:
            loss = backward_fn(batch, step, scale) if backward_fn is not None else loss_fn(batch, step)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite Teacher loss; FP16 forward range cannot be fixed by loss scaling")
            if backward_fn is None:
                (loss * scale).backward()
            total += float(loss.detach())
        try:
            return total, manager.prepare(max_norm)
        except GradientOverflow:
            manager.reduce_scale()
            if attempt == manager.max_retries:
                raise
    raise AssertionError("Unreachable overflow retry state")
