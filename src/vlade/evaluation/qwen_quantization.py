"""Experimental QwenVLA fake quantization for action-quality comparisons.

Round and dequantize weights in memory. Deployment requires packed weights
and a separate inference path that can execute them.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Callable

import torch
from torch import nn

BITS = (2, 4, 8, 16)


def _codes(weight: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    limit = 2 ** (bits - 1) - 1
    return (weight / scale).round().clamp(-limit, limit)


def _scale(weight: torch.Tensor, bits: int) -> torch.Tensor:
    return (weight.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 1)).clamp_min(1e-12)


def round_weights(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric quantize/dequantize over the last axis."""
    if bits == 16:
        return weight
    scale = _scale(weight, bits)
    return _codes(weight, scale, bits) * scale


def _rbf(samples: list[torch.Tensor]) -> torch.Tensor:
    distance = torch.cdist(torch.stack(samples), torch.stack(samples)).square()
    positive = distance[distance > 0]
    bandwidth = positive.median() if positive.numel() else distance.new_tensor(1.0)
    kernel = torch.exp(-distance / (2 * bandwidth))
    return kernel - kernel.mean(0) - kernel.mean(1, keepdim=True) + kernel.mean()


def _allocate(costs: torch.Tensor, sizes: list[int], target: float) -> list[int]:
    """Promote the units with greatest marginal error reduction per added bit."""
    chosen = [0] * len(sizes)
    budget = math.floor(target * sum(sizes)) - BITS[0] * sum(sizes)
    choices = []

    def enqueue(unit: int) -> None:
        level = chosen[unit]
        if level + 1 < len(BITS):
            delta = sizes[unit] * (BITS[level + 1] - BITS[level])
            benefit = float(costs[unit, level] - costs[unit, level + 1])
            if benefit > 0:
                heapq.heappush(choices, (-benefit / delta, unit, delta))

    for unit in range(len(sizes)):
        enqueue(unit)
    while choices:
        _, unit, delta = heapq.heappop(choices)
        if delta <= budget:
            budget -= delta
            chosen[unit] += 1
            enqueue(unit)
    return [BITS[level] for level in chosen]


def _demote(costs: torch.Tensor, sizes: list[int], target: float) -> list[int]:
    """Demote rows stage by stage, ranking by estimated action error per saved bit."""
    chosen = [16] * len(sizes)
    remaining = 16 * sum(sizes)
    budget = math.floor(target * sum(sizes))
    for low, high in reversed(list(zip(BITS, BITS[1:]))):
        marginal = costs[:, BITS.index(low)] - costs[:, BITS.index(high)]
        values = marginal.tolist()
        for row in sorted(range(len(sizes)), key=lambda i: values[i] / sizes[i]):
            if remaining <= budget:
                return chosen
            chosen[row] = low
            remaining -= sizes[row] * (high - low)
    return chosen


def _fisher_round(weight: torch.Tensor, fisher: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 16:
        return weight
    flat = weight.flatten()
    padding = (-flat.numel()) % 32
    blocks = nn.functional.pad(flat, (0, padding)).reshape(-1, 32)
    curvature = nn.functional.pad(fisher.flatten().to(weight.device), (0, padding)).reshape_as(blocks)
    importance = curvature * (blocks.square().mean(-1, keepdim=True) + blocks.square()).sqrt()
    scale = _scale(blocks, bits)
    for _ in range(4):
        codes = _codes(blocks, scale, bits)
        denominator = (importance * codes.square()).sum(-1, keepdim=True)
        fitted = (importance * blocks * codes).sum(-1, keepdim=True) / denominator.clamp_min(1e-30)
        scale = torch.where(denominator > 0, fitted.clamp_min(1e-12), scale)
    return (_codes(blocks, scale, bits) * scale).flatten()[:flat.numel()].reshape_as(weight)


def calibrate(
    mode: str,
    layers: dict[str, nn.Linear],
    run_samples: Callable,
    target_bpw: float,
    seed: int,
) -> tuple[dict, list]:
    """Calibrate and apply one pilot method. Return report fields and live hook handles."""
    layer_items = list(layers.items())
    fisher, inputs, outputs, targets, maxima, powers, gains = {}, {}, {}, [], {}, {}, {}
    handles = []
    random = torch.Generator(device="cpu").manual_seed(seed)

    def observe(name):
        def hook(_module, args, output):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            if mode == "actquant":
                inputs[name].append(x.mean(0).cpu())
                outputs[name].append(output.detach().float().reshape(-1, output.shape[-1]).mean(0).cpu())
            elif mode == "quant":
                maxima[name] = torch.maximum(maxima[name], x.abs().amax(0))
            else:
                powers[name] += x.square().mean(0)
                if output.requires_grad:
                    def track(gradient):
                        g = gradient.detach().float().reshape(-1, gradient.shape[-1])
                        gains[name] += g.square().sum(0)
                    output.register_hook(track)
        return hook

    for name, layer in layer_items:
        device = layer.weight.device
        inputs[name], outputs[name] = [], []
        maxima[name] = torch.zeros(layer.in_features, device=device)
        powers[name] = torch.zeros_like(maxima[name])
        gains[name] = torch.zeros(layer.out_features, device=device)
        if mode == "actquant":
            fisher[name] = torch.zeros(layer.weight.shape, dtype=torch.float32)
        handles.append(layer.register_forward_hook(observe(name)))

    def collect(actions, target, valid):
        if mode == "actquant":
            targets.append(torch.where(valid, target, 0).flatten().cpu())
            loss = (actions.float() - target)[valid].abs().mean()
            grads = torch.autograd.grad(loss, [layer.weight for _, layer in layer_items], allow_unused=True)
            for (name, _), grad in zip(layer_items, grads):
                if grad is not None:
                    fisher[name].add_(grad.detach().cpu().float().square())
        elif mode == "qvla":
            # Rademacher probe estimates action Jacobian energy through one reverse pass.
            sign = torch.randint(0, 2, actions.shape, generator=random).to(actions.device) * 2 - 1
            probe = (actions.float() * sign)[valid].sum()
            torch.autograd.grad(probe, [layer.weight for _, layer in layer_items], allow_unused=True)

    try:
        count = run_samples(collect, gradients=mode != "quant")
    finally:
        for handle in handles:
            handle.remove()

    active = []
    with torch.no_grad():
        if mode == "actquant":
            if count < 3:
                raise ValueError("ActQuant HSIC needs at least three calibration queries")
            action_kernel = _rbf(targets)
            sensitivity = []
            for name, _ in layer_items:
                if len(inputs[name]) != count:
                    raise ValueError(f"Expected one call per calibration query for {name}")
                observed = _rbf(outputs[name])
                score = float((observed * (action_kernel - _rbf(inputs[name]))).sum() / (count - 1) ** 2)
                sensitivity.append(max(0.0, score))
            width_cost = torch.tensor([2.0 ** (-2 * bit) for bit in BITS])
            sizes = [layer.weight.numel() for _, layer in layer_items]
            bits = _allocate(torch.tensor(sensitivity)[:, None] * width_cost, sizes, target_bpw)
            for (name, layer), bit in zip(layer_items, bits):
                layer.weight.copy_(_fisher_round(layer.weight.float(), fisher[name], bit))
            unit_bits = {name: bit for (name, _), bit in zip(layer_items, bits)}
        elif mode == "qvla":
            costs, sizes = [], []
            for name, layer in layer_items:
                weight = layer.weight.float()
                errors = [(round_weights(weight, bit) - weight).square() @ powers[name] for bit in BITS]
                costs.append((torch.stack(errors, dim=1) * gains[name][:, None]).cpu())
                sizes.extend([layer.in_features] * layer.out_features)
            bits = _demote(torch.cat(costs), sizes, target_bpw)
            start, unit_bits = 0, {}
            for name, layer in layer_items:
                assigned = torch.tensor(bits[start:start + layer.out_features], device=layer.weight.device)
                for bit in BITS[:-1]:
                    mask = assigned == bit
                    layer.weight[mask] = round_weights(layer.weight[mask].float(), bit).to(layer.weight.dtype)
                unit_bits[name] = {str(bit): int((assigned == bit).sum()) for bit in BITS}
                start += layer.out_features
        else:
            bits, sizes, unit_bits = [], [], {}
            for name, layer in layer_items:
                weight = layer.weight.float()
                scale = (maxima[name].clamp_min(1e-5) / weight.abs().amax(0).clamp_min(1e-5)).sqrt()
                scale = scale.clamp(1e-2, 1e2)
                layer.weight.copy_(round_weights(weight * scale, 8))

                def unscale(_module, args, factor=scale):
                    return ((args[0].float() / factor).to(args[0].dtype), *args[1:])

                active.append(layer.register_forward_pre_hook(unscale))
                bits.append(8)
                sizes.append(layer.weight.numel())
                unit_bits[name] = 8

    return {
        "calibration_queries": count,
        "weight_bpw_excluding_metadata": sum(bit * size for bit, size in zip(bits, sizes)) / sum(sizes),
        "quantized_scope_parameters": sum(sizes),
        "bit_allocation": unit_bits,
    }, active
