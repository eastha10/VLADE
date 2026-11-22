"""Action-guided PTQ recipes producing deployment modules, not rounded Linears."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any
import torch
from torch import nn

from . import allocation
from .linear import PackedLinear, quantize_rows, dequantize_rows
from .transforms import dual_transform


@dataclass
class Sample:
    args: tuple
    target: torch.Tensor
    valid: torch.Tensor
    kwargs: dict[str, Any] = field(default_factory=dict)


def action(output: Any) -> torch.Tensor:
    value = output if isinstance(output, torch.Tensor) else output.actions
    if not torch.isfinite(value).all():
        raise FloatingPointError('Non-finite calibration action')
    return value


def _kernel(values):
    distance = torch.cdist(torch.stack(values).double(), torch.stack(values).double()).square()
    positive = distance[distance > 0]
    bandwidth = positive.median() if positive.numel() else distance.new_tensor(1.)
    result = torch.exp(-distance / (2 * bandwidth))
    return result - result.mean(0) - result.mean(1, keepdim=True) + result.mean()


def _statistics(model, layers, samples, method, probes, seed):
    inputs, outputs, maxima, powers, gains, fisher, calls = {}, {}, {}, {}, {}, {}, {}
    for name, layer in layers.items():
        inputs[name], outputs[name] = [], []
        maxima[name], powers[name] = torch.zeros(layer.in_features), torch.zeros(layer.in_features)
        gains[name], calls[name] = torch.zeros(layer.out_features), 0
        if method == 'actquant':
            fisher[name] = torch.zeros_like(layer.weight, device='cpu', dtype=torch.float32)
    handles, current = [], {}
    def hook(name):
        def observe(_module, args, output):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            current.setdefault(name, []).append((x.mean(0).cpu(), output.detach().float().reshape(-1, output.shape[-1]).mean(0).cpu()))
            maxima[name] = torch.maximum(maxima[name], x.abs().amax(0).cpu())
            powers[name] += x.square().mean(0).cpu()
            calls[name] += 1
            if method == 'qvla' and output.requires_grad:
                def track(gradient):
                    gains[name].add_(gradient.detach().float().reshape(-1, gradient.shape[-1]).square().sum(0).cpu() / probes)
                output.register_hook(track)
        return observe
    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(hook(name)))
    generator = torch.Generator().manual_seed(seed)
    weights = [layer.weight for layer in layers.values()]
    try:
        for sample in samples:
            current.clear()
            with torch.set_grad_enabled(method != 'quantvla'):
                value = action(model(*sample.args, **sample.kwargs))
                if (value.shape != sample.target.shape or sample.valid.shape != value.shape
                        or sample.valid.dtype != torch.bool or not sample.valid.any()
                        or not sample.target.is_floating_point() or not torch.isfinite(sample.target[sample.valid]).all()):
                    raise ValueError('Calibration target/valid-mask layout differs from action')
                if method == 'actquant':
                    loss = (value.float() - sample.target.float())[sample.valid].abs().mean()
                    gradients = torch.autograd.grad(loss, weights, allow_unused=True)
                    for name, gradient in zip(layers, gradients):
                        if gradient is not None:
                            fisher[name] += gradient.detach().float().cpu().square() / len(samples)
                elif method == 'qvla':
                    for probe in range(probes):
                        sign = torch.randint(0, 2, value.shape, generator=generator).to(value.device) * 2 - 1
                        objective = (value.float() * sign)[sample.valid].sum()
                        torch.autograd.grad(objective, weights, retain_graph=probe+1 < probes, allow_unused=True)
            for name in layers:
                if name not in current:
                    raise ValueError('Selected layer was not observed: ' + name)
                inputs[name].append(torch.stack([p[0] for p in current[name]]).mean(0))
                outputs[name].append(torch.stack([p[1] for p in current[name]]).mean(0))
    finally:
        for handle in handles:
            handle.remove()
    for name in layers:
        powers[name] /= calls[name]
        gains[name] /= calls[name]
    return inputs, outputs, maxima, powers, gains, fisher


def calibrate(model: nn.Module, layers: dict[str, nn.Linear], samples: list[Sample], *, method: str,
              target_bpw: float, group_size: int = 32, backend: str = 'auto', seed: int = 42,
              lambda_redundancy: float = 1., lambda_action: float = 1., jacobian_probes: int = 2,
              refine_channels: int = 32, budget_metric: str = 'parameters', max_prune_fraction: float = .1,
              prune_threshold: float = 1e-5, rotation_block_size: int = 64) -> tuple[dict[str, PackedLinear], dict]:
    """Qwen's continuous L1 action path uses Action-Only Fisher (AMF alpha=1).

    QVLA refines the most sensitive proxy channels with actual masked Action-MSE.
    Each refinement restores the original row/bias and uses immutable sample context.
    QuantVLA here is the non-DiT W8A16 route of Appendix G; no fictitious DiT ATM/OHB.
    """
    if method not in ('actquant', 'qvla', 'quantvla') or not layers or not samples:
        raise ValueError('Require a method, target layers and replayable calibration samples')
    if method == 'actquant' and len(samples) < 3:
        raise ValueError('ActQuant HSIC requires at least three calibration queries')
    if method == 'quantvla' and target_bpw != 8:
        raise ValueError('Non-DiT QuantVLA supports its documented W8A16 route')
    if (group_size < 1 or jacobian_probes < 1 or refine_channels < 0
            or any(not math.isfinite(v) or v < 0 for v in (lambda_redundancy,lambda_action))):
        raise ValueError('Invalid calibration hyperparameters')
    if any(not isinstance(layer,nn.Linear) or layer.weight.dtype != torch.bfloat16 for layer in layers.values()):
        raise ValueError('Deployment recipes require BF16 source Linear weights')
    if len({id(layer.weight) for layer in layers.values()}) != len(layers):
        raise ValueError('Aliased target weights require an explicit architecture adapter')
    flags = {p: p.requires_grad for p in model.parameters()}
    training = {m: m.training for m in model.modules()}
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers.values():
        layer.weight.requires_grad_(method != 'quantvla')
    try:
        stats = _statistics(model, layers, samples, method, jacobian_probes, seed)
        inputs, outputs, maxima, powers, gains, fisher = stats
        scores, cost_parts, sizes, names = {}, [], [], []
        if method == 'actquant':
            labels = [torch.where(s.valid, s.target.detach(), 0).float().flatten().cpu() for s in samples]
            target_kernel = _kernel(labels)
            for name in layers:
                z = _kernel(outputs[name])
                raw = float((z * (lambda_action * target_kernel - lambda_redundancy * _kernel(inputs[name]))).sum() / (len(samples)-1)**2)
                scores[name] = {'raw_hsic': raw, 'sensitivity': max(0., raw)}
            assignment = allocation.actquant({n: v['sensitivity'] for n, v in scores.items()},
                                             {n: layer.weight.numel() for n, layer in layers.items()}, target_bpw)
        elif method == 'qvla':
            with torch.no_grad():
                references = [action(model(*s.args, **s.kwargs)).detach().float().cpu() for s in samples]
                for name, layer in layers.items():
                    w = layer.weight.detach().float().cpu()
                    columns = []
                    for b in (0, 2, 4, 8, 16):
                        q, scale = quantize_rows(w, b, layer.in_features)
                        error = dequantize_rows(q, scale, b, layer.in_features) - w
                        columns.append((error.square() @ powers[name]) * gains[name])
                    part = torch.stack(columns, 1)
                    part[:, -1] = 0
                    cost_parts.append(part)
                    sizes.extend([layer.in_features] * layer.out_features)
                    names.extend((name, row) for row in range(layer.out_features))
                costs = torch.cat(cost_parts)
                important = costs[:, 1].argsort(descending=True, stable=True)[:refine_channels]
                for index in important.tolist():
                    name, row = names[index]
                    layer = layers[name]
                    original = layer.weight[row].detach().clone()
                    bias = None if layer.bias is None else layer.bias[row].detach().clone()
                    try:
                        for j, b in enumerate((0, 2, 4, 8)):
                            q, scale = quantize_rows(original[None], b, layer.in_features)
                            layer.weight[row].copy_(dequantize_rows(q, scale, b, layer.in_features)[0])
                            if bias is not None:
                                if b == 0:
                                    layer.bias[row].zero_()
                                else:
                                    layer.bias[row].copy_(bias)
                            error = 0.
                            for sample, reference in zip(samples, references):
                                value = action(model(*sample.args, **sample.kwargs)).float().cpu()
                                error += float((value-reference)[sample.valid.cpu()].square().sum()) / len(samples)
                            costs[index, j] = error
                    finally:
                        layer.weight[row].copy_(original)
                        if bias is not None:
                            layer.bias[row].copy_(bias)
                bits = allocation.qvla(costs, sizes, target_bpw, metric=budget_metric,
                                       max_prune_fraction=max_prune_fraction, prune_threshold=prune_threshold)
            assignment, offset = {}, 0
            for name, layer in layers.items():
                assignment[name] = bits[offset:offset+layer.out_features]
                offset += layer.out_features
            scores = {'refined_channels': len(important), 'jacobian_probes': jacobian_probes}
        else:
            assignment = dict.fromkeys(layers, 8)
        replacements = {}
        for name, layer in layers.items():
            transformed, transform = None, {}
            if method == 'quantvla':
                transformed, transform = dual_transform(layer.weight, maxima[name], block_size=rotation_block_size)
            module = PackedLinear.from_linear(layer, assignment[name],
                group_size=group_size if method == 'actquant' else layer.in_features,
                fisher=fisher.get(name), transformed_weight=transformed, backend=backend)
            for key, value in transform.items():
                setattr(module, key, value.to(layer.weight.device))
            module.validate()
            replacements[name] = module
        parameters = sum(l.weight.numel() for l in layers.values())
        raw_bits = sum(int(m.bits.long().sum()) * m.in_features for m in replacements.values())
        return replacements, {'method': method, 'calibration_queries': len(samples),
            'target_bpw': target_bpw, 'budget_metric': budget_metric if method == 'qvla' else 'parameters',
            'weight_bpw': raw_bits / parameters, 'target_parameters': parameters,
            'packed_code_bytes': sum(m.codes.numel() for m in replacements.values()),
            'buffer_bytes': sum(t.numel()*t.element_size() for m in replacements.values() for t in m.buffers()),
            'allocation': {n: m.bits.cpu().tolist() for n, m in replacements.items()},
            'sensitivity': scores, 'activation_bits': 16,
            'representation': 'packed_unsigned_codes_with_symmetric_offset; BF16 for 16-bit rows',
            'kernel': 'fused_packed_cuda' if backend == 'cuda' else 'auto_cuda_or_tiled_cpu'}
    finally:
        for parameter, flag in flags.items():
            parameter.requires_grad_(flag)
        for module, mode in training.items():
            module.training = mode


def replace(model: nn.Module, replacements: dict[str, PackedLinear]) -> None:
    """Replace only the named modules whose dimensions were calibrated."""
    for name, module in replacements.items():
        original = model.get_submodule(name)
        if not isinstance(original, nn.Linear) or (original.in_features, original.out_features) != (module.in_features, module.out_features):
            raise ValueError('Packed replacement topology differs: ' + name)
    for name, module in replacements.items():
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, leaf, module)
