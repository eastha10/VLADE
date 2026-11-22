"""Bit-packed mixed-precision inference. No persistent floating weight matrix."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

BITS = (0, 2, 3, 4, 8, 16)


def quantize_rows(weight: torch.Tensor, bits: int, group_size: int,
                  fisher: torch.Tensor | None = None, iterations: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric integer codes and positive block scales; optional ActQuant Eq. 10–11."""
    if bits not in BITS or weight.ndim != 2 or group_size < 1 or iterations < 0:
        raise ValueError('Invalid quantization shape, bits or block size')
    w = weight.detach().float()
    if not torch.isfinite(w).all():
        raise ValueError('Non-finite weights')
    rows, width = w.shape
    groups = math.ceil(width / group_size)
    if bits in (0, 16):
        codes = (torch.zeros_like(w, dtype=torch.int32) if bits == 0 else
                 w.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32).bitwise_and(65535))
        return codes, torch.ones((rows, groups), device=w.device)
    limit = 2 ** (bits - 1) - 1
    padded = F.pad(w, (0, groups * group_size - width)).reshape(rows, groups, group_size)
    scale = (padded.abs().amax(-1) / limit).clamp_min(1e-12)
    if fisher is not None:
        if fisher.shape != weight.shape or not torch.isfinite(fisher).all() or (fisher < 0).any():
            raise ValueError('Invalid Action-Mixed Fisher')
        f = F.pad(fisher.to(w.device).float(), (0, groups * group_size - width)).reshape_as(padded)
        omega = f * (padded.square().mean(-1, keepdim=True) + padded.square()).sqrt()
        # Padding has zero curvature and cannot change the fitted objective.
        best = scale.clone()
        best_error = torch.full_like(scale, float('inf'))
        for _ in range(iterations + 1):
            q = (padded / scale[..., None]).round().clamp(-limit, limit)
            error = (omega * (padded - scale[..., None] * q).square()).sum(-1)
            better = error < best_error
            best = torch.where(better, scale, best)
            best_error = torch.minimum(best_error, error)
            denominator = (omega * q.square()).sum(-1)
            fitted = (omega * padded * q).sum(-1) / denominator.clamp_min(1e-30)
            scale = torch.where(denominator > 0, fitted.clamp_min(1e-12), scale)
        scale = best
    q = (padded / scale[..., None]).round().clamp(-limit, limit)
    return (q.reshape(rows, -1)[:, :width] + limit).to(torch.int32), scale


def dequantize_rows(codes: torch.Tensor, scales: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    if bits == 0:
        return torch.zeros_like(codes, dtype=torch.float32)
    if bits == 16:
        return codes.to(torch.int16).contiguous().view(torch.bfloat16).float()
    index = torch.arange(codes.shape[1], device=codes.device) // group_size
    return (codes.float() - (2 ** (bits - 1) - 1)) * scales[:, index]


class PackedLinear(nn.Module):
    """Variable bits per output row; CUDA reads packed bytes directly in registers.

    CPU uses bounded output-row tiles. Inference never replaces this module with
    a dense floating Linear or caches a dequantized full weight matrix.
    """
    def __init__(self, in_features: int, out_features: int, group_size: int,
                 *, backend: str = 'auto') -> None:
        super().__init__()
        if (any(type(v) is not int or v < 1 for v in (in_features, out_features, group_size))
                or backend not in ('auto', 'cuda', 'portable')):
            raise ValueError('Invalid packed linear dimensions/backend')
        self.in_features, self.out_features, self.group_size = in_features, out_features, group_size
        self.backend = backend
        for name, value in {
            'codes': torch.empty(0, dtype=torch.uint8), 'bits': torch.empty(0, dtype=torch.uint8),
            'offsets': torch.empty(0, dtype=torch.int64), 'scales': torch.empty(0, dtype=torch.float32),
            'bias': torch.empty(0, dtype=torch.float32), 'input_scale': torch.empty(0, dtype=torch.float32),
            'permutation': torch.empty(0, dtype=torch.int64), 'rotation1': torch.empty(0, dtype=torch.float32),
            'rotation2': torch.empty(0, dtype=torch.float32),
        }.items():
            self.register_buffer(name, value)

    @classmethod
    def from_linear(cls, layer: nn.Linear, row_bits: int | list[int], *, group_size: int = 32,
                    fisher: torch.Tensor | None = None, backend: str = 'auto',
                    transformed_weight: torch.Tensor | None = None) -> 'PackedLinear':
        bits = [row_bits] * layer.out_features if isinstance(row_bits, int) else list(row_bits)
        if len(bits) != layer.out_features or any(type(b) is not int or b not in BITS for b in bits):
            raise ValueError('A supported precision is required for every output row')
        module = cls(layer.in_features, layer.out_features, group_size, backend=backend)
        weight = layer.weight.detach().cpu() if transformed_weight is None else transformed_weight.detach().cpu()
        if weight.shape != layer.weight.shape:
            raise ValueError('Transformed weight shape differs')
        module.bits = torch.tensor(bits, dtype=torch.uint8)
        widths = (module.bits.long() * layer.in_features + 7) // 8
        module.offsets = torch.cat((torch.zeros(1, dtype=torch.int64), widths.cumsum(0)))
        module.codes = torch.zeros(int(module.offsets[-1]), dtype=torch.uint8)
        module.scales = torch.ones((layer.out_features, math.ceil(layer.in_features / group_size)))
        module.bias = (layer.bias.detach().float().cpu().clone() if layer.bias is not None else torch.zeros(layer.out_features))
        for b in sorted(set(bits)):
            selected = (module.bits == b).nonzero().flatten()
            if b == 0:
                module.bias[selected] = 0
                continue
            q, scale = quantize_rows(weight[selected], b, group_size,
                                     None if fisher is None else fisher.detach().cpu()[selected])
            module.scales[selected] = scale
            packed = torch.zeros((len(selected), int(widths[selected[0]])), dtype=torch.int32)
            columns = torch.arange(layer.in_features)
            for bit in range(b):
                position = columns * b + bit
                packed.scatter_add_(1, (position // 8).expand(len(selected), -1),
                                    (((q >> bit) & 1) << (position % 8)).to(torch.int32))
            indices = module.offsets[selected, None] + torch.arange(packed.shape[1])
            module.codes[indices] = packed.to(torch.uint8)
        return module.to(device=layer.weight.device)

    def _apply(self, fn, recurse=True):
        # Model.to(dtype=...) must not truncate scales, transform matrices or integer metadata.
        for name, value in self._buffers.items():
            transformed = fn(value)
            self._buffers[name] = value.to(device=transformed.device) if transformed.dtype != value.dtype else transformed
        return self

    def validate(self) -> None:
        if (self.bits.shape != (self.out_features,) or self.bits.dtype != torch.uint8
                or self.offsets.shape != (self.out_features + 1,) or self.offsets.dtype != torch.int64
                or self.codes.ndim != 1 or self.codes.dtype != torch.uint8
                or self.scales.shape != (self.out_features, math.ceil(self.in_features / self.group_size))
                or self.scales.dtype != torch.float32 or self.bias.shape != (self.out_features,)
                or self.bias.dtype != torch.float32):
            raise ValueError('Malformed packed tensors')
        if not all(int(b) in BITS for b in self.bits.cpu()) or int(self.offsets[0]) != 0:
            raise ValueError('Invalid packed precision/offset')
        widths = (self.bits.long() * self.in_features + 7) // 8
        if not torch.equal(self.offsets[1:] - self.offsets[:-1], widths) or int(self.offsets[-1]) != self.codes.numel():
            raise ValueError('Packed buffer size/offset mismatch')
        if not torch.isfinite(self.scales).all() or (self.scales <= 0).any() or not torch.isfinite(self.bias).all():
            raise ValueError('Invalid packed scales/bias')
        if self.input_scale.numel():
            if self.input_scale.shape != (self.in_features,) or (self.input_scale <= 0).any() or not torch.isfinite(self.input_scale).all():
                raise ValueError('Invalid input transform scale')
            if sorted(self.permutation.cpu().tolist()) != list(range(self.in_features)):
                raise ValueError('Invalid channel permutation')
            if self.rotation1.ndim != 3 or self.rotation1.shape != self.rotation2.shape:
                raise ValueError('Invalid dual rotations')
            for r in (self.rotation1, self.rotation2):
                if r.shape[0] * r.shape[1] != self.in_features or r.shape[1] != r.shape[2] or not torch.isfinite(r).all():
                    raise ValueError('Rotation dimensions differ')
                eye = torch.eye(r.shape[1], device=r.device)
                with torch.autocast(r.device.type, enabled=False):
                    if not torch.allclose(r.transpose(-1, -2) @ r, eye.expand_as(r), atol=1e-4, rtol=1e-4):
                        raise ValueError('Non-orthogonal transform')
        elif any(t.numel() for t in (self.permutation, self.rotation1, self.rotation2)):
            raise ValueError('Incomplete transform')

    def dequantized_rows(self, start: int, end: int) -> torch.Tensor:
        """Diagnostic/CPU tile decoder, not a persistent weight property."""
        values = []
        columns = torch.arange(self.in_features, device=self.codes.device)
        for row in range(start, end):
            b = int(self.bits[row])
            if b == 0:
                values.append(torch.zeros(self.in_features, device=self.codes.device))
                continue
            position = columns * b
            q = torch.zeros(self.in_features, dtype=torch.int32, device=self.codes.device)
            for bit in range(b):
                p = position + bit
                byte = self.codes[self.offsets[row] + p // 8].int()
                q |= ((byte >> (p % 8)) & 1) << bit
            values.append(dequantize_rows(q[None], self.scales[row:row+1], b, self.group_size)[0])
        return torch.stack(values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError('Packed Linear input width/dtype mismatch')
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError('Packed deployment modules are inference-only')
        shape, dtype = x.shape[:-1], x.dtype
        x = x.reshape(-1, self.in_features).float()
        if self.input_scale.numel():
            from .transforms import block_rotate
            x = block_rotate(x / self.input_scale, self.rotation1)
            x = block_rotate(x[:, self.permutation], self.rotation2)
        if not x.numel():
            return x.new_empty((*shape, self.out_features), dtype=dtype)
        if self.backend == 'cuda' or (self.backend == 'auto' and x.is_cuda):
            from .cuda import linear
            output = linear(x, self)
        else:
            output = torch.empty((len(x), self.out_features), device=x.device)
            for start in range(0, self.out_features, 16):
                end = min(start + 16, self.out_features)
                output[:, start:end] = F.linear(x, self.dequantized_rows(start, end), self.bias[start:end])
        return output.reshape(*shape, self.out_features).to(dtype)
