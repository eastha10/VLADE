"""Single-process, autograd-preserving two-device placement for AVA/OFT Teachers."""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Iterator, TYPE_CHECKING
if TYPE_CHECKING:
    import torch


def validate_parallel(options: dict | None) -> None:
    if options is None:
        return
    expected = {"devices", "split_layer", "dtype", "gradient_offload", "reserve_gib",
                "loss_scale", "max_overflow_retries"}
    if not isinstance(options, dict) or set(options) != expected:
        raise ValueError("Teacher parallel settings must declare the supported controls")
    devices = options["devices"]
    if (not isinstance(devices, list) or len(devices) != 2
            or any(not isinstance(d, str) or not d.startswith("cuda:") or not d[5:].isdigit() for d in devices)
            or len({int(d[5:]) for d in devices}) != 2):
        raise ValueError("Teacher parallel requires two distinct explicit CUDA devices")
    if options["dtype"] not in {"float16", "bfloat16"} or type(options["gradient_offload"]) is not bool:
        raise ValueError("Teacher parallel requires float16/bfloat16 and explicit gradient_offload")
    if options["split_layer"] is not None and (type(options["split_layer"]) is not int or options["split_layer"] < 1):
        raise ValueError("split_layer must be null (balanced) or a positive decoder boundary")
    for key in ("reserve_gib", "loss_scale"):
        if type(options[key]) not in (int, float) or not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"Invalid Teacher parallel {key}")
    if type(options["max_overflow_retries"]) is not int or not 0 <= options["max_overflow_retries"] <= 16:
        raise ValueError("Invalid max_overflow_retries")


def move_inputs(value: Any, device: Any) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_inputs(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_inputs(item, device) for item in value)
    if isinstance(value, list):
        return [move_inputs(item, device) for item in value]
    return value


def _route(device: Any):
    def hook(module, args, kwargs):
        return move_inputs(args, device), move_inputs(kwargs, device)
    return hook


class TeacherPlacement:
    """Keep vision/AVA/prefix on device 0 and the decoder suffix/action head on 1.

    Hooks run on normal and checkpoint-recomputed decoder calls. They transfer
    tensors without detaching them. Parameter names and checkpoint format stay
    unchanged; this is model placement, not replicated data parallel training.
    """

    def __init__(self, model: Any, head: Any, proprio: Any, generator: Any, options: dict) -> None:
        import torch
        validate_parallel(options)
        self.model, self.head, self.proprio, self.generator = model, head, proprio, generator
        self.options = dict(options)
        self.devices = tuple(torch.device(d) for d in options["devices"])
        layers = model.language_model.model.layers
        if len(layers) < 2:
            raise ValueError("Teacher must have at least two native Llama decoder layers")
        layer_sizes = [sum(p.numel() * p.element_size() for p in layer.parameters()) for layer in layers]
        all_bytes = sum(p.numel() * p.element_size() for module in self.modules for p in module.parameters())
        suffix_bytes = sum(p.numel() * p.element_size() for module in
                           (model.language_model.model.norm, model.language_model.lm_head, head)
                           for p in module.parameters())
        prefix_bytes = all_bytes - sum(layer_sizes) - suffix_bytes
        def sizes(split):
            return prefix_bytes + sum(layer_sizes[:split]), suffix_bytes + sum(layer_sizes[split:])
        self.split = options["split_layer"] or min(range(1, len(layers)), key=lambda s: max(sizes(s)))
        if not 1 <= self.split < len(layers):
            raise ValueError("Teacher split_layer must leave decoder layers on both devices")
        self.weight_bytes = sizes(self.split)
        self.handles = []
        for index, layer in enumerate(layers):
            self.handles.append(layer.register_forward_pre_hook(
                _route(self.devices[index >= self.split]), with_kwargs=True))
        for module in (model.language_model.model.norm, model.language_model.lm_head):
            self.handles.append(module.register_forward_pre_hook(_route(self.devices[1]), with_kwargs=True))
        if generator is not None:
            self.handles.append(generator.register_forward_pre_hook(_route(self.devices[0]), with_kwargs=True))
        self.handles.append(proprio.register_forward_pre_hook(_route(self.devices[0]), with_kwargs=True))

    @property
    def modules(self) -> tuple:
        return tuple(m for m in (self.model, self.head, self.proprio, self.generator) if m is not None)

    def device_for(self, name: str) -> torch.device:
        prefix = "language_model.model.layers."
        if name.startswith(prefix):
            return self.devices[int(name[len(prefix):].split(".", 1)[0]) >= self.split]
        if name.startswith(("language_model.model.norm.", "language_model.lm_head.")):
            return self.devices[1]
        return self.devices[0]

    def activate(self) -> None:
        import torch
        reserve = self.options["reserve_gib"] * 1024**3
        for device, weights in zip(self.devices, self.weight_bytes):
            if device.index >= torch.cuda.device_count():
                raise RuntimeError("Two visible CUDA GPUs are required for Teacher placement")
            if self.options["dtype"] == "bfloat16" and torch.cuda.get_device_capability(device)[0] < 8:
                raise RuntimeError("This Teacher BF16 profile requires native BF16 CUDA support")
            free, _ = torch.cuda.mem_get_info(device)
            resident = sum(p.numel() * p.element_size() for module in self.modules for p in module.parameters()
                           if p.device == device)
            if weights - resident + reserve >= free:
                raise RuntimeError(f"Insufficient free memory on {device} for Teacher weights plus reserve")
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.data = parameter.data.to(self.device_for(name))
            for name, module in self.model.named_modules():
                for key, value in module._buffers.items():
                    if value is not None:
                        module._buffers[key] = value.to(self.device_for(f"{name}.{key}"))
            self.head.to(self.devices[1])
            self.proprio.to(self.devices[0])
            if self.generator is not None:
                self.generator.to(self.devices[0])

    def offload(self) -> None:
        import torch
        for module in self.modules:
            module.to("cpu")
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

    def summary(self) -> dict:
        return {"mode": "teacher_layer_parallel", "devices": [str(d) for d in self.devices],
                "split_layer": self.split, "dtype": self.options["dtype"],
                "weight_bytes_per_device": list(self.weight_bytes),
                "gradient_offload": self.options["gradient_offload"]}


@contextmanager
def generation_outputs(model, device: Any) -> Iterator[None]:
    """HF generation keeps token IDs on the prefix device; return logits there."""
    if getattr(model, "_vlade_teacher_placement", None) is None:
        yield
        return
    def output_hook(module, args, output):
        output.logits = output.logits.to(device)
        return output
    handle = model.language_model.register_forward_hook(output_hook)
    try:
        yield
    finally:
        handle.remove()
