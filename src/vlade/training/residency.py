"""Synchronous model residency for incompatible Teacher/Student worker processes."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator


def require_host_memory(required_bytes: int, *, purpose: str) -> None:
    """Check Linux/Colab available RAM; leave the CUDA/Windows probe independent."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return
    available = next(int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < required_bytes + 2 * 1024**3:
        raise RuntimeError(f"Insufficient host RAM for {purpose}; need at least "
                           f"{required_bytes / 1024**3:.2f} GiB plus 2 GiB reserve, "
                           f"{available / 1024**3:.2f} GiB available. Use verified cache or a high-RAM runtime")


def validate_teacher_residency(settings: dict) -> str:
    mode = settings.get("teacher_residency", "shared")
    if mode not in {"shared", "exclusive"}:
        raise ValueError("teacher_residency must be shared or exclusive")
    if mode == "exclusive" and not settings.get("teacher_device", "cuda:0").startswith("cuda"):
        raise ValueError("Exclusive Teacher residency requires a CUDA teacher_device")
    return mode


class StudentOffload:
    """Evict parameters, accumulated gradients and Adam state before a T1 request.

    Only call before Student forward (or during no-grad validation replay), never
    between forward and backward. Parameter objects and optimizer keys stay fixed.
    The worker acknowledges a response only after releasing its GPU allocations.
    """

    def __init__(self, model: Any, optimizer: Any, device: Any, *, timings=None,
                 memory_path: Path | None = None):
        self.model, self.optimizer, self.device = model, optimizer, device
        self.timings, self.memory_path = timings, memory_path

    def host_budget_bytes(self) -> int:
        from vlade.training.memory import adamw_storage
        return adamw_storage(self.model)["minimum_training_bytes"]

    @contextmanager
    def __call__(self) -> Iterator[None]:
        import torch

        # Keep only original device metadata, never references to evicted GPU storage.
        parameters = list(self.model.parameters())
        parameter_devices = [p.device for p in parameters]
        grad_devices = [p.grad.device if p.grad is not None else None for p in parameters]
        buffers = [(module, key, value.device) for module in self.model.modules()
                   for key, value in module._buffers.items() if value is not None]
        state_devices = {}

        def move_state(value, path=(), *, restore=False):
            if isinstance(value, torch.Tensor):
                if not restore:
                    state_devices[path] = value.device
                return value.to(state_devices[path] if restore else "cpu")
            if isinstance(value, dict):
                return {key: move_state(item, (*path, key), restore=restore) for key, item in value.items()}
            if isinstance(value, list):
                return [move_state(item, (*path, index), restore=restore) for index, item in enumerate(value)]
            if isinstance(value, tuple):
                return tuple(move_state(item, (*path, index), restore=restore) for index, item in enumerate(value))
            return value

        def measure(event):
            return self.timings.measure(event, phase="residency") if self.timings else nullcontext()

        # Check available host RAM on Colab before copying a full optimizer to CPU.
        required = sum(p.numel() * p.element_size() for p in parameters if p.device.type == "cuda")
        required += sum(p.grad.numel() * p.grad.element_size() for p in parameters
                        if p.grad is not None and p.grad.device.type == "cuda")

        def state_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size() if value.device.type == "cuda" else 0
            if isinstance(value, dict):
                return sum(state_bytes(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(state_bytes(item) for item in value)
            return 0

        required += state_bytes(dict(self.optimizer.state))
        require_host_memory(required, purpose="exclusive Teacher/Student offload")
        with measure("s0_offload_to_cpu"):
            with torch.no_grad():
                for parameter in parameters:
                    parameter.data = parameter.data.to("cpu")
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.data.to("cpu")
                for module, key, _ in buffers:
                    module._buffers[key] = module._buffers[key].to("cpu")
                for index, parameter in enumerate(parameters):
                    if parameter in self.optimizer.state:
                        self.optimizer.state[parameter] = move_state(self.optimizer.state[parameter], (index,))
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
        try:
            if self.memory_path is not None:
                from vlade.training.memory import record_memory
                record_memory(self.memory_path, self.device, "student_offloaded")
            yield
        finally:
            with measure("s0_restore_to_gpu"):
                with torch.no_grad():
                    for parameter, original, grad_device in zip(parameters, parameter_devices, grad_devices):
                        parameter.data = parameter.data.to(original)
                        if parameter.grad is not None:
                            parameter.grad.data = parameter.grad.data.to(grad_device)
                    for module, key, original in buffers:
                        module._buffers[key] = module._buffers[key].to(original)
                    for index, parameter in enumerate(parameters):
                        if parameter in self.optimizer.state:
                            self.optimizer.state[parameter] = move_state(self.optimizer.state[parameter], (index,), restore=True)


@contextmanager
def teacher_on_device(policy: Any, device: str, *, exclusive: bool) -> Iterator[None]:
    """Keep the worker on CPU between requests, including its recurrent context."""
    if not exclusive:
        yield
        return
    import torch

    modules = (policy.model, policy.head, policy.proprio, policy.generator)
    placement = getattr(policy.model, "_vlade_teacher_placement", None)
    try:
        if placement is not None:
            placement.activate()
        else:
            for module in modules:
                if module is not None:
                    module.to(device)
        policy.device = torch.device(device)
        if policy.context is not None:
            policy.context = policy.context.to(device)
        yield
    finally:
        if placement is not None:
            placement.offload()
        else:
            for module in modules:
                if module is not None:
                    module.to("cpu")
        if policy.context is not None:
            policy.context = policy.context.to("cpu")
        policy.device = torch.device("cpu")
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
