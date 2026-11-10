"""Bounded TBPTT over independent sequences, with explicit local recurrent state."""

from __future__ import annotations

from typing import Any, Callable, Sequence
import math
import torch


def sequence_objective(
    frames: Sequence[Any],
    frame_fn: Callable,
    *,
    detach_every: int = 2,
    backward_scale: float | None = None,
) -> torch.Tensor:
    """Reset state per sequence; detach graph, not state values, at TBPTT cuts.

    With backward_scale, backward each segment immediately to bound memory.
    Without it, return an ordinary scalar for no-grad validation or diagnostics.
    """
    if not frames or type(detach_every) is not int or detach_every < 1:
        raise ValueError("Require nonempty sequence and positive TBPTT segment")
    if backward_scale is not None and (not math.isfinite(backward_scale) or backward_scale <= 0):
        raise ValueError("Backward scale must be finite and positive")
    context = None
    segment = None
    losses = []
    for index, frame in enumerate(frames):
        loss, context = frame_fn(frame, context)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite or non-scalar sequence loss")
        losses.append(loss.detach() if backward_scale is not None else loss)
        segment = loss if segment is None else segment + loss
        if (index + 1) % detach_every == 0 or index + 1 == len(frames):
            if backward_scale is not None:
                (segment * (backward_scale / len(frames))).backward()
            context = context.detach() if context is not None else None
            segment = None
    return torch.stack(losses).mean()


def fft_objective(
    cfg: Any,
    components: tuple,
    batch: dict,
    device: Any,
    num_patches: int,
    step: int,
    *,
    backward_scale: float | None = None,
) -> torch.Tensor:
    """Average complete sequences while freeing each TBPTT segment immediately."""
    from .fft import compute_loss

    sequences = batch["sequences"]
    if not sequences or any(len(frames) != cfg.sequence_length for frames in sequences):
        raise ValueError("FFT batch must contain complete configured sequences")
    losses = []
    for frames in sequences:
        metadata = [frame["sample_metadata"][0] for frame in frames]
        if len({row["episode_id"] for row in metadata}) != 1 or [
            row["timestep"] for row in metadata
        ] != list(
            range(metadata[0]["timestep"], metadata[0]["timestep"] + cfg.sequence_length * 8, 8)
        ):
            raise ValueError("FFT sequence crosses an episode or skips policy queries")

        def frame_fn(frame, context):
            return compute_loss(
                *components,
                frame,
                device,
                num_patches,
                step,
                temporal_context=context,
                return_temporal_state=True,
                temporal_feature_layer=cfg.temporal_feature_layer,
            )

        losses.append(
            sequence_objective(
                frames,
                frame_fn,
                detach_every=cfg.detach_every,
                backward_scale=None if backward_scale is None else backward_scale / len(sequences),
            )
        )
    return torch.stack(losses).mean()
