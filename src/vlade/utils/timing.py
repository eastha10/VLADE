"""Flushed, per-invocation CSV timings with explicit CUDA synchronization."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Callable, Iterator, TextIO
import uuid


class RunTimings:
    """Record wall time around named operations, without changing their results.

    CUDA synchronization occurs before and after each measured operation. Timings
    therefore include the synchronization overhead and should not be compared to
    unsynchronized launch time. Warm-up rows are retained and explicitly tagged.
    Each invocation writes a unique file; resume never truncates previous CSVs.
    """

    FIELDS = ("session_id", "event", "started_at_utc", "seconds", "status", "error_type",
              "step", "microstep", "phase", "episode_id", "timestep", "is_warmup",
              "replay", "batch_size", "accumulation_steps", "action_chunk_size", "device",
              "gpu", "dtype", "synchronization", "context_json")

    def __init__(self, directory: Path, *, role: str, device: str,
                 metadata: dict | None = None, session_id: str | None = None,
                 synchronize: Callable[[], None] | None = None,
                 stream: TextIO | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        if not role.replace("-", "").isalnum() or not self.session_id.isalnum():
            raise ValueError("Timing role/session must be safe filename components")
        self.device = device
        self.metadata = dict(metadata or {})
        self.stream = stream if stream is not None else sys.stderr
        if synchronize is None and str(device).startswith("cuda"):
            import torch
            synchronize = lambda: torch.cuda.synchronize(device)
        self.synchronize = synchronize or (lambda: None)
        self.synchronization = "cuda_synchronize_before_after" if str(device).startswith("cuda") else "host_wall_clock"
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"timings-{role}-{self.session_id}-{uuid.uuid4().hex[:8]}.csv"
        self.handle = self.path.open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.handle.flush()

    def start(self) -> tuple[str, float]:
        self.synchronize()
        return datetime.now(timezone.utc).isoformat(), time.perf_counter()

    def finish(self, start: tuple[str, float], event: str, *, status: str = "ok",
               error_type: str = "", **context) -> float:
        self.synchronize()
        seconds = time.perf_counter() - start[1]
        values = {**self.metadata, **context}
        row = {key: values.pop(key, "") for key in self.FIELDS}
        row.update(session_id=self.session_id, event=event, started_at_utc=start[0],
                   seconds=seconds, status=status, error_type=error_type, device=self.device,
                   synchronization=self.synchronization,
                   context_json=json.dumps(values, sort_keys=True))
        self.writer.writerow(row)
        self.handle.flush()
        print("TIMING " + json.dumps(row), file=self.stream, flush=True)
        return seconds

    @contextmanager
    def measure(self, event: str, **context) -> Iterator[None]:
        start = self.start()
        try:
            yield
        except BaseException as error:
            self.finish(start, event, status="failed", error_type=type(error).__name__, **context)
            raise
        else:
            self.finish(start, event, **context)

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> RunTimings:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
