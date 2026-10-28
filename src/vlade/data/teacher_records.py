"""Bounded JSONL Teacher records. No pickle, implicit tokenizer conversion, or cache mixing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _tensor(value: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if tuple(result.shape) != shape or not result.is_floating_point() or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite floating point with shape {shape}")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class TeacherProvenance:
    """Hashes identify immutable artifacts, not mutable paths or training step labels."""

    purpose: str  # training_debug or fixed_kd
    checkpoint_id: str
    checkpoint_sha256: str
    dataset_revision: str
    split_sha256: str
    normalization_sha256: str
    tokenizer_id: str
    tokenizer_revision: str
    action_convention: str
    temporal_mode: str  # episode_context or zero_context
    query_interval_steps: int
    frozen: bool
    quantized: bool
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1 or self.purpose not in {"training_debug", "fixed_kd"}:
            raise ValueError("Unsupported record schema/purpose")
        for key in ("checkpoint_id", "dataset_revision", "tokenizer_id", "tokenizer_revision", "action_convention"):
            _text(getattr(self, key), key)
        for key in ("checkpoint_sha256", "split_sha256", "normalization_sha256"):
            value = getattr(self, key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{key} must be a lowercase SHA256 digest")
        _integer(self.query_interval_steps, "query_interval_steps", 1)
        if self.temporal_mode not in {"episode_context", "zero_context"}:
            raise ValueError("Invalid temporal mode")
        if type(self.frozen) is not bool or type(self.quantized) is not bool:
            raise ValueError("frozen and quantized must be bool")
        if self.purpose == "fixed_kd" and (not self.frozen or self.quantized or self.temporal_mode != "episode_context"):
            raise ValueError("Base KD requires a frozen, unquantized, episode-context Teacher")


def validate_record(record: dict[str, Any], provenance: TeacherProvenance) -> None:
    """One query, two current camera refs, explicit targets and optional hidden states."""
    _text(record.get("episode_id"), "episode_id")
    _integer(record.get("timestep"), "timestep")
    _integer(record.get("teacher_step"), "teacher_step")
    _text(record.get("instruction"), "instruction")
    for key in ("episode_start", "episode_end"):
        if type(record.get(key)) is not bool:
            raise ValueError(f"{key} must be bool")
    refs = record.get("observation_refs")
    if not isinstance(refs, dict) or set(refs) != {"primary", "wrist"}:
        raise ValueError("observation_refs requires primary and wrist")
    for key, value in refs.items():
        _text(value, key)
    if record.get("checkpoint_sha256") != provenance.checkpoint_sha256:
        raise ValueError("Teacher checkpoint differs from this stream")
    if record.get("normalization_sha256") != provenance.normalization_sha256:
        raise ValueError("Normalization differs from this stream")
    _tensor(record.get("proprio"), (8,), "proprio")
    _tensor(record.get("gt_action"), (8, 7), "gt_action")
    _tensor(record.get("predicted_action"), (8, 7), "predicted_action")
    valid = torch.as_tensor(record.get("action_valid_mask"))
    if valid.shape != (8, 7) or valid.dtype != torch.bool or not valid.any():
        raise ValueError("action_valid_mask must contain valid elements in [8,7]")
    tokens = record.get("gt_token_ids")
    if tokens is not None:
        token_tensor = torch.as_tensor(tokens)
        if token_tensor.shape != (56,) or token_tensor.dtype != torch.int64 or (token_tensor < 0).any():
            raise ValueError("gt_token_ids must be 56 nonnegative int64 Teacher action IDs")
        if record.get("tokenizer_id") != provenance.tokenizer_id or record.get("tokenizer_revision") != provenance.tokenizer_revision:
            raise ValueError("Token IDs need the stream's exact tokenizer identity/revision")
    hidden = record.get("action_hidden_states")
    if hidden is not None:
        tensor = torch.as_tensor(hidden)
        if tensor.ndim != 2 or tensor.shape[0] != 56 or tensor.shape[1] < 1:
            raise ValueError("action_hidden_states must be [56, hidden_size]")
        _tensor(tensor, tuple(tensor.shape), "action_hidden_states")
        _text(record.get("hidden_layer"), "hidden_layer")
        if record.get("hidden_layout") != "horizon_major_action_dimension_minor":
            raise ValueError("Explicit hidden token semantics are required")


class _Sequence:
    def __init__(self, provenance: TeacherProvenance):
        self.provenance = provenance
        self.episode: str | None = None
        self.last_time: int | None = None
        self.completed: set[str] = set()
        self.teacher_step: int | None = None

    def accept(self, record: dict[str, Any]) -> None:
        validate_record(record, self.provenance)
        if self.provenance.purpose != "fixed_kd":
            return
        episode, time = record["episode_id"], record["timestep"]
        if self.teacher_step is not None and record["teacher_step"] != self.teacher_step:
            raise ValueError("Cannot mix Teacher training steps in a fixed KD stream")
        if self.episode is None:
            if episode in self.completed or not record["episode_start"] or time != 0:
                raise ValueError("Episode must begin at reset/timestep 0 and cannot reappear")
        elif episode != self.episode or record["episode_start"] or time != self.last_time + self.provenance.query_interval_steps:
            raise ValueError("Episode queries must remain ordered at the declared query interval")
        self.teacher_step = record["teacher_step"]
        self.last_time = time
        self.episode = episode
        if record["episode_end"]:
            self.completed.add(episode)
            self.episode = None


class TeacherRecordWriter:
    """Exclusive new stream; every-N sampling is permitted only for debug records.

    A sealed manifest is published only on successful close. A failed/incomplete
    KD stream cannot be consumed by the default reader. Bounds include UTF-8 bytes.
    Call write() on every query; sampling skips serialization and device transfers.
    """

    def __init__(self, directory: Path, provenance: TeacherProvenance, *, every_n: int = 1,
                 max_records: int = 10000, max_bytes: int = 64 * 1024 * 1024,
                 max_record_bytes: int = 16 * 1024 * 1024):
        provenance.validate()
        for name, value in (("every_n", every_n), ("max_records", max_records),
                            ("max_bytes", max_bytes), ("max_record_bytes", max_record_bytes)):
            _integer(value, name, 1)
        if provenance.purpose == "fixed_kd" and every_n != 1:
            raise ValueError("KD queries cannot be subsampled by the writer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.provenance = provenance
        self.every_n, self.max_records = every_n, max_records
        self.max_bytes, self.max_record_bytes = max_bytes, max_record_bytes
        self.count = self.seen = self.bytes_written = 0
        self.digest = hashlib.sha256()
        self.sequence = _Sequence(provenance)
        self.file = (self.directory / "records.jsonl").open("xb")
        self.closed = False
        self.failed = False

    def write(self, record: dict[str, Any]) -> bool:
        try:
            return self._write(record)
        except BaseException:
            self.failed = True
            raise

    def _write(self, record: dict[str, Any]) -> bool:
        if self.closed:
            raise ValueError("Writer is closed")
        index = self.seen
        self.seen += 1
        if index % self.every_n:
            return False
        if self.count >= self.max_records:
            return self._limit()
        # Validate before converting tensors, including integer/bool dtypes.
        validate_record(record, self.provenance)
        raw = (json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(raw) > self.max_record_bytes or self.bytes_written + len(raw) > self.max_bytes:
            return self._limit()
        self.sequence.accept(record)
        self.file.write(raw)
        self.digest.update(raw)
        self.bytes_written += len(raw)
        self.count += 1
        return True

    def _limit(self) -> bool:
        if self.provenance.purpose == "fixed_kd":
            raise ValueError("KD stream capacity exceeded; increase the explicit budget")
        return False

    def close(self, *, success: bool = True) -> None:
        if self.closed:
            return
        self.file.close()
        self.closed = True
        if success and self.failed:
            raise ValueError("Cannot seal a stream after a rejected write")
        if success and (not self.count or (self.provenance.purpose == "fixed_kd" and self.sequence.episode is not None)):
            raise ValueError("Cannot seal an empty or incomplete KD episode stream")
        if success:
            payload = {"provenance": asdict(self.provenance), "count": self.count,
                       "bytes": self.bytes_written, "sha256": self.digest.hexdigest(),
                       "every_n": self.every_n, "max_record_bytes": self.max_record_bytes}
            (self.directory / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def __enter__(self) -> "TeacherRecordWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(success=exc_type is None)


def load_teacher_records(directory: Path, *, for_kd: bool = True,
                         expected_provenance: TeacherProvenance | None = None,
                         max_record_bytes: int = 16 * 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Validate the complete stream before yielding records, then stream from it.

    Callers still resolve and verify referenced observations/stats against their
    dataset manifest. Token IDs remain Teacher-local metadata, never Student labels.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    provenance = TeacherProvenance(**manifest["provenance"])
    provenance.validate()
    if for_kd and provenance.purpose != "fixed_kd":
        raise ValueError("Training debug records are not a KD cache")
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("Cache provenance does not match the requested experiment")
    sequence = _Sequence(provenance)
    digest, count, size = hashlib.sha256(), 0, 0
    path = directory / "records.jsonl"

    def rows(handle):
        while raw := handle.readline(max_record_bytes + 1):
            if len(raw) > max_record_bytes or not raw.endswith(b"\n"):
                raise ValueError("Oversized or truncated Teacher record")
            yield raw, json.loads(raw)

    with path.open("rb") as handle:
        for raw, record in rows(handle):
            sequence.accept(record)
            digest.update(raw)
            count, size = count + 1, size + len(raw)
        if (count, size, digest.hexdigest()) != (manifest["count"], manifest["bytes"], manifest["sha256"]):
            raise ValueError("Teacher stream integrity check failed")
        if not count or (for_kd and sequence.episode is not None):
            raise ValueError("Empty or incomplete Teacher stream")
        handle.seek(0)
        for _, record in rows(handle):
            for key in ("proprio", "gt_action", "predicted_action", "action_hidden_states"):
                if record.get(key) is not None:
                    record[key] = torch.tensor(record[key], dtype=torch.float32)
            record["action_valid_mask"] = torch.tensor(record["action_valid_mask"], dtype=torch.bool)
            if record.get("gt_token_ids") is not None:
                record["gt_token_ids"] = torch.tensor(record["gt_token_ids"], dtype=torch.int64)
            yield record
