"""Canonical LIBERO+ D/seed-42 split validation and trajectory assignment."""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CANONICAL_CANDIDATE = "D_80_10_10"
CANONICAL_SEED = 42
CANONICAL_SHA256 = "487d2be1039ba6a4facc3ba44785ba5159245dfe53462a1d5a41ba8230446e80"
CANONICAL_COUNTS = {"train": 11_477, "validation": 1_428, "test": 1_442}
CANONICAL_CALIBRATION_EPISODES = 728
CANONICAL_GROUPS = 1_681


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_group_id(actions: np.ndarray, states: np.ndarray) -> str:
    """Return the exact action+state lineage key used by the selected manifest."""
    action = np.asarray(actions, dtype="<f4")
    state = np.asarray(states, dtype="<f4")
    if action.ndim != 2 or action.shape[1] != 7 or state.ndim != 2 or state.shape != (len(action), 8):
        raise ValueError(f"Invalid trajectory shape for canonical identity: {action.shape}, {state.shape}")
    payload = struct.pack("<Q", len(action)) + action.tobytes() + state.tobytes()
    return "trajectory-" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CanonicalSplit:
    path: Path
    sha256: str
    episode_ids: frozenset[str]
    group_assignments: dict[str, tuple[str, bool]]
    group_episode_counts: dict[str, int]
    group_episode_ids: dict[str, tuple[str, ...]]

    def assignment(self, actions: np.ndarray, states: np.ndarray) -> tuple[str, bool, str]:
        group_id = trajectory_group_id(actions, states)
        try:
            split, calibration = self.group_assignments[group_id]
        except KeyError as error:
            raise ValueError(f"Trajectory is absent from the canonical manifest: {group_id}") from error
        return split, calibration, group_id


def load_canonical_split(path: Path, expected_sha256: str = CANONICAL_SHA256) -> CanonicalSplit:
    """Validate the immutable candidate bytes and all canonical split invariants."""
    path = Path(path)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(f"Canonical manifest SHA-256 mismatch: {actual_sha256}")
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid canonical manifest JSON at line {line_number}") from error
    ids: set[str] = set()
    groups: dict[str, tuple[str, bool]] = {}
    group_episode_counts: Counter[str] = Counter()
    group_episode_ids: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()
    calibration = 0
    for row in rows:
        if row.get("candidate") != CANONICAL_CANDIDATE or row.get("seed") != CANONICAL_SEED:
            raise ValueError("Canonical manifest candidate/seed mismatch")
        episode_id = row.get("episode_id")
        group_id = row.get("confirmed_lineage_group_id")
        split = row.get("split")
        is_calibration = row.get("is_calibration")
        if (not isinstance(episode_id, str) or episode_id in ids or episode_id in {".", ".."} or
                any(char in episode_id for char in ("/", "\\", ":"))):
            raise ValueError("Canonical manifest has invalid or duplicate episode IDs")
        if not isinstance(group_id, str) or split not in CANONICAL_COUNTS or type(is_calibration) is not bool:
            raise ValueError("Canonical manifest row schema is invalid")
        if is_calibration and split != "train":
            raise ValueError("Calibration must be a subset of train")
        assignment = (split, is_calibration)
        if group_id in groups and groups[group_id] != assignment:
            raise ValueError("Canonical content group crosses split/calibration boundaries")
        ids.add(episode_id)
        groups[group_id] = assignment
        group_episode_counts[group_id] += 1
        group_episode_ids.setdefault(group_id, []).append(episode_id)
        counts[split] += 1
        calibration += int(is_calibration)
    if dict(counts) != CANONICAL_COUNTS:
        raise ValueError(f"Canonical episode counts mismatch: {dict(counts)}")
    if len(groups) != CANONICAL_GROUPS or calibration != CANONICAL_CALIBRATION_EPISODES:
        raise ValueError("Canonical group or calibration count mismatch")
    return CanonicalSplit(path.resolve(), actual_sha256, frozenset(ids), groups, dict(group_episode_counts),
                          {group_id: tuple(values) for group_id, values in group_episode_ids.items()})


def assert_observed_assignments(assignments: Iterable[tuple[str, bool, str]], canonical: CanonicalSplit) -> None:
    """Require a full source pass to reproduce the exact selected split counts."""
    rows = list(assignments)
    counts = Counter(split for split, _, _ in rows)
    calibration = sum(is_calibration for _, is_calibration, _ in rows)
    groups = Counter(group_id for _, _, group_id in rows)
    if dict(counts) != CANONICAL_COUNTS or calibration != CANONICAL_CALIBRATION_EPISODES:
        raise ValueError(f"Prepared episodes do not reproduce canonical counts: {dict(counts)}")
    if dict(groups) != canonical.group_episode_counts:
        raise ValueError("Prepared source does not contain the complete canonical content-group set")
