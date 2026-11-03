"""Verified, episode-bounded access to the fixed T1 action cache produced by 02."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from vlade.data.colab_assets import sha256
from vlade.data.episode_data import CONVENTION, EpisodeData
from vlade.data.teacher_records import TeacherProvenance, load_teacher_records


class CachedActions:
    """Validate every part once; retain only one episode's action tensors in RAM.

    No Teacher model or Teacher checkpoint is loaded. Images stay in EpisodeData.
    A partial pilot cache is accepted only with an explicit episode_limit.
    """

    def __init__(self, root: Path, data: EpisodeData, *, episode_limit: int = 0):
        self.root = Path(root)
        if not (self.root / "COMPLETE").is_file():
            raise ValueError("A completed Teacher cache is required")
        manifest = json.loads((self.root / "cache_manifest.json").read_text(encoding="utf-8"))
        config = json.loads((self.root / "cache_config.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("scope") not in {"pilot", "full_train_split"}:
            raise ValueError("Invalid Teacher cache status/scope")
        if config.get("data_manifest_sha256") != sha256(data.root / "manifest.json"):
            raise ValueError("Teacher cache/data manifest mismatch")
        selection = config["selection"]
        if (selection.get("frozen") is not True or selection.get("quantized") is not False
                or any(selection.get(key) != data.manifest[key]
                       for key in ("normalization_sha256", "split_sha256"))):
            raise ValueError("Teacher selection/statistics/split mismatch")
        entries = {entry["episode_id"]: entry for entry in data.entries if entry["split"] == "train"}
        required = list(entries)[:episode_limit] if episode_limit else list(entries)
        parts = manifest.get("episodes", [])
        identifiers = [part["episode_id"] for part in parts]
        if (not required or len(identifiers) != len(set(identifiers))
                or set(identifiers) - entries.keys() or set(required) - set(identifiers)):
            raise ValueError("Teacher cache must cover the requested train episodes exactly once")
        if not episode_limit and (manifest["scope"] != "full_train_split" or set(identifiers) != set(entries)):
            raise ValueError("Full Student training requires the complete train cache, not a pilot")
        self.parts = {part["episode_id"]: part for part in parts}
        self.provenance = None
        queries, size = 0, 0
        for entry, samples in data.episodes("train", order=identifiers):
            identifier = entry["episode_id"]
            folder = self.root / identifier
            part = self.parts[identifier]
            if sha256(folder / "manifest.json") != part["manifest_sha256"]:
                raise ValueError("Teacher cache part manifest changed")
            provenance = TeacherProvenance(**json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["provenance"])
            provenance.validate()
            if (provenance.purpose != "fixed_kd" or provenance.query_interval_steps != 8
                    or provenance.action_convention != CONVENTION
                    or provenance.dataset_revision != data.manifest["dataset_revision"]
                    or provenance.checkpoint_id != selection["checkpoint"]
                    or provenance.checkpoint_sha256 != selection["checkpoint_sha256"]
                    or provenance.normalization_sha256 != data.manifest["normalization_sha256"]
                    or provenance.split_sha256 != data.manifest["split_sha256"]):
                raise ValueError("Teacher cache provenance/data/selection mismatch")
            if self.provenance is not None and provenance != self.provenance:
                raise ValueError("Mixed Teacher provenance in cache")
            self.provenance = provenance
            count = 0
            for row in load_teacher_records(folder, expected_provenance=provenance):
                if count >= len(samples):
                    raise ValueError("Extra Teacher query")
                sample = samples[count]
                refs = {camera: f"episodes/{identifier}/{sample[camera]}" for camera in ("primary", "wrist")}
                if (row["episode_id"] != identifier or row["timestep"] != sample["timestep"]
                        or row["instruction"] != sample["instruction"] or row["observation_refs"] != refs
                        or row["teacher_step"] != selection["teacher_step"]
                        or row["episode_start"] != (count == 0)
                        or row["episode_end"] != (count == len(samples) - 1)
                        or any(not torch.equal(row[key], torch.from_numpy(sample[key]))
                               for key in ("proprio", "gt_action", "action_valid_mask"))):
                    raise ValueError("Teacher query does not match its source observation/target")
                count += 1
            if count != len(samples) or count != part["queries"]:
                raise ValueError("Missing Teacher query/count mismatch")
            queries += count
            size += (folder / "records.jsonl").stat().st_size
        if (queries, size) != (manifest.get("queries"), manifest.get("bytes")):
            raise ValueError("Teacher cache aggregate count/bytes mismatch")
        self.signature = {"cache_config_sha256": sha256(self.root / "cache_config.json"),
                          "cache_manifest_sha256": sha256(self.root / "cache_manifest.json"),
                          "checkpoint_sha256": selection["checkpoint_sha256"],
                          "queries": queries, "scope": manifest["scope"]}
        self._episode = None
        self._actions = {}

    def get(self, sample: dict) -> torch.Tensor:
        identifier = sample["episode_id"]
        if identifier != self._episode:
            part = self.parts[identifier]
            folder = self.root / identifier
            if sha256(folder / "manifest.json") != part["manifest_sha256"]:
                raise ValueError("Teacher cache part changed during training")
            self._actions = {row["timestep"]: row["predicted_action"]
                             for row in load_teacher_records(folder, expected_provenance=self.provenance)}
            self._episode = identifier
        return self._actions[sample["timestep"]]
