"""Episode-preserving LIBERO+ queries shared by FFT, fixed cache and Student BC.

Preparation keeps only query-time camera payloads and raw numeric trajectories.
Train-only quantiles are exact over all training transitions, not query samples.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from vlade.data.colab_assets import DATA_REV, sha256
from vlade.utils.filesystem import publish_directory

# Canonical D identifiers from vlade.data.canonical_split in the validated preparation source.
CANONICAL_SHA256 = "487d2be1039ba6a4facc3ba44785ba5159245dfe53462a1d5a41ba8230446e80"
CANONICAL_COUNTS = {"train": 11_477, "validation": 1_428, "test": 1_442}
CANONICAL_CALIBRATION_EPISODES = 728

CONVENTION = "libero_relative_xyz_axis_angle; model_gripper_1_open_0_closed; proprio_state_first6_last2"


def write_json(path: Path, value: Any) -> None:
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def group_split(content_sha256: str, seed: int = 42) -> str:
    value = int(hashlib.sha256(f"{seed}:{content_sha256}".encode()).hexdigest(), 16) / 2**256
    return "train" if value < 0.8 else "validation" if value < 0.9 else "test"


def _language(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _image_bytes(value: Any) -> tuple[bytes, str]:
    if isinstance(value, (bytes, np.bytes_)):
        raw = bytes(value)
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        return raw, ".jpg" if raw.startswith(b"\xff\xd8") else ".png"
    image = Image.fromarray(np.asarray(value, dtype=np.uint8)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), ".png"


def tfds_episodes(directory: Path) -> Iterator[list[dict]]:
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder_from_directory(str(directory))
    obs = builder.info.features["steps"]["observation"]
    if not {"image", "wrist_image", "state"}.issubset(obs.keys()):
        raise ValueError(f"Expected LIBERO image/wrist_image/state; actual schema: {obs}")
    decoders = {"steps": {"observation": {key: tfds.decode.SkipDecoding() for key in ("image", "wrist_image")}}}
    dataset = builder.as_dataset(split="train", shuffle_files=False, decoders=decoders,
                                read_config=tfds.ReadConfig(interleave_cycle_length=1,
                                                            num_parallel_calls_for_decode=1))
    for episode in tfds.as_numpy(dataset):
        yield list(episode["steps"])


def prepare(episodes: Iterator[list[dict]], destination: Path, *, seed: int = 42,
            source: dict | None = None) -> dict:
    """Publish only a completed, duplicate-grouped 80/10/10 prepared dataset."""
    if destination.exists():
        raise FileExistsError(f"Use a new prepared dataset path: {destination}")
    stage = destination.with_name(destination.name + ".incomplete")
    stage.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, steps in enumerate(episodes):
        if not steps:
            raise ValueError("Empty episode")
        episode_id = f"episode-{index:07d}"
        folder = stage / "episodes" / episode_id
        folder.mkdir(parents=True)
        actions = np.asarray([step["action"] for step in steps], dtype=np.float32)
        states = np.asarray([step["observation"]["state"] for step in steps], dtype=np.float32)
        if actions.shape != (len(steps), 7) or states.ndim != 2 or states.shape[1] < 8:
            raise ValueError(f"Invalid action/state shape: {actions.shape}, {states.shape}")
        proprio = np.concatenate((states[:, :6], states[:, -2:]), axis=-1)
        if not np.isfinite(actions).all() or not np.isfinite(proprio).all():
            raise ValueError("Non-finite trajectory")
        if np.any(actions[:, 6] < -1.0001) or np.any(actions[:, 6] > 1.0001):
            raise ValueError("Gripper outside the expected raw [-1,1] range")
        digest = hashlib.sha256()
        files = {}
        queries = []
        for t, step in enumerate(steps):
            instruction = _language(step["language_instruction"])
            if not instruction.strip():
                raise ValueError("Missing language instruction")
            # Length-prefix bytes to avoid ambiguous concatenations.
            values = [actions[t].tobytes(), states[t].tobytes(), instruction.encode("utf-8")]
            for camera, key in (("primary", "image"), ("wrist", "wrist_image")):
                value = step["observation"][key]
                raw = bytes(value) if isinstance(value, (bytes, np.bytes_)) else np.asarray(value).tobytes()
                values.append(raw)
                if t % 8 == 0:
                    payload, suffix = _image_bytes(value)
                    name = f"{camera}-{t:07d}{suffix}"
                    (folder / name).write_bytes(payload)
                    files[name] = hashlib.sha256(payload).hexdigest()
            for value in values:
                digest.update(len(value).to_bytes(8, "little"))
                digest.update(value)
            if t % 8 == 0:
                queries.append({"timestep": t, "instruction": instruction,
                                "primary": next(name for name in files if name.startswith(f"primary-{t:07d}.")),
                                "wrist": next(name for name in files if name.startswith(f"wrist-{t:07d}."))})
        np.savez(folder / "trajectory.npz", actions=actions, proprio=proprio)
        files["trajectory.npz"] = sha256(folder / "trajectory.npz")
        write_json(folder / "queries.json", queries)
        files["queries.json"] = sha256(folder / "queries.json")
        content = digest.hexdigest()
        entries.append({"episode_id": episode_id, "content_sha256": content,
                        "split": group_split(content, seed), "steps": len(steps), "queries": len(queries),
                        "files": files})
        if index % 100 == 0:
            print(f"Prepared {index + 1} episodes", flush=True)
    if not entries or any(not any(e["split"] == split for e in entries) for split in ("train", "validation", "test")):
        raise ValueError("Need nonempty train/validation/test splits; do not use a tiny subset as the formal dataset")
    training = []
    for entry in entries:
        if entry["split"] == "train":
            with np.load(stage / "episodes" / entry["episode_id"] / "trajectory.npz") as numeric:
                training.append({key: numeric[key].copy() for key in ("actions", "proprio")})
    statistics = {}
    for key, mask in (("actions", [True] * 6 + [False]), ("proprio", [True] * 8)):
        values = np.concatenate([item[key] for item in training], axis=0)
        if key == "actions":
            values[:, 6] = 1 - np.clip(values[:, 6], 0, 1)
        statistics["action" if key == "actions" else key] = {
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(), "mask": mask}
    split_rows = [{k: e[k] for k in ("episode_id", "content_sha256", "split", "steps", "queries")} for e in entries]
    write_json(stage / "splits.json", split_rows)
    statistics.update(convention=CONVENTION, split_sha256=sha256(stage / "splits.json"), train_only=True)
    write_json(stage / "normalization.json", statistics)
    manifest = {"schema_version": 1, "dataset_revision": DATA_REV, "seed": seed,
                "split_policy": "content_group_sha256_seed_threshold_80_10_10; approximate_episode_counts",
                "query_interval_steps": 8, "image_orientation": "as_stored_in_rlds_no_rotation",
                "normalization_sha256": sha256(stage / "normalization.json"),
                "split_sha256": sha256(stage / "splits.json"), "source": source,
                "counts": dict(Counter(e["split"] for e in entries)), "episodes": entries}
    write_json(stage / "manifest.json", manifest)
    (stage / "COMPLETE").write_text("ok\n", encoding="utf-8")
    publish_directory(stage, destination)
    return manifest


def normalize(values: np.ndarray, stats: dict) -> np.ndarray:
    low, high, mask = (np.asarray(stats[key]) for key in ("q01", "q99", "mask"))
    return np.where(mask, np.clip(2 * (values - low) / (high - low + 1e-8) - 1, -1, 1), values).astype(np.float32)


class EpisodeData:
    def __init__(self, root: Path):
        self.root = root
        if not (root / "COMPLETE").is_file():
            raise ValueError("Prepared dataset is incomplete")
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for filename, key in (("splits.json", "split_sha256"), ("normalization.json", "normalization_sha256")):
            if sha256(root / filename) != self.manifest[key]:
                raise ValueError(f"Prepared data identity mismatch: {filename}")
        self.stats = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
        self.entries = self.manifest["episodes"]
        splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
        schema = self.manifest.get("schema_version")
        keys = (("episode_id", "content_sha256", "content_group_id", "split", "is_calibration", "steps", "queries")
                if schema == 2 else ("episode_id", "content_sha256", "split", "steps", "queries"))
        if type(schema) is not int or schema not in {1, 2} or self.manifest.get("query_interval_steps") != 8:
            raise ValueError("Unsupported prepared-data schema/query interval")
        if not self.stats.get("train_only") or self.stats.get("split_sha256") != self.manifest["split_sha256"]:
            raise ValueError("Normalization is not bound to the declared train split")
        if self.manifest.get("canonical_manifest_sha256") is not None:
            if (self.manifest["canonical_manifest_sha256"] != CANONICAL_SHA256 or
                    self.stats.get("canonical_manifest_sha256") != CANONICAL_SHA256 or
                    self.stats.get("canonical_normalization_sha256") !=
                    self.manifest.get("canonical_normalization_sha256")):
                raise ValueError("Prepared data is not bound to the canonical manifest/normalization")
        if [{key: entry.get(key) for key in keys} for entry in self.entries] != splits:
            raise ValueError("Episode manifest differs from the sealed split manifest")
        identities, groups = set(), {}
        for entry in self.entries:
            identifier = entry["episode_id"]
            if (not isinstance(identifier, str) or identifier in identities or identifier in {".", ".."} or
                    any(char in identifier for char in ("/", "\\", ":"))):
                raise ValueError("Invalid/duplicate episode identity")
            identities.add(identifier)
            if schema == 2 and (not isinstance(entry.get("content_group_id"), str) or
                                not entry["content_group_id"] or type(entry.get("is_calibration")) is not bool):
                raise ValueError("Invalid episode content group/calibration metadata")
            if (entry["split"] not in {"train", "validation", "test"} or type(entry["steps"]) is not int or
                    entry["steps"] < 1 or type(entry["queries"]) is not int or
                    entry["queries"] != (entry["steps"] + 7) // 8):
                raise ValueError("Invalid episode split/step/query count")
            if entry.get("is_calibration") and entry["split"] != "train":
                raise ValueError("Calibration episodes must be a subset of Train")
            assignment = (entry["split"], entry.get("is_calibration", False))
            group = entry["content_group_id"] if schema == 2 else entry["content_sha256"]
            if group in groups and groups[group] != assignment:
                raise ValueError("Duplicate episode content crosses splits")
            groups[group] = assignment
        if dict(Counter(entry["split"] for entry in self.entries)) != self.manifest.get("counts"):
            raise ValueError("Episode split counts do not match the manifest")
        if self.manifest.get("canonical_manifest_sha256") is not None:
            if (self.manifest.get("counts") != CANONICAL_COUNTS or
                    self.manifest.get("calibration_episodes") != CANONICAL_CALIBRATION_EPISODES or
                    sum(bool(entry["is_calibration"]) for entry in self.entries) != CANONICAL_CALIBRATION_EPISODES):
                raise ValueError("Prepared data does not reproduce canonical split/calibration counts")

    def episodes(self, split: str, *, order: list[str] | None = None) -> Iterator[tuple[dict, list[dict]]]:
        entries = [entry for entry in self.entries if entry["split"] == split]
        if order is not None:
            lookup = {entry["episode_id"]: entry for entry in entries}
            if len(order) != len(set(order)) or set(order) - lookup.keys():
                raise ValueError("Invalid episode order/subset")
            entries = [lookup[key] for key in order]
        for entry in entries:
            if entry["split"] != split:
                continue
            folder = self.root / "episodes" / entry["episode_id"]
            for filename in ("trajectory.npz", "queries.json"):
                if sha256(folder / filename) != entry["files"][filename]:
                    raise ValueError(f"Episode integrity mismatch: {entry['episode_id']}/{filename}")
            with np.load(folder / "trajectory.npz") as raw:
                actions = raw["actions"].copy()
                proprio = raw["proprio"].copy()
            if (actions.shape != (entry["steps"], 7) or proprio.shape != (entry["steps"], 8) or
                    not np.isfinite(actions).all() or not np.isfinite(proprio).all()):
                raise ValueError("Trajectory shape/length/values differ from episode metadata")
            actions[:, 6] = 1 - np.clip(actions[:, 6], 0, 1)
            actions = normalize(actions, self.stats["action"])
            proprio = normalize(proprio, self.stats["proprio"])
            queries = json.loads((folder / "queries.json").read_text(encoding="utf-8"))
            if len(queries) != entry["queries"] or [q.get("timestep") for q in queries] != list(range(0, entry["steps"], 8)):
                raise ValueError("Query length/order differs from episode metadata")
            samples = []
            for query in queries:
                if not isinstance(query.get("instruction"), str) or not query["instruction"].strip():
                    raise ValueError("Missing query instruction")
                for camera in ("primary", "wrist"):
                    filename = query.get(camera)
                    if (not isinstance(filename, str) or filename not in entry["files"] or filename in {".", ".."} or
                            any(char in filename for char in ("/", "\\", ":"))):
                        raise ValueError("Invalid camera observation reference")
                t = query["timestep"]
                count = min(8, len(actions) - t)
                chunk = np.zeros((8, 7), dtype=np.float32)
                chunk[:, 6] = actions[-1, 6]
                chunk[:count] = actions[t:t + count]
                mask = np.zeros((8, 7), dtype=bool)
                mask[:count] = True
                samples.append({**query, "episode_id": entry["episode_id"], "folder": folder,
                                "gt_action": chunk, "action_valid_mask": mask, "proprio": proprio[t],
                                "image_hashes": {key: entry["files"][query[key]] for key in ("primary", "wrist")}})
            yield entry, samples

    def image(self, sample: dict, camera: str) -> Image.Image:
        path = sample["folder"] / sample[camera]
        if sha256(path) != sample["image_hashes"][camera]:
            raise ValueError(f"Image integrity mismatch: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")


def query_count(data: EpisodeData, split: str = "train", *, episode_limit: int = 0) -> int:
    entries = [entry for entry in data.entries if entry["split"] == split]
    if episode_limit:
        entries = entries[:episode_limit]
    return sum(entry["queries"] for entry in entries)


def query_stream(data: EpisodeData, split: str, *, seed: int = 42, repeat: bool = False,
                 skip: int = 0, episode_limit: int = 0, limit: int = 0) -> Iterator[tuple[dict, list[dict]]]:
    """Shuffle episodes per epoch, retain causal queries, resume at a query cursor.

    A bounded validation subset is spread uniformly over the complete split's
    query positions. It is deterministic and does not read the test split.
    """
    if split not in {"train", "validation"} or min(skip, episode_limit, limit) < 0:
        raise ValueError("Invalid stream settings; test is sealed from training/selection")
    entries = [entry for entry in data.entries if entry["split"] == split]
    if episode_limit:
        entries = entries[:episode_limit]
    total = sum(entry["queries"] for entry in entries)
    if not total:
        raise ValueError(f"Empty {split} split")
    if limit and repeat:
        raise ValueError("Bounded evaluation cannot repeat")
    epoch, cursor = divmod(skip, total) if repeat else (0, skip)
    while True:
        order = [entry["episode_id"] for entry in entries]
        if split == "train":
            random.Random(seed + epoch * 1000003).shuffle(order)
        chosen = {index * total // min(limit, total) for index in range(min(limit, total))} if limit else None
        position = 0
        for _, queries in data.episodes(split, order=order):
            for index, sample in enumerate(queries):
                if position >= cursor and (chosen is None or position in chosen):
                    yield sample, queries[max(0, index - 1):index + 1]
                position += 1
        if not repeat:
            return
        epoch += 1
        cursor = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfds-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    infos = list(args.tfds_root.rglob("dataset_info.json"))
    if len(infos) != 1:
        raise ValueError("Exactly one TFDS dataset_info.json is required")
    source = {"dataset_info": json.loads(infos[0].read_text(encoding="utf-8")),
              "dataset_info_sha256": sha256(infos[0]),
              "extraction_manifest": json.loads((args.tfds_root / "extraction_manifest.json").read_text(encoding="utf-8"))}
    prepare(tfds_episodes(infos[0].parent), args.output, source=source)


if __name__ == "__main__":
    main()
