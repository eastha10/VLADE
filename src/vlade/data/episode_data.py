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

# Only training sampling uses these labels. No suite labels are inferred from text.
SUITES = ("spatial", "goal", "object", "libero_10")
SUITE_ALIASES = {
    "spatial": "spatial", "libero_spatial": "spatial",
    "goal": "goal", "libero_goal": "goal",
    "object": "object", "libero_object": "object",
    "libero_10": "libero_10", "libero10": "libero_10", "10": "libero_10",
}



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
        from vlade.storage.assets import resolve_path
        root = resolve_path(root)
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


def _canonical_suite(label: Any) -> str:
    if not isinstance(label, str):
        raise ValueError(f"Suite must be a string, got: {label!r}")
    key = label.strip().lower().replace("-", "_")
    if key not in SUITE_ALIASES:
        raise ValueError(f"Unknown LIBERO suite: {label!r}; expected one of {SUITES}")
    return SUITE_ALIASES[key]


def _read_suite_mapping(data: EpisodeData, suite_map: dict | Path | None) -> tuple[dict, dict]:
    """Load trusted, explicit labels without changing the sealed split manifest.

    Optional <prepared-root>/suite_map.json format:
      {"by_episode_id": {"episode-0000000": "goal"},
       "by_instruction": {"exact task instruction": "spatial"}}

    A flat {episode_id: suite} mapping is also accepted for convenience.
    """
    if suite_map is None:
        source = data.root / "suite_map.json"
        suite_map = source if source.is_file() else {}
    if isinstance(suite_map, (str, Path)):
        suite_map = json.loads(Path(suite_map).read_text(encoding="utf-8"))
    if not isinstance(suite_map, dict):
        raise ValueError("suite_map must be a dict or a JSON file path")
    if "by_episode_id" in suite_map or "by_instruction" in suite_map:
        if set(suite_map) - {"by_episode_id", "by_instruction"}:
            raise ValueError("Unknown suite mapping fields")
        by_episode = suite_map.get("by_episode_id", {})
        by_instruction = suite_map.get("by_instruction", {})
    else:
        by_episode, by_instruction = suite_map, {}
    if not isinstance(by_episode, dict) or not isinstance(by_instruction, dict):
        raise ValueError("Suite mappings must be objects mapping identifiers to suite names")
    return ({key: _canonical_suite(value) for key, value in by_episode.items()},
            {key: _canonical_suite(value) for key, value in by_instruction.items()})


def _episode_suite(entry: dict, queries: list[dict], by_episode: dict, by_instruction: dict) -> str:
    episode_id = entry["episode_id"]
    labels = []
    if entry.get("suite") is not None:
        labels.append(_canonical_suite(entry["suite"]))
    if episode_id in by_episode:
        labels.append(by_episode[episode_id])
    if by_instruction:
        instructions = {query["instruction"] for query in queries}
        if len(instructions) != 1:
            raise ValueError(f"Episode {episode_id} does not have a single stable task instruction")
        instruction = next(iter(instructions))
        if instruction in by_instruction:
            labels.append(by_instruction[instruction])
    if not labels:
        raise ValueError(f"Suite mapping missing for train episode {episode_id}; "
                         "provide suite_map.json with verified labels")
    if len(set(labels)) != 1:
        raise ValueError(f"Conflicting suite labels for train episode {episode_id}: {labels}")
    return labels[0]


def _balanced_query_stream(data: EpisodeData, entries: list[dict], *, seed: int,
                           repeat: bool, skip: int, suite_map: dict | Path | None
                           ) -> Iterator[tuple[dict, list[dict]]]:
    """Select equal suite quotas *before* a deterministic constrained query shuffle.

    A virtual epoch has the same Query count as the original train stream.
    Underrepresented suites are repeated with replacement only after all of
    their unique Queries have been selected. Abundant suites are subsampled.
    Shuffle suites in groups of four so every completed group is 25% each,
    even if a time-limited run stops partway through an epoch.
    """
    by_episode, by_instruction = _read_suite_mapping(data, suite_map)
    if (not by_episode and not by_instruction and
            any(entry.get("suite") is None for entry in entries)):
        raise ValueError("The prepared manifest has no suite labels. Supply verified "
                         "<prepared-root>/suite_map.json or the suite_map argument")

    buckets: dict[str, list[tuple[dict, dict | None]]] = {suite: [] for suite in SUITES}
    # Materialize numeric Query samples once, never load camera images into RAM.
    # This avoids re-opening thousands of episode files for every shuffled Query.
    for entry, queries in data.episodes("train", order=[item["episode_id"] for item in entries]):
        suite = _episode_suite(entry, queries, by_episode, by_instruction)
        previous = None
        for sample in queries:
            sample["suite"] = suite
            buckets[suite].append((sample, previous))
            previous = sample
    missing = [suite for suite in SUITES if not buckets[suite]]
    if missing:
        raise ValueError(f"Cannot balance train: empty suite(s) {missing}")

    total = sum(map(len, buckets.values()))
    epoch, cursor = divmod(skip, total) if repeat else (0, skip)
    while True:
        rng = random.Random(seed + epoch * 1000003)
        remainder_order = list(SUITES)
        rng.shuffle(remainder_order)
        base, remainder = divmod(total, len(SUITES))
        quotas = {suite: base + (suite in remainder_order[:remainder]) for suite in SUITES}
        selected: dict[str, list[tuple[dict, dict | None]]] = {}
        for suite in SUITES:
            pool, count = buckets[suite], quotas[suite]
            if count <= len(pool):
                indices = rng.sample(range(len(pool)), count)
            else:
                indices = list(range(len(pool)))
                indices.extend(rng.randrange(len(pool)) for _ in range(count - len(pool)))
                rng.shuffle(indices)
            selected[suite] = [pool[index] for index in indices]

        # Constrained shuffle: shuffle all Query orders within each suite,
        # then randomly shuffle suite order each round. Every four consecutive
        # queries contain one from each suite until final (<4) remainder.
        positions = {suite: 0 for suite in SUITES}
        emitted = 0
        while emitted < total:
            suite_order = [suite for suite in SUITES if positions[suite] < quotas[suite]]
            rng.shuffle(suite_order)
            for suite in suite_order:
                sample, previous = selected[suite][positions[suite]]
                positions[suite] += 1
                if emitted >= cursor:
                    yield sample, [sample] if previous is None else [previous, sample]
                emitted += 1
        if not repeat:
            return
        epoch += 1
        cursor = 0


def query_stream(data: EpisodeData, split: str, *, seed: int = 42, repeat: bool = False,
                 skip: int = 0, episode_limit: int = 0, limit: int = 0,
                 sampling: str = "episode", suite_map: dict | Path | None = None
                 ) -> Iterator[tuple[dict, list[dict]]]:
    """Emit Queries in legacy episode order or opt-in suite-balanced order.

    Bounded validation remains deterministic and unaffected. Balanced training
    uses verified suite labels, and preserves Query interval/action chunk size.
    """
    if split not in {"train", "validation"} or min(skip, episode_limit, limit) < 0:
        raise ValueError("Invalid stream settings; test is sealed from training/selection")
    if sampling not in {"episode", "suite_balanced"}:
        raise ValueError(f"Unknown Query sampling strategy: {sampling}")
    entries = [entry for entry in data.entries if entry["split"] == split]
    if episode_limit:
        entries = entries[:episode_limit]
    total = sum(entry["queries"] for entry in entries)
    if not total:
        raise ValueError(f"Empty {split} split")
    if sampling == "suite_balanced":
        if split != "train" or limit:
            raise ValueError("Suite-balanced sampling is train-only and does not accept limit")
        yield from _balanced_query_stream(data, entries, seed=seed, repeat=repeat,
                                          skip=skip, suite_map=suite_map)
        return
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
    from vlade.storage.assets import resolve_input
    args.tfds_root = resolve_input(args.tfds_root)
    infos = list(args.tfds_root.rglob("dataset_info.json"))
    if len(infos) != 1:
        raise ValueError("Exactly one TFDS dataset_info.json is required")
    source = {"dataset_info": json.loads(infos[0].read_text(encoding="utf-8")),
              "dataset_info_sha256": sha256(infos[0]),
              "extraction_manifest": json.loads((args.tfds_root / "extraction_manifest.json").read_text(encoding="utf-8"))}
    prepare(tfds_episodes(infos[0].parent), args.output, source=source)


if __name__ == "__main__":
    main()
