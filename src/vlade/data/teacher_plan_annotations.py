"""Prepare reviewed Plan targets from verified Train observations, then publish them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vlade.distillation.teacher_plan import PLAN_PROMPT_VERSION, normalize_plan
from vlade.storage.assets import NAME, resolve_input, sha256


def _data(reference: str) -> Any:
    from vlade.data.episode_data import EpisodeData

    return EpisodeData(resolve_input(reference))


def _identity(data: Any) -> dict:
    return {"manifest_sha256": sha256(data.root / "manifest.json"),
            **{key: data.manifest[key] for key in ("split_sha256", "normalization_sha256")}}


def export_template(reference: str, output: Path) -> dict:
    """Export empty Train targets and current-view references; never invent labels."""
    data = _data(reference)  # Verify external input before creating any output.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "prompt_version": PLAN_PROMPT_VERSION,
                "annotation_source": "unreviewed", "prepared_data_identity": _identity(data)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    count = 0
    with (output / "plans.jsonl").open("x", encoding="utf-8") as plans, \
            (output / "observations.jsonl").open("x", encoding="utf-8") as observations:
        for _, queries in data.episodes("train"):
            for sample in queries:
                row = {key: sample[key] for key in ("episode_id", "timestep", "instruction")}
                plans.write(json.dumps({"split": "train", **row, "plan": ""}, ensure_ascii=False) + "\n")
                views = {key: (sample["folder"] / sample[key]).relative_to(data.root).as_posix()
                         for key in ("primary", "wrist")}
                observations.write(json.dumps({**row, "data": reference, "views": views,
                    "proprio": sample["proprio"].tolist()}, ensure_ascii=False) + "\n")
                count += 1
    return {"status": "unreviewed_template", "queries": count, "output": str(output)}


def validate_reviewed(reference: str, draft: Path, review_source: str) -> dict:
    """Check complete Train coverage and exact instructions after explicit review."""
    if review_source not in {"human_reviewed", "externally_reviewed"}:
        raise ValueError("Declare the actual human_reviewed or externally_reviewed source")
    data = _data(reference)
    manifest = json.loads((draft / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("prompt_version") != PLAN_PROMPT_VERSION
            or manifest.get("prepared_data_identity") != _identity(data)):
        raise ValueError("Draft does not match the verified prepared input")
    rows = {}
    with (draft / "plans.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {"split", "episode_id", "timestep", "instruction", "plan"}:
                raise ValueError("Unexpected Plan row fields")
            key = row["episode_id"], row["timestep"]
            if row["split"] != "train" or type(key[1]) is not int or key in rows:
                raise ValueError("Plans must be unique Train queries")
            normalize_plan(row["plan"])
            rows[key] = row
    count = 0
    for _, queries in data.episodes("train"):
        for sample in queries:
            row = rows.pop((sample["episode_id"], sample["timestep"]), None)
            if row is None or row["instruction"] != sample["instruction"]:
                raise ValueError("Missing query or mismatched instruction in Plan targets")
            count += 1
    if rows or not count:
        raise ValueError("Plans contain extra queries or no Train queries")
    return {**manifest, "annotation_source": review_source}


def publish_reviewed(reference: str, draft: Path, review_source: str, key: str) -> dict:
    """Explicitly publish reviewed annotations using the existing private asset uploader."""
    if not NAME.fullmatch(key) or ".." in key or not key.startswith("teacher_plans_"):
        raise ValueError("Use a distinct teacher_plans_<experiment> asset key")
    manifest = validate_reviewed(reference, draft, review_source)
    package = draft / "publication"
    package.mkdir(exist_ok=True)
    path = package / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Publication manifest changed; use a new annotation draft/asset")
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    from huggingface_hub import HfApi
    from vlade.storage.migrate import publish_files, published_registry

    api = HfApi()
    existing = published_registry(api)["assets"].get(key)
    sources = {"manifest.json": path, "plans.jsonl": draft / "plans.jsonl"}
    if existing is not None:
        identities = {name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for name, p in sources.items()}
        if (existing.get("repo_id") != "VLADE-konkuk/libero-plus-prepared"
                or existing.get("repo_type") != "dataset" or existing.get("files") != identities):
            raise ValueError("This annotation asset key already has different content; choose a new key")
    return publish_files(api, key, sources,
                         repo="VLADE-konkuk/libero-plus-prepared", kind="dataset", local_name=key)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("template", "validate", "publish"))
    parser.add_argument("--data", default="hfasset://prepared_d_seed42")
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--review-source", choices=("human_reviewed", "externally_reviewed"))
    parser.add_argument("--asset-key", default="teacher_plans_d_seed42")
    args = parser.parse_args(argv)
    if args.action == "template":
        result = export_template(args.data, args.draft)
    elif args.action == "validate":
        result = validate_reviewed(args.data, args.draft, args.review_source)
    else:
        result = publish_reviewed(args.data, args.draft, args.review_source, args.asset_key)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
