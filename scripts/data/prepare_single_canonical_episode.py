"""Prepare one canonical Train episode for a bounded real-model FFT smoke run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vlade.data.canonical_split import CANONICAL_SHA256, load_canonical_split  # noqa: E402
from vlade.data.colab_assets import DATA_REV, sha256  # noqa: E402
from vlade.data.episode_data import (  # noqa: E402
    prepare_canonical_train_episode,
    tfds_episodes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfds-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id", help="Optional exact canonical Train Episode ID")
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-sha256", default=CANONICAL_SHA256)
    parser.add_argument("--canonical-normalization", type=Path, required=True)
    parser.add_argument("--canonical-normalization-sha256", required=True)
    args = parser.parse_args()

    infos = list(args.tfds_root.rglob("dataset_info.json"))
    if len(infos) != 1:
        raise ValueError("Exactly one TFDS dataset_info.json is required")
    canonical = load_canonical_split(args.canonical_manifest, args.canonical_sha256)
    actual_normalization_sha256 = sha256(args.canonical_normalization)
    if actual_normalization_sha256 != args.canonical_normalization_sha256.lower():
        raise ValueError("Canonical normalization SHA-256 mismatch")
    normalization = json.loads(args.canonical_normalization.read_text(encoding="utf-8"))
    extraction = args.tfds_root / "extraction_manifest.json"
    source = {
        "dataset_revision": DATA_REV,
        "dataset_info": json.loads(infos[0].read_text(encoding="utf-8")),
        "dataset_info_sha256": sha256(infos[0]),
        "extraction_manifest": (json.loads(extraction.read_text(encoding="utf-8"))
                                if extraction.is_file() else None),
    }
    manifest = prepare_canonical_train_episode(
        tfds_episodes(infos[0].parent),
        args.output,
        source=source,
        canonical=canonical,
        canonical_normalization=normalization,
        canonical_normalization_sha256=actual_normalization_sha256,
        episode_id=args.episode_id,
    )
    print(json.dumps({
        "output": str(args.output),
        "episode_id": manifest["episodes"][0]["episode_id"],
        "source_episode_index": manifest["selected_source_episode_index"],
        "queries": manifest["episodes"][0]["queries"],
        "dataset_scope": manifest["dataset_scope"],
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "canonical_normalization_sha256": manifest["canonical_normalization_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
