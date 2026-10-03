"""Stage the pinned public LIBERO+ checkpoints and archives on local disk."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

from download_hf_file import download_file


ROOT = Path(__file__).resolve().parents[2]
AVA_REPO = "LiAuto-DSR/avavla-libero-4in1"
AVA_REV = "090bf68d234ea13bc78476236f82458b25b797b8"
OFT_REPO = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
OFT_REV = "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"
ASSETS_REPO = "Sylvest/LIBERO-plus"
ASSETS_REV = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
RLDS_REPO = "Sylvest/libero_plus_rlds"
RLDS_REV = "fb0c7029b076030d5d57227229e4f7460def1f7c"


def specification(include_dataset: bool) -> list[tuple[str, str, str, str, Path, Path | None]]:
    ava_dir = ROOT / "checkpoints/teacher/avavla-libero-4in1"
    oft_dir = ROOT / "checkpoints/baselines/openvla_oft/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    files: list[tuple[str, str, str, str, Path, Path | None]] = []
    for number in range(1, 5):
        filename = f"model-{number:05d}-of-00004.safetensors"
        partial = None
        if number == 1:
            partial = (
                ava_dir
                / ".cache/huggingface/download/"
                / "IO4xwqmZYzFmxznkwkiNSBwO1H0=.84a8f32f7d5a7bbd8d15fd9fda5d3206d08e70bbe93c8256f98fc0c30f2111eb.incomplete"
            )
        files.append(("model", AVA_REPO, AVA_REV, filename, ava_dir, partial))
        files.append(("model", OFT_REPO, OFT_REV, filename, oft_dir, None))
    for filename in (
        "action_head--40000_checkpoint.pt",
        "proprio_projector--40000_checkpoint.pt",
        "vision_attn_weight_generator--40000_checkpoint.pt",
    ):
        files.append(("model", AVA_REPO, AVA_REV, filename, ava_dir, None))
    for filename in (
        "action_head--300000_checkpoint.pt",
        "proprio_projector--300000_checkpoint.pt",
        "lora_adapter/adapter_model.safetensors",
    ):
        files.append(("model", OFT_REPO, OFT_REV, filename, oft_dir, None))
    if include_dataset:
        files.append(
            ("dataset", ASSETS_REPO, ASSETS_REV, "assets.zip", ROOT / "data/raw/libero_plus_assets", None)
        )
        for filename in ("libero_plus_mixdata.z01", "libero_plus_mixdata.z02", "libero_plus_mixdata.zip"):
            files.append(("dataset", RLDS_REPO, RLDS_REV, filename, ROOT / "data/raw/libero_plus_rlds", None))
    return files


def stage_one(item: tuple[str, str, str, str, Path, Path | None], chunk_mb: int) -> str:
    repo_type, repository, revision, filename, directory, partial = item
    prefix = "datasets/" if repo_type == "dataset" else ""
    url = f"https://huggingface.co/{prefix}{repository}/resolve/{revision}/{quote(filename, safe='/')}"
    if partial is not None and not partial.exists():
        partial = None
    download_file(url, directory / filename, chunk_mb * 1024 * 1024, 50, partial)
    return f"{repository}/{filename}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-dataset", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-mb", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_mb < 1:
        parser.error("--workers and --chunk-mb must be positive")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(stage_one, item, args.chunk_mb): item for item in specification(args.include_dataset)}
        for future in as_completed(futures):
            item = futures[future]
            try:
                print(f"COMPLETE {future.result()}", flush=True)
            except Exception as error:
                failures.append(item)
                print(f"FAILED {item[1]}/{item[3]}: {error}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} file(s) incomplete; rerun to resume")
    print("All selected public checkpoint and archive files verified.", flush=True)


if __name__ == "__main__":
    main()
