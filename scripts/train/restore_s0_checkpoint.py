"""Restore the S0-400 checkpoint from an authorized copy of its tar parts."""
import argparse
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlade.training.s0_checkpoint import restore_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, required=True, help="Directory containing all 11 parts")
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent of the new checkpoint")
    parser.add_argument("--expected-sha256", help="Trusted SHA-256 of the complete tar, if available")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "experiments/manifests/s0_step400.json").read_text(encoding="utf-8"))
    try:
        checkpoint, digest = restore_checkpoint(
            args.parts_dir, args.output_dir, metadata, expected_sha256=args.expected_sha256,
        )
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"Restore failed: {exc}\n")
    print(f"Checkpoint: {checkpoint}")
    print(f"Tar SHA-256: {digest}")


if __name__ == "__main__":
    main()
