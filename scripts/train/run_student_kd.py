"""Train Qwen with ground-truth supervision and cached T1 action distillation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.training.student_task import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings.get("method") != "action_kd":
        raise ValueError("This entrypoint requires method=action_kd")
    run(settings, resume=args.resume, preflight=args.preflight)


if __name__ == "__main__":
    main()
