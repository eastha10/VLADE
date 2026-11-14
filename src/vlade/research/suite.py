"""Verified suite-map inputs and the pilot/main boundary for balanced FFT."""

from __future__ import annotations

import json
from pathlib import Path


def prepare_phase(config: dict, output: Path) -> tuple[str, Path]:
    """Resolve the audited map and require a sealed four-step pilot before main."""
    from vlade.storage.assets import resolve_input, sha256
    from vlade.training.run_control import verify_checkpoint
    from vlade.utils.source_identity import source_identity

    phase = config["settings"]["phase"]
    if phase not in {"pilot4", "main"}:
        raise ValueError("Suite phase must be pilot4 or main")
    suite_map = resolve_input(config["settings"]["suite_map"])
    if not suite_map.is_dir():
        raise ValueError("Suite map asset must be a verified directory")
    suite_map = suite_map / "suite_map.json"
    if not suite_map.is_file():
        raise FileNotFoundError("Verified suite map asset lacks suite_map.json")
    if phase == "main":
        report = json.loads((output / "pilot4/pilot_report.json").read_text())
        checkpoint = Path(report["checkpoint"])
        if not checkpoint.resolve().is_relative_to((output / "pilot4/checkpoints").resolve()):
            raise ValueError("Suite pilot checkpoint is outside this run")
        verify_checkpoint(checkpoint)
        if (
            report.get("status") != "completed"
            or report.get("step") != 4
            or report.get("fingerprint", {}).get("suite_map.json") != sha256(suite_map)
            or report.get("fingerprint", {}).get("source_commit")
            != source_identity(Path(__file__).resolve().parents[3])["git_commit"]
        ):
            raise ValueError("Suite main requires this source/map's completed four-step pilot")
    return phase, suite_map
