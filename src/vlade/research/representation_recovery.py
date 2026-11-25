"""Source replacement for a representation run that stopped before training."""
from __future__ import annotations

import json
from pathlib import Path
import re

from vlade.distillation.representation_config import representation_settings
from vlade.utils.source_identity import source_identity


def recover_preflight_source(code: Path, workspace: Path, group: str, previous_commit: str) -> None:
    """Keep settings/pins/group while preserving the failed preflight's source record.

    This is not checkpoint migration. Any local output or remote run payload
    prevents replacement; trained runs retain the normal exact-source contract.
    """
    from vlade.research.distillation import load_settings, prepare_inputs
    from vlade.storage.assets import REGISTRY_REPO, configure_http

    if (not re.fullmatch(r"[0-9a-f]{40}", previous_commit)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", group)):
        raise ValueError("Invalid preflight source or run group")
    folder = workspace.resolve() / "runs" / group
    path = folder / "run-contract.json"
    original = path.read_bytes()
    previous = json.loads(original)
    workflow, variants = load_settings(folder / "representation-config.json", {})
    expected = {"workflow": workflow, "variants": variants, "run_group": group}
    if (list(variants) != ["gt_kd"] or representation_settings(variants["gt_kd"].get("distillation", {})) is None):
        raise ValueError("Only a plan-free representation preflight can be recovered")
    source = source_identity(code, require_clean=True)
    backup = folder / f"preflight-source-{previous_commit}.json"
    if previous == {**expected, **source} and backup.is_file():
        if json.loads(backup.read_bytes()) != {**expected, "git_commit": previous_commit}:
            raise ValueError("Previous preflight record differs")
        return  # An interrupted preflight may already have completed the transition.
    if previous != {**expected, "git_commit": previous_commit} or source["git_commit"] == previous_commit:
        raise ValueError("Preflight source/settings differ from the requested transition")
    allowed = {"run-contract.json", "representation-config.json", "gt_kd.json", "inputs.json", backup.name}
    if any(item.name not in allowed or not item.is_file() or item.is_symlink() for item in folder.iterdir()):
        raise ValueError("Run has local output or unknown files; source replacement is forbidden")
    if json.loads((folder / "gt_kd.json").read_text(encoding="utf-8")) != variants["gt_kd"]:
        raise ValueError("Saved preflight settings differ")
    if not (folder / "inputs.json").is_file():
        raise ValueError("The failed preflight must already have pinned inputs")
    # Existing registry comparison refuses changed revisions or file identities.
    prepare_inputs(workspace, folder, workflow, variants)
    configure_http()
    from huggingface_hub import HfApi

    info = HfApi().model_info(REGISTRY_REPO)
    if info.private is not True:
        raise ValueError("Preflight recovery requires the private checkpoint repository")
    prefix = f"runs/{group}/"
    if any(item.rfilename.startswith(prefix) for item in info.siblings):
        raise ValueError("Run has remote output; source replacement is forbidden")
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError("Preflight source backup already contains another record")
    else:
        with backup.open("xb") as stream:
            stream.write(original)
    # Preserve the old bytes before changing only the source commit in the live contract.
    temporary = folder / "run-contract.recovery.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump({**expected, **source}, stream, indent=2)
    if path.read_bytes() != original:
        temporary.unlink()
        raise RuntimeError("Run contract changed during recovery")
    temporary.replace(path)
    print(f"Recovered preflight source: {previous_commit} -> {source['git_commit']}", flush=True)
