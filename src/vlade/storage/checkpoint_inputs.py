"""Pin already published HF run checkpoints as verified external model inputs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

from vlade.storage.assets import REGISTRY_REPO, NAME, configure_http, safe_relative


def checkpoint_record(repo_id: str, revision: str, prefix: str) -> dict:
    """Read only metadata; validate sealed file identities against one immutable HF commit."""
    if repo_id != REGISTRY_REPO or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Checkpoint inputs require the private VLADE repo and an immutable 40-character revision")
    safe_relative(prefix)
    if not re.fullmatch(r"runs/[^/]+/[^/]+/checkpoints/step-\d+", prefix):
        raise ValueError("Use runs/<group>/<output>/checkpoints/step-XXXXXXX from HF Files")
    configure_http()
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if info.private is not True or info.sha != revision:
        raise ValueError("Checkpoint repo must be private and match the pinned revision")
    inventory = {item.rfilename: item for item in info.siblings}

    def small_file(relative: str) -> bytes:
        item = inventory.get(prefix + "/" + relative)
        if item is None or item.size is None or item.size > 1024**2:
            raise ValueError("Missing or oversized checkpoint metadata: " + relative)
        path = Path(hf_hub_download(repo_id, prefix + "/" + relative, revision=revision))
        payload = path.read_bytes()
        if len(payload) != item.size:
            raise ValueError("HF checkpoint metadata size mismatch")
        return payload

    seal_bytes = small_file("checkpoint_manifest.json")
    complete = small_file("COMPLETE")
    seal = json.loads(seal_bytes)
    step = seal.get("step")
    if (seal.get("schema_version") != 1 or type(step) is not int or step < 1
            or int(prefix.rsplit("-", 1)[1]) != step
            or type(seal.get("microsteps")) is not int or seal["microsteps"] < step
            or not seal.get("files") or complete.strip() != b"ok"):
        raise ValueError("Incomplete or invalid HF checkpoint seal")
    files = copy.deepcopy(seal["files"])
    if set(files) & {"COMPLETE", "checkpoint_manifest.json"}:
        raise ValueError("Checkpoint seal cannot include itself or COMPLETE")
    for relative, expected in files.items():
        safe_relative(relative)
        if (type(expected.get("bytes")) is not int or expected["bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", expected.get("sha256", ""))):
            raise ValueError("Invalid sealed checkpoint file identity")
        item = inventory.get(prefix + "/" + relative)
        if item is None or item.size != expected["bytes"]:
            raise ValueError("HF checkpoint payload size mismatch: " + relative)
        if item.lfs:
            actual = getattr(item.lfs, "sha256", None)
            if actual is None and isinstance(item.lfs, dict):
                actual = item.lfs.get("sha256")
        else:
            actual = hashlib.sha256(small_file(relative)).hexdigest()
        if actual != expected["sha256"]:
            raise ValueError("HF checkpoint payload SHA256 mismatch: " + relative)
    actual_files = {name[len(prefix) + 1:] for name in inventory if name.startswith(prefix + "/")}
    if actual_files != set(files) | {"checkpoint_manifest.json", "COMPLETE"}:
        raise ValueError("HF checkpoint file set differs from its seal")
    for name, payload in (("checkpoint_manifest.json", seal_bytes), ("COMPLETE", complete)):
        files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    return {"repo_id": repo_id, "repo_type": "model", "revision": revision,
            "format": "hub_checkpoint", "prefix": prefix,
            "local_name": f"step-{step:07d}", "files": files}


def pin_checkpoint_inputs(base: dict, selections: dict[str, dict], output: Path) -> dict:
    """Write a runtime registry; no model download, remote write or committed lock mutation."""
    registry = copy.deepcopy(base)
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported base asset registry")
    for key, selection in selections.items():
        if not NAME.fullmatch(key) or ".." in key:
            raise ValueError("Invalid checkpoint asset key")
        record = checkpoint_record(selection["repo_id"], selection["revision"], selection["prefix"])
        if key in registry["assets"] and registry["assets"][key] != record:
            raise ValueError("Use a new asset key for a different checkpoint")
        registry["assets"][key] = record
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(registry, indent=2) + "\n"
    if output.exists() and json.loads(output.read_text(encoding="utf-8")) != registry:
        raise FileExistsError("Input selection changed; use a new run group/registry output")
    output.write_text(payload, encoding="utf-8")
    return registry
