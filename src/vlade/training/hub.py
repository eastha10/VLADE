"""Opt-in private Hub backup of sealed optimizer-boundary checkpoints.

Authentication is provided by HF_TOKEN/Colab Secrets or the standard local HF
credential store. No credentials are accepted in experiment configuration.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from vlade.training.run_control import verify_checkpoint

STATE = "hub_state.json"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) or ".." in value:
        raise ValueError("Hub run names must use letters, digits, underscore, dot or hyphen")
    return value


class HubCheckpoints:
    """Synchronous atomic commits; failed uploads remain in a durable FIFO queue.

    An explicit run group plus the output directory name separates pilot/main and
    model runs. Use one writer per group/output; keep both stable on resume.
    """

    def __init__(self, output: Path, *, resume: Path | None = None,
                 metric: str = "val_l1", api: Any = None) -> None:
        self.output = output
        self.repo = os.environ.get("VLADE_HUB_REPO", "")
        self.enabled = bool(self.repo)
        self.metric = metric
        self.best: dict[str, Any] | None = None
        self.api = api
        self.queue: list[dict[str, Any]] = []
        if not self.enabled:
            return
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", self.repo):
            raise ValueError("VLADE_HUB_REPO must be namespace/private-model-repository")
        group = _name(os.environ.get("VLADE_HUB_RUN_ID", ""))
        self.run = group + "/" + _name(output.name)
        self.prefix = "runs/" + self.run
        self.pending_path = output / "hub_pending.json"
        if resume and (resume / STATE).is_file():
            previous = json.loads((resume / STATE).read_text(encoding="utf-8"))
            if (previous["repo"], previous["run"], previous["metric"]) != (self.repo, self.run, metric):
                raise ValueError("Hub repository/run/metric differs from the resumed checkpoint")
            self.best = previous["best"]
        if self.pending_path.is_file():
            pending = json.loads(self.pending_path.read_text(encoding="utf-8"))
            if (pending["repo"], pending["run"]) != (self.repo, self.run):
                raise ValueError("Pending uploads belong to another Hub run")
            self.queue = pending["queue"]
        if self.api is None:
            from vlade.storage.assets import configure_http
            configure_http()
            from huggingface_hub import HfApi
            self.api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        # Fail before loading a model on authentication/private-repo mistakes.
        from vlade.training.loop import capture_rng, restore_rng
        rng = capture_rng()
        try:
            self.api.create_repo(repo_id=self.repo, repo_type="model", private=True, exist_ok=True)
            self._private()
        except Exception as error:
            raise RuntimeError(f"Private Hub connection failed ({type(error).__name__}); check authentication and repository privacy") from None
        finally:
            restore_rng(rng)

    def _private(self) -> None:
        # private=True on create_repo does not change an existing public repo.
        if self.api.model_info(self.repo).private is not True:
            raise ValueError("Hub uploads require an existing private repository; refusing public upload")

    def is_best(self, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.metric)
        return bool(self.enabled and value is not None and math.isfinite(value) and
                    (self.best is None or value < self.best["value"]))

    def prepare(self, stage: Path, step: int, metrics: dict[str, float]) -> None:
        """Add non-secret selection state before the adapter seals the checkpoint."""
        if not self.enabled:
            return
        best = {"step": step, "value": float(metrics[self.metric])} if self.is_best(metrics) else self.best
        _write(stage / STATE, {"schema_version": 1, "repo": self.repo, "run": self.run,
                              "metric": self.metric, "direction": "min", "best": best})

    def publish(self, checkpoint: Path, *, final: bool = False) -> None:
        """Queue only fully sealed local state; a failed transfer never loses it."""
        if not self.enabled:
            return
        seal = verify_checkpoint(checkpoint)
        if not (checkpoint / "trainer_state.pt").is_file() and not (checkpoint / "state.pt").is_file():
            raise ValueError("Hub checkpoints must contain resumable optimizer/scheduler state")
        state = json.loads((checkpoint / STATE).read_text(encoding="utf-8"))
        self.best = state["best"]
        item = {"checkpoint": str(checkpoint.resolve()), "step": seal["step"],
                "best": bool(self.best and self.best["step"] == seal["step"]), "final": final}
        if self.queue and self.queue[-1]["step"] == item["step"]:
            self.queue[-1]["final"] |= final
        else:
            self.queue.append(item)
        self._save_queue()
        self.flush()

    def _save_queue(self, *, error_type: str | None = None) -> None:
        _write(self.pending_path, {"schema_version": 1, "repo": self.repo, "run": self.run,
                                  "queue": self.queue, "last_error_type": error_type})

    def _commit(self, item: dict[str, Any]) -> None:
        from huggingface_hub import CommitOperationAdd
        checkpoint = Path(item["checkpoint"])
        seal = verify_checkpoint(checkpoint)
        if seal["step"] != item["step"]:
            raise ValueError("Pending checkpoint step mismatch")
        state = json.loads((checkpoint / STATE).read_text(encoding="utf-8"))
        if (state["repo"], state["run"]) != (self.repo, self.run):
            raise ValueError("Pending checkpoint belongs to another run")
        remote = f"{self.prefix}/checkpoints/step-{seal['step']:07d}"
        files = [*seal["files"], "checkpoint_manifest.json", "COMPLETE"]
        operations = []
        for relative in files:
            path = checkpoint / relative
            if path.is_symlink() or not path.resolve().is_relative_to(checkpoint.resolve()):
                raise ValueError("Hub checkpoints must contain their own files")
            sensitive = {"token", "tokens", "stored_tokens", "credentials", "credentials.json",
                         "hf_token", "hf_token.txt", "hf_token.json"}
            if any(part.startswith(".") or part.lower() in sensitive
                   for part in Path(relative).parts):
                raise ValueError("Credential/hidden files cannot be uploaded as checkpoints")
            operations.append(CommitOperationAdd(path_in_repo=remote + "/" + relative, path_or_fileobj=str(path)))
        pointer = {"schema_version": 1, "checkpoint": remote, "step": seal["step"],
                   "metric": state["metric"], "best": state["best"]}
        for alias in ["last"] + (["best"] if item["best"] else []) + (["final"] if item["final"] else []):
            operations.append(CommitOperationAdd(path_in_repo=f"{self.prefix}/{alias}.json",
                path_or_fileobj=io.BytesIO(json.dumps(pointer, allow_nan=False).encode())))
        self._private()
        self.api.create_commit(repo_id=self.repo, repo_type="model", operations=operations,
                               commit_message=f"VLADE {self.run} step {seal['step']}")

    def flush(self) -> bool:
        """Retry network failures; preserve FIFO aliases and sanitized diagnostics."""
        if not self.enabled:
            return True
        from vlade.training.loop import capture_rng, restore_rng
        rng = capture_rng()
        try:
            return self._flush()
        finally:
            restore_rng(rng)

    def _flush(self) -> bool:
        while self.queue:
            for attempt in range(3):
                try:
                    self._commit(self.queue[0])
                    break
                except Exception as error:
                    # Exception strings/HTTP details can contain request credentials.
                    if attempt == 2:
                        self._save_queue(error_type=type(error).__name__)
                        print(f"Hub upload pending ({type(error).__name__}); local checkpoint retained.", flush=True)
                        return False
                    time.sleep(2 ** attempt)
            self.queue.pop(0)
            self._save_queue()
        return True


def download_checkpoint(repo: str, run: str, output: Path, *, alias: str = "last",
                        revision: str = "main") -> Path:
    """Pin the pointer and all payload files to the same immutable Hub commit."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    if alias not in {"last", "best", "final"}:
        raise ValueError("Unknown Hub checkpoint alias")
    parts = run.split("/")
    if len(parts) != 2:
        raise ValueError("run must be group/output-directory-name")
    prefix = "runs/" + "/".join(_name(part) for part in parts)
    info = HfApi().model_info(repo, revision=revision)
    if info.private is not True:
        raise ValueError("Resume requires a private Hub repository")
    pointer_path = hf_hub_download(repo, f"{prefix}/{alias}.json", revision=info.sha)
    pointer = json.loads(Path(pointer_path).read_text(encoding="utf-8"))
    expected = f"{prefix}/checkpoints/step-{int(pointer['step']):07d}"
    if pointer["checkpoint"] != expected:
        raise ValueError("Invalid Hub checkpoint pointer")
    downloaded = snapshot_download(repo, revision=info.sha, allow_patterns=[expected + "/*"])
    source = Path(downloaded) / expected
    # HF cache payloads can be symlinks; materialize only this checkpoint locally.
    import shutil
    import uuid
    from vlade.utils.filesystem import publish_directory
    destination = output / f"step-{int(pointer['step']):07d}"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing resume checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    stage = output / f".hub-{uuid.uuid4().hex}.incomplete"
    shutil.copytree(source, stage, symlinks=False, ignore=shutil.ignore_patterns(".cache"))
    verify_checkpoint(stage)
    publish_directory(stage, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="VLADE private Hub checkpoint tools")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--output", type=Path, required=True)
    retry = sub.add_parser("retry")
    retry.add_argument("--output", type=Path, required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--run", required=True)
    fetch.add_argument("--alias", choices=["last", "best", "final"], default="last")
    fetch.add_argument("--revision", default="main")
    fetch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        print(download_checkpoint(args.repo, args.run, args.output, alias=args.alias, revision=args.revision))
    else:
        hub = HubCheckpoints(args.output)
        if not hub.enabled:
            raise ValueError("Set VLADE_HUB_REPO and VLADE_HUB_RUN_ID first")
        if args.command == "retry" and not hub.flush():
            raise SystemExit(1)
        print("Private Hub connection verified" if args.command == "check" else "All pending uploads completed")


if __name__ == "__main__":
    main()
