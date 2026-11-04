"""Copy approved Drive/local assets into private HF repositories without deleting sources."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

from vlade.storage.assets import REGISTRY_FILE, REGISTRY_REPO, configure_http, safe_relative, sha256

PREPARED_IDENTITY = {
    "manifest.json": "9a09c51dac170cb7452e0f3cd33f563e73429d6fe8f7a880daa8568ec2cf981b",
    "splits.json": "379376e5cfb19d0b9d0a5bdd548e01e4c658f09de3fdd1bff178b3aad7ed9900",
    "normalization.json": "e1dbf29ea83cc2e0fbcb90e68a7ad15660eb210baa5a333d20bd16c80d275af3",
}


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _private(api, repo: str, kind: str) -> None:
    api.create_repo(repo_id=repo, repo_type=kind, private=True, exist_ok=True)
    if api.repo_info(repo_id=repo, repo_type=kind).private is not True:
        raise ValueError("Migration refuses a public destination repository")


def _verify_remote(api, repo: str, kind: str, revision: str, sources: dict[str, Path], hashes: dict) -> None:
    info = api.repo_info(repo_id=repo, repo_type=kind, revision=revision, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    for relative, path in sources.items():
        item = remote.get(relative)
        if item is None or item.size != path.stat().st_size:
            raise ValueError("Uploaded asset size/file set mismatch")
        if item.lfs:
            actual = getattr(item.lfs, "sha256", None)
            if actual is None and isinstance(item.lfs, dict): actual = item.lfs.get("sha256")
            if actual != hashes[relative]:
                raise ValueError("Uploaded HF LFS SHA256 mismatch")
        else:
            digest = hashlib.sha1()
            digest.update(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""): digest.update(block)
            if digest.hexdigest() != item.blob_id:
                raise ValueError("Uploaded HF Git blob hash mismatch")


def update_registry(api, key: str, record: dict) -> dict:
    """Merge independently uploaded assets using a parent-commit compare-and-swap."""
    from huggingface_hub import CommitOperationAdd, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    _private(api, REGISTRY_REPO, "model")
    for attempt in range(4):
        head = api.model_info(REGISTRY_REPO).sha
        try:
            current = json.loads(Path(hf_hub_download(REGISTRY_REPO, REGISTRY_FILE, revision=head)).read_text())
        except EntryNotFoundError:
            current = {"schema_version": 1, "organization": "VLADE-konkuk", "assets": {}}
        existing = current["assets"].get(key)
        if existing and existing != record:
            # Same content at a new HF commit can safely reuse the already pinned asset.
            a = {k:v for k,v in existing.items() if k != "revision"}
            b = {k:v for k,v in record.items() if k != "revision"}
            if a != b: raise ValueError(f"Asset {key} already has different content; use a new asset key")
            return current
        current["assets"][key] = record
        try:
            api.create_commit(repo_id=REGISTRY_REPO, repo_type="model", parent_commit=head,
                operations=[CommitOperationAdd(path_in_repo=REGISTRY_FILE, path_or_fileobj=io.BytesIO(_json_bytes(current)))],
                commit_message="Register verified VLADE asset " + key)
            return current
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status not in {409, 412} or attempt == 3: raise
    raise RuntimeError("Concurrent registry update did not settle")


def publish_files(api, key: str, sources: dict[str, Path], *, repo: str, kind: str,
                  local_name: str = "content", expected: dict | None = None) -> dict:
    from huggingface_hub import CommitOperationAdd
    _private(api, repo, kind)
    prefix = "assets/" + key
    rows, operations, remote_sources, hashes = {}, [], {}, {}
    for relative, path in sorted(sources.items()):
        safe_relative(relative)
        if not path.is_file() or path.is_symlink(): raise ValueError("Regular source files required")
        digest = sha256(path)
        row = {"bytes": path.stat().st_size, "sha256": digest}
        if expected and relative in expected and row != expected[relative]:
            raise ValueError(f"Source checkpoint seal mismatch: {relative}")
        rows[relative] = row
        remote = prefix + "/files/" + relative
        remote_sources[remote] = path
        hashes[remote] = digest
        operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)))
        print(json.dumps({"asset":key,"verified_source":relative,"bytes":row["bytes"]}), flush=True)
    metadata = {"schema_version":1,"key":key,"files":rows}
    operations.append(CommitOperationAdd(path_in_repo=prefix+"/asset_manifest.json", path_or_fileobj=io.BytesIO(_json_bytes(metadata))))
    commit = api.create_commit(repo_id=repo, repo_type=kind, operations=operations,
                               commit_message="Import verified VLADE asset " + key)
    _verify_remote(api, repo, kind, commit.oid, remote_sources, hashes)
    record = {"repo_id":repo,"repo_type":kind,"revision":commit.oid,"format":"directory",
              "prefix":prefix,"local_name":local_name,"files":rows}
    registry = update_registry(api, key, record)
    print(json.dumps({"asset":key,"status":"published_verified","revision":record["revision"]}), flush=True)
    return registry


def publish_prepared(api, archive: Path) -> dict:
    from huggingface_hub import CommitOperationAdd
    if archive.stat().st_size != 11_371_123_200:
        raise ValueError("Expected the confirmed canonical D archive byte size")
    identity, root = {}, None
    with tarfile.open(archive,"r:*") as stream:
        for member in stream:
            safe_relative(member.name.rstrip("/"))
            if not (member.isdir() or member.isfile()): raise ValueError("Unsafe TAR member")
            name = Path(member.name).name
            if name in PREPARED_IDENTITY and member.isfile():
                with stream.extractfile(member) as payload:
                    digest = hashlib.sha256(payload.read()).hexdigest()
                if digest != PREPARED_IDENTITY[name]: raise ValueError("Archive differs from canonical D identity")
                candidate = Path(member.name).parent.as_posix()
                if root is not None and root != candidate: raise ValueError("Mixed prepared dataset roots")
                root = candidate
                identity[name] = {"bytes":member.size,"sha256":digest}
    if set(identity) != set(PREPARED_IDENTITY): raise ValueError("Canonical D identity files missing")
    key = "prepared_d_seed42"
    repo = "VLADE-konkuk/libero-plus-prepared"
    remote = "assets/" + key + "/libero_plus_episode80_d_seed42.tar"
    digest = sha256(archive)
    record = {"repo_id":repo,"repo_type":"dataset","format":"archive","archive_path":remote,
        "archive_bytes":archive.stat().st_size,"archive_sha256":digest,"archive_root":"" if root=="." else root,
        "identity_files":identity,"local_name":"libero_plus_episode80_d_seed42"}
    _private(api,repo,"dataset")
    commit = api.create_commit(repo_id=repo,repo_type="dataset",operations=[
        CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(archive)),
        CommitOperationAdd(path_in_repo="assets/"+key+"/asset_manifest.json",path_or_fileobj=io.BytesIO(_json_bytes(record)))],
        commit_message="Import canonical D prepared dataset with identity verification")
    _verify_remote(api,repo,"dataset",commit.oid,{remote:archive},{remote:digest})
    record["revision"] = commit.oid
    result = update_registry(api,key,record)
    print(json.dumps({"asset":key,"status":"published_verified","revision":commit.oid}),flush=True)
    return result


def checkpoint_sources(directory: Path) -> tuple[dict,dict]:
    """Restore Drive copy suffixes through the checkpoint seal, without renaming Drive files."""
    candidates = {}
    for path in directory.rglob("*"):
        if not path.is_file(): continue
        canonical = "/".join(re.sub(r"(?:의 사본| - Copy)$", "", part).rstrip()
                              for part in path.relative_to(directory).parts)
        if canonical in candidates: raise ValueError("Ambiguous copied checkpoint file names")
        candidates[canonical] = path
    if "checkpoint_manifest.json" not in candidates or "COMPLETE" not in candidates:
        raise ValueError("Completed sealed checkpoint required")
    seal = json.loads(candidates["checkpoint_manifest.json"].read_text(encoding="utf-8"))
    names = set(seal["files"]) | {"checkpoint_manifest.json","COMPLETE"}
    if not names <= set(candidates): raise ValueError("Copied checkpoint is missing sealed files")
    if not {"trainer_state.pt","state.pt"} & names: raise ValueError("Resumable checkpoint state required")
    return {name:candidates[name] for name in names},seal


def model_sources(directory: Path) -> dict[str,Path]:
    allowed = {".safetensors",".bin",".pt",".model",".json",".py",".txt",".jinja",".md"}
    result = {}
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if (not path.is_file() or path.is_symlink() or any(p.startswith(".") or p in {"logs","wandb"} for p in relative.parts)
                or ".back." in path.name or path.suffix not in allowed and path.name not in {"LICENSE","NOTICE"}): continue
        result[relative.as_posix()] = path
    if not any(name.endswith(".safetensors") and "adapter" not in name for name in result):
        raise ValueError("Initial model folder has no complete base weights")
    index = result.get("model.safetensors.index.json")
    if index:
        expected = set(json.loads(index.read_text())["weight_map"].values())
        if not expected <= set(result): raise ValueError("Initial model shards are missing")
    return result


def migrate(root: Path, *, output_registry: Path | None = None, checkpoint: Path | None = None) -> None:
    configure_http()
    from huggingface_hub import HfApi
    api = HfApi()
    account = api.whoami()
    if "VLADE-konkuk" not in {org["name"] for org in account.get("orgs",[])}:
        raise PermissionError("HF token account must belong to VLADE-konkuk")
    archive = root / "data/splits/libero_plus_episode80_d_seed42.tar"
    if not archive.is_file(): raise FileNotFoundError("Select the project root containing the canonical D TAR")
    print(json.dumps({'status':'checking_prepared_archive','bytes':archive.stat().st_size}),flush=True)
    registry = publish_prepared(api,archive)
    if output_registry:
        output_registry.parent.mkdir(parents=True,exist_ok=True)
        output_registry.write_bytes(_json_bytes(registry))
    teacher = root / "checkpoints/teacher/avavla-libero-4in1"
    if teacher.is_dir():
        registry = publish_files(api,"ava_t0",model_sources(teacher),repo=REGISTRY_REPO,kind="model",local_name="avavla-libero-4in1")
    checkpoints = [checkpoint] if checkpoint else sorted({p.parent for p in (root/"checkpoints/fft").rglob("checkpoint_manifest.json*")})
    if not checkpoints: raise FileNotFoundError("No completed FFT checkpoints found under the selected project root")
    for folder in checkpoints:
        sources,seal = checkpoint_sources(folder)
        key = "ava_fft_step_"+str(seal["step"])
        if seal['step'] != 1975:
            key += "_" + sha256(sources["checkpoint_manifest.json"])[:8]
        registry = publish_files(api,key,sources,repo=REGISTRY_REPO,kind="model",
            local_name=f"step-{seal['step']:07d}",expected=seal["files"])
        if output_registry: output_registry.write_bytes(_json_bytes(registry))
    if "ava_fft_step_1975" not in registry["assets"]:
        raise FileNotFoundError("The confirmed step-1975 checkpoint has not been migrated")
    print(json.dumps({"status":"migration_completed","assets":sorted(registry["assets"])}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output-registry",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    args=parser.parse_args()
    try: migrate(args.root,output_registry=args.output_registry,checkpoint=args.checkpoint)
    except Exception as error:
        print(json.dumps({"status":"migration_failed","error_type":type(error).__name__}),flush=True)
        raise SystemExit(1) from None


if __name__=="__main__":main()
