"""Fetch only pinned T1 metadata and check config against its asset manifest."""
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
TASK = Path(__file__).resolve().parents[1]
REVISION = "1fdc672416f2129624b22022e6977345a3b429e3"
REGISTRY_REVISION = "195d32e54c7f48960411c32963d91746183de488"
REPO = "VLADE-konkuk/VLADE-checkpoints"


def main():
    token_path = Path(os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token")))
    token = os.environ.get("HF_TOKEN") or token_path.read_text().strip()

    def get(path, revision):
        # Credentials go through stdin, never the process argument list or a file.
        options = (f'header = "Authorization: Bearer {token}"\n'
                   f'url = "https://huggingface.co/{REPO}/resolve/{revision}/{path}"\n')
        payload = subprocess.run(["curl.exe", "-fsSL", "--tlsv1.2", "--tls-max", "1.2",
                                  "--retry", "2", "--retry-all-errors", "--max-time", "20", "--config", "-"],
                                 input=options.encode(), stdout=subprocess.PIPE, check=True).stdout
        assert len(payload) <= 1024**2
        return payload

    registry = json.loads(get("assets/registry.json", REGISTRY_REVISION))
    record = registry["assets"]["T1"]
    assert record["revision"] == REVISION and record["repo_id"] == REPO
    prefix = record["prefix"] + ("/files" if record["format"] == "directory" else "")
    config_bytes = get(prefix + "/config.json", REVISION)
    identity = {"bytes": len(config_bytes), "sha256": hashlib.sha256(config_bytes).hexdigest()}
    assert identity == record["files"]["config.json"]
    def metadata(name):
        payload = get(prefix + "/" + name, REVISION)
        assert {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} == record["files"][name]
        return json.loads(payload)
    seal = metadata("checkpoint_manifest.json")
    assert seal["files"]["config.json"] == identity
    fft = metadata("vlade_fft.json")
    target = TASK / "scratch/t1"
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_bytes(config_bytes)
    config = json.loads(config_bytes)
    text = config.get("text_config")
    report = {"repo": REPO, "revision": REVISION, "registry_revision": REGISTRY_REVISION,
              "prefix": prefix, "config_identity": identity,
              "top_level_keys": sorted(config),
              "identifiers": {key: config.get(key) for key in
                              ("model_type", "architectures", "llm_backbone_id", "hf_llm_id")},
              "text_config": text,
              "checkpoint_step": seal["step"],
              "temporal_feature_layer": fft.get("configuration", {}).get("temporal_feature_layer", -2)}
    if "--contract" in sys.argv:
        sys.path.insert(0, str(ROOT / "src"))
        from vlade.distillation.representation_config import representation_contract
        report["contract"] = representation_contract(target, student_dim=1024,
                                                       student_layer=-2, teacher_layer=report["temporal_feature_layer"],
                                                       teacher_python=Path(sys.executable))
        import importlib.metadata
        report["validation_environment"] = {"python": sys.version.split()[0],
                                               "transformers": importlib.metadata.version("transformers"),
                                               "gpu_forward_tested": False}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    reports = TASK / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "t1-config.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
