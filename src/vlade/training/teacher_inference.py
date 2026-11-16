"""Frozen T1 inference in its own AVA runtime, without a persisted action cache."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch
    from vlade.data.episode_data import EpisodeData


class TeacherPolicy(Protocol):
    def reset(self) -> None: ...
    def predict(self, sample: dict, data: EpisodeData) -> tuple[torch.Tensor, torch.Tensor]: ...


def teacher_identity(checkpoint: str | Path, data: EpisodeData) -> tuple[Path, dict]:
    """Verify HF provenance, checkpoint seal, FFT role and normalization before GPU use."""
    from vlade.storage.assets import resolve_input, sha256
    from vlade.training.run_control import verify_checkpoint

    root = resolve_input(checkpoint)
    seal = verify_checkpoint(root)
    metadata = json.loads((root / "vlade_fft.json").read_text(encoding="utf-8"))
    expected = {"manifest_sha256": sha256(data.root / "manifest.json"),
                **{k: data.manifest[k] for k in ("normalization_sha256", "split_sha256")}}
    if (metadata.get("model") != "ava_vla" or metadata.get("step") != seal["step"]
            or metadata.get("prepared_data_identity") != expected
            or metadata.get("quantized", False) is not False):
        raise ValueError("T1 must be a completed, unquantized AVA FFT checkpoint with matching data")
    layer = metadata.get("configuration", {}).get("temporal_feature_layer", -2)
    if type(layer) is not int:
        raise ValueError("Invalid T1 temporal feature layer")
    if not (root / "config.json").is_file() or not any((root / name).is_file() for name in
            ("model.safetensors", "model.safetensors.index.json")):
        raise ValueError("T1 must include its complete backbone weights and config")
    from vlade.training.fft import find_component
    for name in ("action_head", "proprio_projector", "vision_attn_weight_generator"):
        find_component(root, name)
    return root, {"checkpoint_manifest_sha256": sha256(root / "checkpoint_manifest.json"),
                  "step": seal["step"], "temporal_mode": "episode_context", "frozen": True,
                  "temporal_feature_layer": layer}


class EpisodeTeacher:
    """Replay causal prefixes on resume/reordering; never carry state across episodes."""

    def __init__(self, teacher: TeacherPolicy, data: EpisodeData) -> None:
        self.teacher, self.data = teacher, data
        self.episode = None
        self.samples = []
        self.cursor = -1

    def get(self, episode: str, timestep: int) -> torch.Tensor:
        if type(timestep) is not int or timestep < 0 or timestep % 8:
            raise ValueError("Expected an eight-step query timestep")
        if episode != self.episode:
            _, self.samples = next(self.data.episodes("train", order=[episode]))
            self.episode, self.cursor = episode, -1
            self.teacher.reset()
        index = timestep // 8
        if index >= len(self.samples) or self.samples[index]["timestep"] != timestep:
            raise ValueError("Teacher request is outside the verified train queries")
        if index <= self.cursor:
            self.teacher.reset()
            self.cursor = -1
        action = None
        while self.cursor < index:
            self.cursor += 1
            action, _ = self.teacher.predict(self.samples[self.cursor], self.data)
        return action


class InferenceActions:
    """JSON pipes isolate incompatible AVA and Qwen Transformers environments."""

    def __init__(self, settings: dict, data: EpisodeData) -> None:
        root, self.signature = teacher_identity(settings["teacher_checkpoint"], data)
        python = Path(settings["teacher_python"])
        if not python.is_file():
            raise FileNotFoundError("Set up the AVA inference runtime before Action KD")
        self.timeout = settings.get("teacher_timeout_seconds", 600)
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("teacher_timeout_seconds must be finite and positive")
        code = Path(__file__).resolve().parents[3]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": os.pathsep.join((str(code / "src"), str(code)))}
        self.process = subprocess.Popen([str(python), "-B", "-m", "vlade.training.teacher_inference",
            "--checkpoint", str(root), "--data", str(data.root),
            "--device", settings.get("teacher_device", "cuda:0")], cwd=code, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self.replies = queue.Queue()

        def read():
            for line in self.process.stdout:
                self.replies.put(line)
            self.replies.put(None)

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            if self._receive() != {"ready": self.signature}:
                raise ValueError("T1 inference worker identity mismatch")
        except BaseException:
            self.close()
            raise

    def _receive(self) -> dict:
        try:
            line = self.replies.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError("T1 inference timed out") from error
        if line is None:
            raise RuntimeError("T1 inference worker exited; see its stderr")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError("T1 inference failed: " + result["error"])
        return result

    def get(self, sample: dict) -> torch.Tensor:
        import torch

        request = {"episode_id": sample["episode_id"], "timestep": sample["timestep"]}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self._receive()
        if any(result.get(key) != value for key, value in request.items()):
            raise ValueError("T1 action does not match the requested observation")
        action = torch.tensor(result["action"], dtype=torch.float32)
        if action.shape != (8, 7) or not torch.isfinite(action).all():
            raise ValueError("T1 must predict finite normalized actions [8,7]")
        return action.detach()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    wire = sys.stdout

    def reply(value: dict) -> None:
        wire.write(json.dumps(value) + "\n")
        wire.flush()

    # Model/data libraries sometimes print to stdout; reserve it for the protocol.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from vlade.storage.assets import resolve_input
            from vlade.data.episode_data import EpisodeData
            from vlade.training.fixed_cache import SequentialTeacher

            data = EpisodeData(resolve_input(args.data))
            root, signature = teacher_identity(args.checkpoint, data)
            teacher = EpisodeTeacher(SequentialTeacher(root, device=args.device,
                temporal_feature_layer=signature["temporal_feature_layer"]), data)
            reply({"ready": signature})
            for line in sys.stdin:
                request = json.loads(line)
                action = teacher.get(request["episode_id"], request["timestep"])
                reply({**request, "action": action.detach().float().cpu().tolist()})
        except Exception as error:
            reply({"error": type(error).__name__})
            raise


if __name__ == "__main__":
    main()
