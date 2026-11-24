"""Frozen T1 action and optional approach-plan inference in its own AVA runtime."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import traceback
import uuid
from typing import TYPE_CHECKING, Protocol
from vlade.distillation.teacher_plan import InvalidTeacherPlan, normalize_plan, plan_token_budget, require_plan_supervision
from vlade.utils.timing import RunTimings

if TYPE_CHECKING:
    import torch
    from vlade.data.episode_data import EpisodeData


class TeacherPolicy(Protocol):
    def reset(self) -> None: ...
    def predict(self, sample: dict, data: EpisodeData) -> tuple[torch.Tensor, torch.Tensor]: ...
    def plan(self, sample: dict, data: EpisodeData, *, max_new_tokens: int) -> str: ...


@dataclass(frozen=True)
class TeacherSignal:
    """Paired targets from the exact query sent to T1."""

    action: torch.Tensor
    plan: str


def teacher_identity(checkpoint: str | Path, data: EpisodeData) -> tuple[Path, dict]:
    """Verify HF provenance, checkpoint seal, FFT role and normalization before GPU use."""
    from vlade.storage.assets import resolve_input, sha256
    from vlade.training.run_control import verify_checkpoint

    root = resolve_input(checkpoint)
    seal = verify_checkpoint(root)
    metadata = json.loads((root / "vlade_fft.json").read_text(encoding="utf-8"))
    expected = {"manifest_sha256": sha256(data.root / "manifest.json"),
                **{k: data.manifest[k] for k in ("normalization_sha256", "split_sha256")}}
    recorded = metadata.get("prepared_data_identity")
    for key in ("canonical_manifest_sha256", "canonical_normalization_sha256"):
        if metadata.get("plan_supervision") is not None or isinstance(recorded, dict) and key in recorded:
            expected[key] = data.manifest[key]
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
                  "plan_supervision": metadata.get("plan_supervision"),
                  "temporal_feature_layer": layer}


class EpisodeTeacher:
    """Replay causal prefixes on resume/reordering; never carry state across episodes."""

    def __init__(self, teacher: TeacherPolicy, data: EpisodeData,
                 timings: RunTimings | None = None) -> None:
        self.teacher, self.data = teacher, data
        self.timings = timings
        self.predictions = 0
        self.episode = None
        self.samples = []
        self.cursor = -1
        self.plan_episode = None
        self.plan_samples = []

    @staticmethod
    def _query_index(timestep: int) -> int:
        if type(timestep) is not int or timestep < 0 or timestep % 8:
            raise ValueError("Expected an eight-step query timestep")
        return timestep // 8

    def get(self, episode: str, timestep: int) -> torch.Tensor:
        index = self._query_index(timestep)
        if episode != self.episode:
            _, self.samples = next(self.data.episodes("train", order=[episode]))
            self.episode, self.cursor = episode, -1
            self.teacher.reset()
        if index >= len(self.samples) or self.samples[index]["timestep"] != timestep:
            raise ValueError("Teacher request is outside the verified train queries")
        if index <= self.cursor:
            self.teacher.reset()
            self.cursor = -1
        action = None
        while self.cursor < index:
            self.cursor += 1
            with (self.timings.measure("t1_inference", episode_id=episode,
                                      timestep=self.samples[self.cursor]["timestep"],
                                      replay=self.cursor < index, is_warmup=self.predictions < 2)
                  if self.timings is not None else contextlib.nullcontext()):
                action, _ = self.teacher.predict(self.samples[self.cursor], self.data)
            self.predictions += 1
        return action

    def plan(self, episode: str, timestep: int, *, split: str, max_new_tokens: int) -> str:
        """Generate text from one current query without moving the action replay cursor."""
        index = self._query_index(timestep)
        if split not in {"train", "validation"}:
            raise ValueError("Teacher plans are limited to train and validation")
        if split == "train" and episode == self.episode:
            samples = self.samples
        else:
            key = split, episode
            if key != self.plan_episode:
                _, self.plan_samples = next(self.data.episodes(split, order=[episode]))
                self.plan_episode = key
            samples = self.plan_samples
        if index >= len(samples) or samples[index]["timestep"] != timestep:
            raise ValueError("Teacher plan request is outside the verified queries")
        return self.teacher.plan(samples[index], self.data, max_new_tokens=max_new_tokens)


class InferenceActions:
    """JSON pipes isolate incompatible AVA and Qwen Transformers environments."""

    def __init__(self, settings: dict, data: EpisodeData,
                 timings: RunTimings | None = None, *, live: bool = False) -> None:
        self.timings = timings
        self.plan_tokens = plan_token_budget(settings)
        root, self.signature = teacher_identity(settings["teacher_checkpoint"], data)
        if self.plan_tokens:
            require_plan_supervision({"step": self.signature["step"],
                                      "plan_supervision": self.signature.get("plan_supervision")})
        python = Path(settings["teacher_python"])
        if not python.is_file():
            raise FileNotFoundError("Set up the AVA inference runtime before Action KD")
        self.timeout = settings.get("teacher_timeout_seconds", 600)
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("teacher_timeout_seconds must be finite and positive")
        code = Path(__file__).resolve().parents[3]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": os.pathsep.join((str(code / "src"), str(code)))}
        timing_args = (["--timing-dir", str(timings.path.parent), "--timing-session", timings.session_id]
                       if timings is not None else [])
        plan_args = ["--plan-max-new-tokens", str(self.plan_tokens)] if self.plan_tokens else []
        live_args = ["--live-observations"] if live else []
        self.process = subprocess.Popen([str(python), "-B", "-m", "vlade.training.teacher_inference",
            "--checkpoint", str(root), "--data", str(data.root),
            "--device", settings.get("teacher_device", "cuda:0"), *timing_args, *plan_args, *live_args], cwd=code, env=env,
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
            missing = result.get("missing_module")
            detail = f" ({missing})" if missing else ""
            if result.get("error") == "InvalidTeacherPlan" and isinstance(result.get("error_message"), str):
                detail += ": " + result["error_message"]
            raise RuntimeError("T1 inference failed: " + result["error"] + detail)
        return result

    def _query(self, sample: dict, **options: str) -> dict:
        request = {"episode_id": sample["episode_id"], "timestep": sample["timestep"]}
        with (self.timings.measure("t1_request", **request) if self.timings is not None else contextlib.nullcontext()):
            request.update(options)
            result = self._send(request)
        if any(result.get(key) != value for key, value in request.items()):
            raise ValueError("T1 response does not match the requested observation")
        return result

    def _send(self, request: dict) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self._receive()

    def reset_live(self) -> None:
        if self._send({"live": "reset"}) != {"live": "reset"}:
            raise ValueError("T1 live reset was not acknowledged")

    def observe_live(self, query, *, action: bool, plan: bool) -> tuple[torch.Tensor | None, str | None]:
        from vlade.evaluation.plan_observations import encode_query

        request = {"live": "query", "action": action, "plan": plan, **encode_query(query)}
        result = self._send(request)
        if (result.get("task_id"), result.get("timestep")) != (query.task_id, query.timestep):
            raise ValueError("T1 response does not match the live observation")
        text = normalize_plan(result["plan"]) if plan and result["plan"] is not None else None
        return self._action(result) if action else None, text

    @staticmethod
    def _action(result: dict) -> torch.Tensor:
        import torch

        action = torch.tensor(result["action"], dtype=torch.float32)
        if action.shape != (8, 7) or not torch.isfinite(action).all():
            raise ValueError("T1 must predict finite normalized actions [8,7]")
        return action.detach()

    def get(self, sample: dict) -> torch.Tensor:
        return self._action(self._query(sample))

    def get_signal(self, sample: dict) -> TeacherSignal:
        if not self.plan_tokens:
            raise ValueError("T1 plan generation is disabled")
        result = self._query(sample, signal="action_and_plan")
        return TeacherSignal(self._action(result), normalize_plan(result.get("plan")))

    def get_plan(self, sample: dict, *, split: str) -> str:
        if not self.plan_tokens or split not in {"train", "validation"}:
            raise ValueError("T1 plan requests require a configured train or validation query")
        result = self._query(sample, signal="plan", split=split)
        return normalize_plan(result.get("plan"))

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


def probe_teacher_plans(settings: dict, data_reference: str, output: Path) -> Path:
    """Check at most two causal queries per split with T1 only, before loading S0."""
    from vlade.data.episode_data import EpisodeData
    from vlade.storage.assets import resolve_input, sha256

    if not plan_token_budget(settings):
        raise ValueError("The plan probe requires online CoT distillation")
    data = EpisodeData(resolve_input(data_reference))
    selected = {}
    for split in ("train", "validation"):
        first = next(data.episodes(split), None)
        if first is None or not first[1]:
            raise ValueError(f"The plan probe requires a nonempty {split} split")
        selected[split] = first[1][:2]
    output.mkdir(parents=True, exist_ok=True)
    report = output / f"teacher-plan-probe-{uuid.uuid4().hex[:12]}.jsonl"
    with report.open("x", encoding="utf-8") as stream:
        def record(row: dict) -> None:
            print(json.dumps(row, ensure_ascii=False), file=stream, flush=True)

        try:
            with contextlib.closing(InferenceActions(settings, data)) as teacher:
                record({"status": "identity", "teacher": teacher.signature,
                        "data_manifest_sha256": sha256(data.root / "manifest.json"),
                        "teacher_plan_max_new_tokens": teacher.plan_tokens,
                        "scope": "format_and_action_contract; semantic quality needs review"})
                for split, samples in selected.items():
                    for sample in samples:
                        row = {key: sample[key] for key in ("episode_id", "timestep", "instruction")}
                        row["split"] = split
                        try:
                            if split == "train":
                                signal = teacher.get_signal(sample)
                                plan = signal.plan
                                row["action_shape"] = list(signal.action.shape)
                                row["action_status"] = "predicted"
                            else:
                                plan = teacher.get_plan(sample, split=split)
                                row.update(action_shape=None, action_status="not_requested_plan_only")
                        except (RuntimeError, ValueError, TimeoutError) as error:
                            record({**row, "status": "plan_error", "error": str(error)})
                            raise
                        record({**row, "status": "ok", "plan": plan})
        except (RuntimeError, ValueError, TimeoutError) as error:
            record({"status": "failed", "error": str(error)})
            print(f"T1 plan probe failed; report: {report}", file=sys.stderr, flush=True)
            raise
        record({"status": "complete", "queries": sum(len(rows) for rows in selected.values())})
    print(f"T1 plan probe completed; semantic review required: {report}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--timing-session")
    parser.add_argument("--plan-max-new-tokens", type=int, default=0)
    parser.add_argument("--live-observations", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan_tokens = plan_token_budget({"teacher_mode": "inference", "teacher_plan_max_new_tokens": args.plan_max_new_tokens})
    wire = sys.stdout

    def reply(value: dict) -> None:
        wire.write(json.dumps(value) + "\n")
        wire.flush()

    # Model/data libraries sometimes print to stdout; reserve it for the protocol.
    with contextlib.redirect_stdout(sys.stderr), contextlib.ExitStack() as stack:
        try:
            from vlade.storage.assets import resolve_input
            from vlade.data.episode_data import EpisodeData
            from vlade.training.fixed_cache import SequentialTeacher
            import torch

            data = EpisodeData(resolve_input(args.data))
            root, signature = teacher_identity(args.checkpoint, data)
            timings = (stack.enter_context(RunTimings(args.timing_dir, role="t1", device=args.device,
                session_id=args.timing_session, metadata={
                    "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu",
                    "dtype": "bfloat16", "batch_size": 1, "action_chunk_size": 8,
                    "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "checkpoint_manifest_sha256": signature["checkpoint_manifest_sha256"],
                })) if args.timing_dir is not None else None)
            with (timings.measure("t1_model_load") if timings is not None else contextlib.nullcontext()):
                policy = SequentialTeacher(root, device=args.device,
                    temporal_feature_layer=signature["temporal_feature_layer"], timings=timings)
            teacher = EpisodeTeacher(policy, data, timings=timings)
            reply({"ready": signature})
            for line in sys.stdin:
                request = json.loads(line)
                if args.live_observations:
                    from vlade.evaluation.plan_observations import live_response

                    reply(live_response(policy, request, plan_tokens))
                    continue
                episode, timestep = request["episode_id"], request["timestep"]
                signal = request.get("signal", "action")
                if signal not in {"action", "action_and_plan", "plan"} or (signal != "action" and not plan_tokens):
                    raise ValueError("Unsupported T1 signal request")
                response = dict(request)
                if signal != "plan":
                    action = teacher.get(episode, timestep)
                    response["action"] = action.detach().float().cpu().tolist()
                if signal != "action":
                    split = request["split"] if signal == "plan" else "train"
                    response["plan"] = teacher.plan(episode, timestep, split=split, max_new_tokens=plan_tokens)
                reply(response)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            failure = {
                "error": type(error).__name__,
                "missing_module": error.name if isinstance(error, ModuleNotFoundError) else None,
            }
            if isinstance(error, InvalidTeacherPlan):
                failure["error_message"] = str(error)
            reply(failure)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
