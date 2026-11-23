"""Evaluate T1 plan alignment and paired S0 plan interventions on LIBERO+."""
from __future__ import annotations

import csv
from collections import deque
from contextlib import closing
import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
import torch

from vlade.evaluation.s0_libero_rollout import prepare_rollouts, run_query_rollouts


CRITERIA = ("target_motion", "gripper", "order")
LABELS = {"match", "mismatch", "unclear", "na"}
CONDITIONS = ("current_plan", "lagged_plan", "no_plan", "action_only_kd")


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_alignment(path: Path) -> dict:
    """Report per-criterion agreement and show all unreviewed/unclear counts."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts = {name: dict.fromkeys(sorted(LABELS | {"unreviewed"}), 0) for name in CRITERIA}
    for row in rows:
        for name in CRITERIA:
            value = row[name].strip().lower() or "unreviewed"
            if value not in counts[name]:
                raise ValueError(f"{row['chunk_id']} has an unknown label in {name}: {value}")
            if row["plan_status"] == "invalid" and value not in {"unclear", "na", "unreviewed"}:
                raise ValueError(f"{row['chunk_id']} has no usable plan to score")
            counts[name][value] += 1
    result = {}
    for name, values in counts.items():
        judged = values["match"] + values["mismatch"]
        result[name] = {**values, "decidable": judged,
                        "agreement": values["match"] / judged if judged else None}
    return {"captured": len(rows), "invalid_plans": sum(r["plan_status"] == "invalid" for r in rows),
            "criteria": result}


def _save_clip(path: Path, frames: list) -> None:
    combined = []
    for primary, wrist in frames:
        image = Image.new("RGB", (primary.width + wrist.width, primary.height))
        image.paste(primary, (0, 0))
        image.paste(wrist, (primary.width, 0))
        combined.append(image)
    combined[0].save(path, save_all=True, append_images=combined[1:], duration=160, loop=0)


def evaluate_alignment(settings: dict, rollout, teacher, stats: dict, output: Path) -> dict:
    """Execute T1 actions and record selected actual simulator trajectories."""
    rows = []
    clips = output / "clips"
    clips.mkdir()
    text = None
    stride, count = settings["capture_every_queries"], settings["captures_per_task"]

    def selected(query) -> bool:
        number = query.timestep // 8
        return number % stride == 0 and number // stride < count

    def predict(query, state):
        nonlocal text
        if state is None:
            teacher.reset_live()
        actions, text = teacher.observe_live(query, action=True, plan=selected(query))
        return actions.numpy(), True

    with (output / "chunks.jsonl").open("x", encoding="utf-8") as details:
        def capture(query, actions):
            if not selected(query):
                return None
            chunk_id = f"task-{query.task_id:04d}-t-{query.timestep:04d}"
            clip = f"clips/{chunk_id}.gif"

            def record(trace, frames, executed):
                row = {"chunk_id": chunk_id, "task_id": query.task_id, "timestep": query.timestep,
                       "instruction": query.instruction, "plan": text or "",
                       "plan_status": "valid" if text else "invalid", "clip": clip,
                       "executed_steps": len(executed)}
                _save_clip(output / clip, frames)
                details.write(json.dumps({**row, "normalized_action_chunk": actions.tolist(),
                                          "executed_actions": executed, "states": trace}) + "\n")
                details.flush()
                rows.append(row)

            return record

        result = run_query_rollouts(rollout, predict, stats, max_steps=settings["max_steps"], capture=capture)
    fields = ("chunk_id", "task_id", "timestep", "instruction", "plan", "plan_status",
              "clip", "executed_steps", *CRITERIA, "note")
    write_csv(output / "review.csv", fields, rows)
    return {**result, "captured": len(rows), "invalid_plans": sum(r["plan_status"] == "invalid" for r in rows)}


class StudentRollout:
    """Retain the exact processor, observation normalization, and causal model state."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        from vlade.models.students.backbones.qwen import load_saved_processor
        from vlade.models.students.qwen_vla import QwenVLA

        self.device = device
        self.model = QwenVLA.load(checkpoint / "model").to(device=device, dtype=torch.bfloat16).eval()
        self.processor = load_saved_processor(self.model.student_config, checkpoint / "processor")

    @torch.inference_mode()
    def predict(self, query, state, text: str | None):
        from vlade.models.students.backbones.qwen import prepare_observation

        inputs = prepare_observation(self.processor, self.model.student_config, instruction=query.instruction,
                                     primary=query.primary, wrist=query.wrist, timestep=query.timestep,
                                     teacher_plan=text)
        proprio = torch.as_tensor(query.proprio, dtype=torch.bfloat16, device=self.device)[None]
        result = self.model({name: value.to(self.device) for name, value in inputs.items()}, proprio, state)
        return result.actions[0].float().cpu().numpy(), result.recurrent_state


def compare_trials(results: dict) -> list[dict]:
    """Pair every condition to the same task, seed and initial simulator state."""
    def by_case(report):
        if report["status"] != "complete":
            raise ValueError("Only complete rollout sets can be compared")
        return {(row["task_id"], row["seed"], row["init_index"], row["init_sha256"]): row["success"]
                for row in report["trials"]}

    reference = by_case(results["current_plan"])
    comparisons = []
    for name in CONDITIONS[1:]:
        other = by_case(results[name])
        if other.keys() != reference.keys():
            raise ValueError(f"Rollout pairs differ for {name}")
        wins = sum(reference[key] and not other[key] for key in reference)
        losses = sum(other[key] and not reference[key] for key in reference)
        comparisons.append({"comparison": name, "episodes": len(reference), "current_plan_only": wins,
                            "comparison_only": losses, "ties": len(reference) - wins - losses,
                            "success_delta_pp": 100 * (wins - losses) / len(reference)})
    return comparisons


def _student_condition(name: str, policy, teacher, lag: int, probes: list[dict]):
    history = deque(maxlen=lag)
    counters = dict(queries=0, invalid_plans=0, missing_applied_plans=0,
                    warm_start_queries=0, different_lagged_plans=0)

    def predict(query, state):
        if state is None:
            history.clear()
        counters["queries"] += 1
        if name in ("no_plan", "action_only_kd"):
            return policy.predict(query, state, None)
        _, current = teacher.observe_live(query, action=False, plan=True)
        counters["invalid_plans"] += current is None
        ready = len(history) == lag
        lagged = history[0] if ready else current
        history.append(current)
        counters["warm_start_queries"] += not ready
        counters["different_lagged_plans"] += ready and current is not None and lagged is not None and current != lagged
        applied = lagged if name == "lagged_plan" else current
        counters["missing_applied_plans"] += applied is None
        actions, next_state = policy.predict(query, state, applied)
        if name == "current_plan" and ready and current and lagged and current != lagged:
            alternate, _ = policy.predict(query, state, lagged)
            absent, _ = policy.predict(query, state, None)
            probes.append({"task_id": query.task_id, "timestep": query.timestep, "current_plan": current,
                           "lagged_plan": lagged, "current_vs_lagged_l1": float(np.abs(actions-alternate).mean()),
                           "current_vs_none_l1": float(np.abs(actions-absent).mean())})
        return actions, next_state

    return predict, counters


def evaluate_student(settings: dict, rollout, teacher, policy, load_action_only, stats: dict, output: Path) -> dict:
    probes = []
    results = {}
    lag = settings["lag_queries"]
    for name in CONDITIONS[:3]:
        predict, counters = _student_condition(name, policy, teacher, lag, probes)
        report = run_query_rollouts(rollout, predict, stats, max_steps=settings["max_steps"])
        results[name] = {"model": "S2", **report, **counters}
        if report["status"] != "complete":
            return {"status": "error", "conditions": results, "failed_condition": name}
    # The T1 worker is no longer needed for the Action-only baseline.
    teacher.close()
    baseline = load_action_only()
    predict, counters = _student_condition("action_only_kd", baseline, None, lag, probes)
    report = run_query_rollouts(rollout, predict, stats, max_steps=settings["max_steps"])
    results["action_only_kd"] = {"model": "S1", **report, **counters}
    if report["status"] != "complete":
        return {"status": "error", "conditions": results, "failed_condition": "action_only_kd"}
    fields = ("task_id", "timestep", "current_plan", "lagged_plan", "current_vs_lagged_l1", "current_vs_none_l1")
    write_csv(output / "action_probes.csv", fields, probes)
    comparison = compare_trials(results)
    write_csv(output / "paired_success.csv", tuple(comparison[0]), comparison)
    return {"status": "complete", "conditions": results, "paired_success": comparison, "probes": len(probes),
            "mean_action_l1": {name: mean(row[name] for row in probes) if probes else None for name in fields[-2:]}}


def validate_student_pair(cot: dict, action: dict, cot_step: int, action_step: int, teacher: dict) -> int:
    """Verify common training settings and that the CoT model used this T1."""
    from vlade.distillation.teacher_plan import plan_token_budget

    def without_plan(settings):
        ignored = {"teacher_plan_max_new_tokens", "teacher_python", "teacher_device", "teacher_timeout_seconds"}
        kd = {key: value for key, value in settings["distillation"].items() if key not in ignored}
        return {**settings, "distillation": kd}

    plan_tokens = plan_token_budget(cot["settings"]["distillation"])
    if not plan_tokens or plan_token_budget(action["settings"]["distillation"]):
        raise ValueError("Expected a CoT+Action KD checkpoint and an Action-only KD checkpoint")
    for key in ("data_sha256", "initial_sha256", "git_commit"):
        if not cot.get(key) or cot[key] != action.get(key):
            raise ValueError(f"Student checkpoint identities differ in {key}")
    if cot["teacher_inference"] != teacher or action["teacher_inference"] != teacher:
        raise ValueError("Both Students must use the evaluated T1")
    if cot_step != action_step or without_plan(cot["settings"]) != without_plan(action["settings"]):
        raise ValueError("Compare the same optimizer step and matched training settings")
    return plan_tokens


def run(settings: dict, workspace: Path, teacher_python: Path, output: Path, source: dict) -> dict:
    """Resolve experiment inputs before launching the isolated Teacher."""
    from vlade.data.episode_data import EpisodeData, write_json
    from vlade.runtime import benchmark, environment
    from vlade.storage.assets import resolve_input, sha256
    from vlade.training.run_control import verify_checkpoint
    from vlade.training.teacher_inference import InferenceActions, teacher_identity

    device = torch.device(settings["student_device"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Plan evaluation requires a BF16 CUDA runtime")
    data = EpisodeData(resolve_input(settings["data"]))
    stats = json.loads((data.root / "normalization.json").read_text(encoding="utf-8"))
    _, identity = teacher_identity(settings["teacher"], data)
    mode = settings["mode"]
    checkpoints = {}
    if mode == "student":
        for key in ("cot_checkpoint", "action_checkpoint"):
            path = resolve_input(settings[key])
            seal = verify_checkpoint(path)
            checkpoints[key] = (path, seal, json.loads((path / "identity.json").read_text(encoding="utf-8")))
        _, cot_seal, cot_id = checkpoints["cot_checkpoint"]
        _, action_seal, action_id = checkpoints["action_checkpoint"]
        tokens = validate_student_pair(cot_id, action_id, cot_seal["step"], action_seal["step"], identity)
        if cot_id["data_sha256"] != sha256(data.root / "manifest.json"):
            raise ValueError("Student checkpoints use different normalization data")
    else:
        tokens = settings["plan_tokens"]
    benchmark.prepare(workspace, settings["benchmark_assets"])
    rollout = prepare_rollouts(settings["trials"], settings["seed"])
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    write_json(output / "manifest.json", {
        **source, "settings": settings, "teacher_identity": identity, "plan_tokens": tokens,
        "data_sha256": sha256(data.root / "manifest.json"), "task_ids": list(rollout.task_ids),
        "rollout_runtime": environment.specification(root, "student_eval"),
        "teacher_runtime": environment.specification(root, "ava"),
        "rollout_gpu": torch.cuda.get_device_name(device),
        "teacher_gpu": torch.cuda.get_device_name(settings["teacher_device"]),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "checkpoints": {key: {"model": "S2" if key == "cot_checkpoint" else "S1",
                              "step": seal["step"], "sha256": sha256(path / "checkpoint_manifest.json")}
                        for key, (path, seal, _) in checkpoints.items()},
    })
    teacher_settings = {"teacher_mode": "inference", "teacher_checkpoint": settings["teacher"],
                        "teacher_device": settings["teacher_device"], "teacher_python": str(teacher_python),
                        "teacher_plan_max_new_tokens": tokens,
                        "teacher_timeout_seconds": settings["teacher_timeout_seconds"]}
    with closing(InferenceActions(teacher_settings, data, live=True)) as teacher:
        if mode == "alignment":
            report = evaluate_alignment(settings, rollout, teacher, stats, output)
        else:
            policy = StudentRollout(checkpoints["cot_checkpoint"][0], device)
            def load_baseline():
                policy.model.to("cpu")
                return StudentRollout(checkpoints["action_checkpoint"][0], device)

            report = evaluate_student(settings, rollout, teacher, policy, load_baseline, stats, output)
    write_json(output / "results.json", report)
    if report["status"] != "complete":
        failed = report.get("failed_condition")
        raise RuntimeError(report["conditions"][failed]["reason"] if failed else report["reason"])
    return report
