"""Exercise both evaluation workflows without weights or a GPU."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))

from vlade.distillation.teacher_plan import InvalidTeacherPlan
from vlade.evaluation.plan_observations import encode_query, live_response
from vlade.evaluation.plan_rollout import (
    CONDITIONS, compare_trials, evaluate_alignment, evaluate_student, summarize_alignment, validate_student_pair,
)
from vlade.evaluation.s0_libero_rollout import RolloutPlan, RolloutQuery, SETTLE_STEPS, run_rollouts
from vlade.research import plan_evaluation


def benchmark():
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    class Environment:
        def __init__(self, **kwargs):
            self.steps = 0
            self.position = 0

        def seed(self, seed):
            self.seed_value = seed

        def reset(self):
            self.steps = 0

        def set_init_state(self, _):
            return self.observe()

        def observe(self):
            return {"agentview_image": image + self.position, "robot0_eye_in_hand_image": image,
                    "robot0_eef_pos": [self.position, 0., 0.], "robot0_eef_quat": np.zeros(4),
                    "robot0_gripper_qpos": [0., 0.], "cup_pos": [100., 0., 0.]}

        def step(self, action):
            self.steps += 1
            self.position += int(action[0] > 0.5)
            return self.observe(), 0, False, {}

        def check_success(self):
            return self.steps >= SETTLE_STEPS + 17 and self.position >= 17

        def close(self):
            pass

    suite = SimpleNamespace(get_task_init_states=lambda task: [np.array([task])],
                            get_task_bddl_file_path=lambda task: "example",
                            get_task=lambda task: SimpleNamespace(language="Pick up the cup"))
    return RolloutPlan(suite, Environment, lambda quat: np.zeros(3), (4, 9), 42)


def stats():
    def identity(n):
        return dict(mask=[False] * n, q01=[-1.] * n, q99=[1.] * n)
    return {"action": identity(7), "proprio": identity(8)}


def test_live_wire_preserves_frames_and_proprio_and_handles_invalid_plan():
    query = RolloutQuery(5, 8, "Pick up cup", Image.new("RGB", (256, 256), "red"),
                         Image.new("RGB", (256, 256), "blue"), np.arange(8) / 8.)
    saw = []

    def predict(row, data):
        assert row["timestep"] == 8
        assert data.image(row, "primary").getpixel((0, 0)) == (255, 0, 0)
        assert data.image(row, "wrist").getpixel((0, 0)) == (0, 0, 255)
        np.testing.assert_equal(row["proprio"], query.proprio)
        return torch.ones((8, 7)), None

    def invalid(*a, **kw):
        raise InvalidTeacherPlan("empty")

    model = SimpleNamespace(predict=predict, plan=invalid, reset=lambda: saw.append("reset"))
    assert live_response(model, {"live": "reset"}, 48) == {"live": "reset"}
    assert saw == ["reset"]
    response = live_response(model, {"live": "query", **encode_query(query), "action": True, "plan": True}, 48)
    assert response == {"task_id": 5, "timestep": 8, "action": [[1.] * 7] * 8, "plan": None}


def test_alignment_records_executed_frames_and_manual_scores(tmp_path):
    resets = []
    teacher = SimpleNamespace(reset_live=lambda: resets.append(1),
                              observe_live=lambda query, **kw: (torch.ones((8, 7)), "Move toward the cup"))
    result = evaluate_alignment({"capture_every_queries": 1, "captures_per_task": 3, "max_steps": 24},
                                benchmark(), teacher, stats(), tmp_path)
    assert (result["successes"], result["captured"], resets) == (2, 6, [1, 1])
    rows = [json.loads(line) for line in (tmp_path / "chunks.jsonl").read_text().splitlines()]
    assert [row["executed_steps"] for row in rows] == [8, 8, 1] * 2
    assert rows[0]["states"][0]["eef_xyz"] == [0., 0., 0.]
    assert rows[0]["states"][-1]["eef_xyz"] == [8., 0., 0.]
    with Image.open(tmp_path / rows[0]["clip"]) as clip:
        assert clip.n_frames == 9
    score = summarize_alignment(tmp_path / "review.csv")
    assert score["criteria"]["target_motion"]["unreviewed"] == 6
    source = tmp_path / "review.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        writer_fields = tuple(csv.DictReader(stream).fieldnames)
    with source.open(encoding="utf-8", newline="") as stream:
        labels = list(csv.DictReader(stream))
    labels[0]["target_motion"] = "match"
    labels[1]["target_motion"] = "mismatch"
    edited = tmp_path / "review-labeled.csv"
    with edited.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=writer_fields)
        writer.writeheader()
        writer.writerows(labels)
    assert summarize_alignment(edited)["criteria"]["target_motion"]["agreement"] == 0.5
    assert run_rollouts(benchmark(), lambda *args: (np.ones((8, 7)), None), stats())["successes"] == 2


def test_student_comparison_fixes_state_for_intervention_and_pairs_initial_states(tmp_path):
    closes = []
    teacher = SimpleNamespace(close=lambda: closes.append(True),
                              observe_live=lambda query, **kw: (None, f"Approach at {query.timestep}."))
    same_state = {}

    def predict(q, state, text):
        start = (q.task_id, q.timestep)
        same_state.setdefault(start, []).append(state)
        action = float(text == f"Approach at {q.timestep}.")
        return np.full((8, 7), action), q.timestep

    policy = SimpleNamespace(predict=predict)
    baseline = SimpleNamespace(predict=lambda q, state, text: (np.zeros((8, 7)), q.timestep))
    result = evaluate_student({"lag_queries": 2, "max_steps": 24}, benchmark(), teacher, policy,
                              lambda: baseline, stats(), tmp_path)
    assert result["status"] == "complete"
    assert set(result["conditions"]) == set(CONDITIONS)
    assert result["conditions"]["current_plan"]["successes"] == 2
    assert result["conditions"]["lagged_plan"]["successes"] == 0
    assert result["conditions"]["current_plan"]["model"] == "S2"
    assert result["conditions"]["action_only_kd"]["model"] == "S1"
    assert all(row["success_delta_pp"] == 100 for row in result["paired_success"])
    assert result["probes"] == 2 and same_state[(4, 16)] == [8] * 5
    assert (tmp_path / "paired_success.csv").exists() and closes == [True]
    result["conditions"]["lagged_plan"]["trials"][0]["init_sha256"] = "wrong"
    with pytest.raises(ValueError, match="pairs differ"):
        compare_trials(result["conditions"])


def test_checkpoint_comparison_requires_same_training_except_plan():
    base = {"git_commit": "source", "data_sha256": "data", "initial_sha256": "S0",
            "teacher_inference": {"step": 1975}}
    def identity(tokens, lr=0.1):
        return {**base, "settings": {"learning_rate": lr, "distillation": {
            "teacher_mode": "inference", "teacher_plan_max_new_tokens": tokens}}}
    assert validate_student_pair(identity(48), identity(0), 100, 100, base["teacher_inference"]) == 48
    with pytest.raises(ValueError, match="matched training"):
        validate_student_pair(identity(48), identity(0, 0.2), 100, 100, base["teacher_inference"])


@pytest.mark.parametrize("name", ["t1_plan_alignment", "s1_s2_plan_effect"])
def test_notebook_calls_plan_with_its_config_and_does_not_create_workspace(name, tmp_path):
    path = ROOT / "experiments" / f"colab_{name}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    scope = {"CODE": ROOT, "Path": Path}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    exec(cells["config"], scope)
    scope["WORKSPACE"] = str(tmp_path / "new")
    exec(cells["run"], scope)
    assert not Path(scope["WORKSPACE"]).exists()
    assert scope["out"].name == scope["RUN_NAME"]


def test_plan_pilot_bounds_and_bad_output_path(tmp_path):
    config = ROOT / "configs/experiments/s1_s2_plan_effect.json"
    assert plan_evaluation.load_settings(config, {"trials": 1, "max_steps": 32})["trials"] == 1
    with pytest.raises(ValueError, match="Unknown evaluation settings"):
        plan_evaluation.load_settings(config, {"lag": 0})
    with pytest.raises(ValueError, match="run name"):
        plan_evaluation.output_path(tmp_path, "../other")
