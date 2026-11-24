"""Bounded real-runtime probe routing and its failure/success records."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))

from vlade.research import distillation
from vlade.training import teacher_inference


def prepare_probe(tmp_path, monkeypatch, *, invalid=False):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    calls = []

    def episodes(split):
        yield {}, [{"episode_id": split, "timestep": t, "instruction": "Move cup"} for t in (0, 8, 16)]

    data = SimpleNamespace(root=tmp_path, episodes=episodes)
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda reference: tmp_path)
    monkeypatch.setattr("vlade.data.episode_data.EpisodeData", lambda path: data)

    class Teacher:
        signature, plan_tokens = {"step": 5}, 48

        def __init__(self, settings, actual):
            assert actual is data

        def get_signal(self, sample):
            calls.append(("train", sample["timestep"]))
            if invalid:
                raise RuntimeError("InvalidTeacherPlan; token_ids=[2]")
            return teacher_inference.TeacherSignal(torch.zeros(8, 7), "Move cup.")

        def get_plan(self, sample, *, split):
            calls.append((split, sample["timestep"]))
            return "Approach cup."

        def close(self):
            calls.append("close")

    monkeypatch.setattr(teacher_inference, "InferenceActions", Teacher)
    return calls


def test_probe_limits_queries_and_records_identities_and_plans(tmp_path, monkeypatch):
    calls = prepare_probe(tmp_path, monkeypatch)
    settings = {"teacher_mode": "inference", "teacher_plan_max_new_tokens": 48}
    report = teacher_inference.probe_teacher_plans(settings, "hfasset://data", tmp_path / "probe")
    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["teacher"] == {"step": 5} and len(rows[0]["data_manifest_sha256"]) == 64
    assert [row["plan"] for row in rows[1:-1]] == ["Move cup."]*2 + ["Approach cup."]*2
    assert rows[-1] == {"status": "complete", "queries": 4}
    assert calls == [("train", 0), ("train", 8), ("validation", 0), ("validation", 8), "close"]


def test_probe_failure_keeps_query_details_and_does_not_mark_complete(tmp_path, monkeypatch):
    calls = prepare_probe(tmp_path, monkeypatch, invalid=True)
    settings = {"teacher_mode": "inference", "teacher_plan_max_new_tokens": 48}
    with pytest.raises(RuntimeError, match="token_ids"):
        teacher_inference.probe_teacher_plans(settings, "hfasset://data", tmp_path / "probe")
    report, = (tmp_path / "probe").glob("*.jsonl")
    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["status"] == "plan_error" and rows[1]["timestep"] == 0
    assert rows[-1]["status"] == "failed" and all(row["status"] != "complete" for row in rows)
    assert calls[-1] == "close"


def test_probe_routes_to_teacher_without_training_or_hub(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(teacher_inference, "probe_teacher_plans", lambda *args: calls.append(args))
    monkeypatch.setattr(distillation.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected training/Hub call"))
    _, variants = distillation.load_settings(ROOT / "configs/experiments/qwen_vla_plan_kd.json", {})
    distillation._run_variant("gt_kd", variants["gt_kd"], tmp_path, "probe-test", "probe", Path(sys.executable), None)
    settings, data, output = calls[0]
    assert settings["teacher_python"] == sys.executable and data == "hfasset://prepared_d_seed42"
    assert output == tmp_path
    with pytest.raises(ValueError, match="online CoT"):
        distillation._run_variant("gt_only", {}, tmp_path, "probe-test", "probe", None, None)


def test_non_cot_probe_is_rejected_before_creating_a_workspace(tmp_path):
    work = tmp_path / "not-created"
    with pytest.raises(SystemExit) as caught:
        distillation.main(["--config", str(ROOT / "configs/experiments/qwen_vla_action_kd.json"),
                           "--workspace", str(work), "--run-group", "reject-action-only", "--action", "probe"])
    assert caught.value.code == 2 and not work.exists()
