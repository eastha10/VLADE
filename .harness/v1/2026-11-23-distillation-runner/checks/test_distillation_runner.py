"""CPU regressions for the config-only clients and native validation history."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from vlade.models.contracts import PolicyOutput
from vlade.research import distillation
from vlade.training import qwen_vla_task

ACTION = ROOT / "configs/experiments/qwen_vla_action_kd.json"
PLAN = ROOT / "configs/experiments/qwen_vla_plan_kd.json"


@pytest.mark.parametrize("name,config,variant", [
    ("colab_action_distillation", ACTION.name, "gt_kd"),
    ("colab_teacher_plan_kd", PLAN.name, "gt_kd"),
    ("colab_qwen_vla_unfrozen", ACTION.name, "gt_only"),
])
def test_notebook_delegates_to_config_only(name, config, variant, monkeypatch):
    notebook = json.loads((ROOT / "experiments" / f"{name}.ipynb").read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"] if cell["cell_type"] == "code"}
    assert list(cells) == ["source", "config", "run"]
    source = {name: "".join(cell["source"]) for name, cell in cells.items()}
    for name, code in source.items():
        compile(code, name, "exec")
        assert cells[name]["outputs"] == []
    assert "refs/heads/main" in source["source"]
    calls = []
    monkeypatch.setattr(distillation, "job", lambda config, **kwargs: calls.append((config, kwargs)))
    scope = {"CODE": ROOT}
    exec(source["config"], scope)
    exec(source["run"], scope)
    group = scope["RUN_GROUP"]
    exec(source["config"], scope)
    assert scope["RUN_GROUP"] == group
    assert calls[0][0].name == config
    assert calls[0][1]["action"] == "plan"
    _, variants = distillation.load_settings(calls[0][0], calls[0][1]["overrides"])
    assert list(variants) == [variant]


def test_plan_and_variants_are_read_only(tmp_path):
    work = tmp_path / "never-created"
    process = subprocess.run([sys.executable, "-B", str(distillation.CLI), "--config", str(PLAN),
                              "--workspace", str(work), "--run-group", "plan-local"], capture_output=True,
                             text=True, check=True, cwd=ROOT)
    planned = json.loads(process.stdout)
    assert planned["variants"]["gt_kd"]["distillation"]["teacher_plan_max_new_tokens"] == 48
    assert planned["hub_runs"] == {"S2": "runs/plan-local/S2"}
    assert not work.exists()
    _, origins = distillation.load_settings(ACTION, {"workflow.variant": "both"})
    assert origins["gt_only"] == {key: value for key, value in origins["gt_kd"].items() if key != "distillation"}
    assert origins["gt_kd"]["distillation"]["teacher_plan_max_new_tokens"] == 0
    with pytest.raises(ValueError, match="input prompts"):
        distillation.load_settings(PLAN, {"workflow.variant": "both"})
    with pytest.raises(ValueError, match="Unknown config key"):
        distillation.load_settings(ACTION, {"distillation.bad_field": 1})


def test_worker_routes_pilot_resume_final_and_retry(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(distillation, "source_identity", lambda *a, **kw: {"git_commit": "b" * 40})
    monkeypatch.setattr(distillation, "locate", lambda *a: Path("teacher-python"))
    monkeypatch.setattr(distillation, "prepare_inputs", lambda *a: tmp_path / "runtime-inputs.json")
    monkeypatch.setenv("VLADE_ASSET_REGISTRY", str(tmp_path / "registry.json"))

    def run(command, *, env, **options):
        calls.append((command, env["VLADE_HUB_RUN_ID"]))
        assert env["VLADE_HUB_REPO"] == distillation.REGISTRY_REPO
        if "--stop-after" in command or "download" in command:
            step = int(command[-1]) if "--stop-after" in command else 4
            output = Path(command[command.index("--output") + 1])
            (output / f"step-{step:07d}").mkdir(parents=True)

    monkeypatch.setattr(distillation.subprocess, "run", run)
    base = ["--config", str(ACTION), "--workspace", str(tmp_path), "--worker"]

    def invoke(action, group="synthetic", *extra):
        distillation.main([*base, "--run-group", group, "--action", action, *extra])

    with pytest.raises(RuntimeError, match="pilot"):
        invoke("train")
    invoke("preflight")
    assert calls[-1][0][-1] == "--preflight"
    invoke("pilot")
    pilot, hub_run = calls[-1]
    assert hub_run == "synthetic"
    assert Path(pilot[pilot.index("--output") + 1]).name == "S1"
    assert pilot[-2:] == ["--stop-after", "1"]
    assert "--resume" not in pilot and "--teacher-python" in pilot
    invoke("train")
    train = calls[-1][0]
    assert train[-2:] == ["--stop-after", "100"]
    assert Path(train[train.index("--resume") + 1]).name == "step-0000001"
    invoke("train")
    assert calls[-1][0][-1] == "--preflight"
    with pytest.raises(ValueError, match="run group"):
        invoke("pilot", "synthetic", "--overrides", '{"distillation.action_kd_weight": 0.5}')
    invoke("train", "restore", "--resume-revisions", json.dumps({"S1": "a" * 40}))
    download, train = (row[0] for row in calls[-2:])
    assert download[download.index("--run") + 1] == "restore/S1"
    assert Path(train[train.index("--resume") + 1]).name == "step-0000004"
    invoke("retry", "restore")
    assert "retry" in calls[-1][0]


@pytest.mark.parametrize("model_name", ["S1", "S2"])
def test_named_final_hub_upload_returns_evaluation_pin(model_name, tmp_path, monkeypatch, capsys):
    from vlade.training.hub import HubCheckpoints
    from vlade.training.run_control import seal_checkpoint

    sent = {}

    def create_commit(*, operations: list, **kwargs):
        for operation in operations:
            sent[operation.path_in_repo] = operation.path_or_fileobj
        return SimpleNamespace(oid="1" * 40)

    api = SimpleNamespace(create_repo=lambda **kw: None, model_info=lambda *a: SimpleNamespace(private=True),
                          create_commit=create_commit)
    monkeypatch.setenv("VLADE_HUB_REPO", distillation.REGISTRY_REPO)
    monkeypatch.setenv("VLADE_HUB_RUN_ID", "paired")
    hub = HubCheckpoints(tmp_path / model_name, api=api)
    folder = hub.output / "step-0000001"
    folder.mkdir(parents=True)
    (folder / "trainer_state.pt").write_bytes(b"optimizer")
    hub.prepare(folder, 1, {})
    seal_checkpoint(folder, 1, 1)
    (folder / "COMPLETE").write_text("ok\n", encoding="utf-8")
    hub.publish(folder, final=True)
    root = f"runs/paired/{model_name}"
    remote = f"{root}/checkpoints/step-0000001"
    assert json.loads(sent[root + "/final.json"].getvalue())["checkpoint"] == remote
    assert f"{remote}/trainer_state.pt" in sent
    line = capsys.readouterr().out.strip()
    assert json.loads(line.removeprefix("CHECKPOINT_PINS ")) == {
        model_name: {"repo_id": distillation.REGISTRY_REPO, "revision": "1" * 40, "prefix": remote}}


def test_sparse_validation_replays_missing_queries_in_episode_order(monkeypatch):
    """A query should see the same recurrence regardless of the validation stride."""
    def query(episode, step):
        return {"episode_id": episode, "timestep": step, "proprio": np.zeros(8, dtype=np.float32),
                "gt_action": np.zeros((8, 7), dtype=np.float32), "action_valid_mask": np.ones((8, 7), dtype=bool)}

    episodes = {episode: [query(episode, step) for step in (0, 8, 16, 24, 32)] for episode in ("a", "b")}
    data = SimpleNamespace(episodes=lambda split, order: iter([({}, episodes[order[0]])]))
    history = []

    def observation(data, processor, config, sample, plan=None):
        return {"timestep": torch.tensor(sample["timestep"])}

    def forward(inputs, proprio, state, *, batch_idx):
        value = state.item() + 1 if state is not None else 1
        history.append((inputs["timestep"].item(), value))
        output = PolicyOutput(torch.zeros(1, 8, 7), torch.zeros(1, 8, 7))
        output.recurrent_state = torch.tensor(value)
        return output

    plans = []

    def get_plan(row, *, split):
        assert split == "validation"
        plans.append((row["episode_id"], row["timestep"]))
        return "Approach the object."

    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(qwen_vla_task, "observation_inputs", observation)
    loss, _, _ = qwen_vla_task.action_loss(forward, data, None, None, torch.device("cpu"),
                                          phase="validation", plan_teacher=SimpleNamespace(get_plan=get_plan))
    for episode, index in [("a", 0), ("a", 3), ("a", 4), ("b", 2)]:
        queries = episodes[episode]
        loss((queries[index], None, queries[index - 1] if index else None), 0)
    assert history == [(0, 1), (8, 2), (16, 3), (24, 4), (32, 5), (0, 1), (8, 2), (16, 3)]
    assert sorted(plans) == [("a", step) for step in (0, 8, 16, 24, 32)] + [("b", s) for s in (0, 8, 16)]


def test_native_preflight_verifies_teacher_and_student_without_cuda(tmp_path, monkeypatch, capsys):
    _, variants = distillation.load_settings(PLAN, {})
    settings = variants["gt_kd"]
    settings["distillation"]["teacher_python"] = sys.executable
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(qwen_vla_task, "source_identity", lambda *a, **k: {"git_commit": "a" * 40})
    monkeypatch.setattr(qwen_vla_task, "resolve_input", lambda value: tmp_path)
    monkeypatch.setattr(qwen_vla_task, "EpisodeData", lambda path: SimpleNamespace(root=path, manifest={"counts": {}}))
    validated = []

    def verify_student(path, config, fingerprint):
        validated.append(("s0", path))
        return 800

    def verify_teacher(checkpoint, data):
        validated.append(("t1", checkpoint))
        return tmp_path, {"step": 42}

    monkeypatch.setattr(qwen_vla_task, "verified_s0", verify_student)
    monkeypatch.setattr(qwen_vla_task, "teacher_identity", verify_teacher)
    output = tmp_path / "not-created"
    qwen_vla_task.run(settings, output, preflight=True)
    assert validated == [("s0", tmp_path), ("t1", "hfasset://T1")]
    assert json.loads(capsys.readouterr().out)["initial_student_step"] == 800
    assert not output.exists()
