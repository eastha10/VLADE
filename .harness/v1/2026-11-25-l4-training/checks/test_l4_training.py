"""L4 controls and native tiny-model regressions; no downloads or official runs."""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import copy
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".harness/v1/2026-11-11-ava-teacher-code/checks")]
from qwen_vla_smoke import tiny_inputs, tiny_student
from vlade.distillation.action import action_distillation_loss
from vlade.models.contracts import masked_action_l1
from vlade.models.students.qwen_vla import QwenVLA
from vlade.research import distillation
from vlade.training import qwen_vla_task, teacher_inference
from vlade.training.loop import (LoopState, capture_rng, restore_rng, make_scheduler,
                                 restore_training_state, training_state)
from vlade.training.memory import adamw_storage, configure_qwen_memory, validate_memory_profile
from vlade.training.residency import StudentOffload, teacher_on_device
from vlade.training.run_control import seal_checkpoint, verify_checkpoint
from vlade.distillation.teacher_plan import InvalidTeacherPlan, PLAN_PROMPT_VERSION

CONFIGS = ROOT / "configs/experiments"


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(42)


def config(name="qwen_vla_s0_l4"):
    return json.loads((CONFIGS / (name + ".json")).read_text())


@pytest.mark.parametrize("l4,original", [
    ("qwen_vla_s0_l4", "qwen_vla_s0"),
    ("qwen_vla_action_kd_l4", "qwen_vla_action_kd"),
    ("qwen_vla_plan_kd_l4", "qwen_vla_plan_kd"),
])
def test_preserves_research_settings(l4, original):
    profile, before = config(l4), config(original)
    assert validate_memory_profile(profile)["attention_backend"] == "sdpa"
    for key in ("model_config", "initial_checkpoint", "data", "seed", "max_steps", "accumulation_steps",
                "learning_rate", "weight_decay", "save_steps", "validation_queries"):
        assert profile[key] == before[key]
    assert profile["trainable_scope"] == "full"
    if "distillation" in before:
        assert {k: v for k, v in profile["distillation"].items() if k != "teacher_residency"} == before["distillation"]
        assert profile["workflow"] == before["workflow"]


@pytest.mark.parametrize("field,value", [("effective_batch_size", 4), ("microbatch_size", 2),
    ("parameter_dtype", "bfloat16"), ("attention_backend", "flash_attention_2"),
    ("reserve_gib", float("nan")), ("reserve_gib", -1), ("gradient_checkpointing", False)])
def test_invalid_memory_controls_rejected(field, value):
    settings = config()
    settings["memory_profile"][field] = value
    with pytest.raises(ValueError):
        validate_memory_profile(settings)


def test_unsafe_online_residency_rejected_but_valid_cache_needs_no_teacher():
    settings = config("qwen_vla_action_kd_l4")
    settings["distillation"]["teacher_residency"] = "shared"
    with pytest.raises(ValueError, match="exclusive"):
        validate_memory_profile(settings)
    _, variants = distillation.load_settings(CONFIGS / "qwen_vla_action_kd_l4_cache.json", {})
    kd = variants["gt_kd"]["distillation"]
    assert kd["teacher_cache"] == "hfasset://teacher_cache_t1"
    assert kd["teacher_mode"] == "cache" and "teacher_checkpoint" not in kd


@pytest.mark.parametrize("name,label", [("qwen_vla_s0_l4", "S0"), ("qwen_vla_action_kd_l4", "S1"),
    ("qwen_vla_action_kd_l4_cache", "S1"), ("qwen_vla_plan_kd_l4", "S2")])
def test_plans_are_read_only_and_keep_hub_roles(name, label, tmp_path):
    work = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, "-B", str(distillation.CLI), "--config", str(CONFIGS / (name + ".json")),
                             "--workspace", str(work), "--run-group", "l4-check"],
                            capture_output=True, text=True, check=True, cwd=ROOT)
    plan = json.loads(result.stdout)
    assert plan["hub_runs"] == {label: f"runs/l4-check/{label}"}
    assert plan["hub_repo"] == "VLADE-konkuk/VLADE-checkpoints"
    assert not work.exists()


def test_native_sdpa_checkpointing_keeps_ava_outputs_and_gradients():
    eager = tiny_student().train()
    optimized = copy.deepcopy(eager).train()
    details = configure_qwen_memory(optimized, config())
    assert details["effective_batch_size"] == 8
    assert optimized.language_model.config._attn_implementation == "sdpa"
    assert optimized.vision_backbone.config._attn_implementation == "sdpa"
    assert all(layer.gradient_checkpointing for layer in optimized.language_model.layers)
    assert optimized.backbone.is_gradient_checkpointing
    calls = []
    handle = optimized.language_model.layers[3].register_forward_pre_hook(lambda *a: calls.append(1))
    inputs, proprio = tiny_inputs(), torch.randn(1, 8)
    target, valid = torch.randn(1, 8, 7), torch.ones(1, 8, 7, dtype=torch.bool)
    teacher = torch.randn(1, 8, 7)
    rng = capture_rng()
    a = eager(inputs, proprio)
    restore_rng(rng)
    b = optimized(inputs, proprio)
    torch.testing.assert_close(a.actions, b.actions, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(a.visual_weights, b.visual_weights)
    for model, output in ((eager, a), (optimized, b)):
        loss, _ = action_distillation_loss(output, target, teacher, valid, task_weight=1., action_weight=.7)
        loss.backward()
    assert len(calls) >= 2  # recomputation occurs in the actual native decoder
    handle.remove()
    for (name, p), (_, q) in zip(eager.named_parameters(), optimized.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=2e-3, msg=name)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in optimized.ava.parameters())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable; L4 residency requires hardware verification"))])
def test_offload_preserves_accumulated_gradients_optimizer_and_parameter_objects(device):
    baseline = torch.nn.Linear(3, 2).to(device)
    model = copy.deepcopy(baseline)
    a = torch.optim.AdamW(baseline.parameters(), lr=.01, foreach=False)
    b = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
    identities = [id(p) for p in model.parameters()]
    batches = [torch.randn(1, 3, device=device) for _ in range(8)]
    offload = StudentOffload(model, b, device)
    for _ in range(2):  # second pass has initialized moments as well as live gradients
        a.zero_grad(set_to_none=True)
        b.zero_grad(set_to_none=True)
        for batch in batches:
            with offload():
                assert all(p.device.type == "cpu" for p in model.parameters())
                assert all(p.grad is None or p.grad.device.type == "cpu" for p in model.parameters())
            (baseline(batch).square().mean() / 8).backward()
            (model(batch).square().mean() / 8).backward()
        a.step()
        b.step()
    assert [id(p) for p in model.parameters()] == identities
    for p, q in zip(baseline.parameters(), model.parameters()):
        torch.testing.assert_close(p, q)
        torch.testing.assert_close(a.state[p]["exp_avg"], b.state[q]["exp_avg"])
    with pytest.raises(RuntimeError, match="request failed"):
        with offload():
            raise RuntimeError("request failed")
    assert all(str(p.device) == device for p in model.parameters())


def test_teacher_offload_keeps_recurrence_and_cleans_up_before_response(monkeypatch):
    moves = []
    class Module:
        def __init__(self, name): self.name = name
        def to(self, device): moves.append((self.name, str(device)))
    policy = SimpleNamespace(model=Module("model"), head=Module("head"), proprio=Module("proprio"),
                             generator=Module("ava"), context=torch.ones(1), device=torch.device("cpu"))
    # CPU exercises restoration even when inference fails; real CUDA test above is conditional.
    with pytest.raises(ValueError):
        with teacher_on_device(policy, "cpu", exclusive=True):
            policy.context += 2
            raise ValueError("failed")
    assert moves == [(name, "cpu") for name in ("model", "head", "proprio", "ava")] * 2
    assert policy.context.item() == 3 and policy.device.type == "cpu"


def test_timeout_kills_worker_before_student_restore():
    events = []
    teacher = teacher_inference.InferenceActions.__new__(teacher_inference.InferenceActions)
    @contextmanager
    def offload():
        events.append("student_cpu")
        try: yield
        finally: events.append("student_cuda")
    teacher.request_context = offload
    teacher.process = SimpleNamespace(stdin=io.StringIO())
    teacher._receive = lambda: (_ for _ in ()).throw(TimeoutError("timeout"))
    teacher.close = lambda: events.append("teacher_stopped")
    with pytest.raises(TimeoutError):
        teacher._send({"episode_id": "a", "timestep": 0})
    assert events == ["student_cpu", "teacher_stopped", "student_cuda"]


def test_action_target_is_fetched_before_student_graph(monkeypatch):
    events = []
    monkeypatch.setattr(torch, "autocast", lambda *args, **kw: nullcontext())
    model = tiny_student()
    model.register_forward_pre_hook(lambda *args: events.append("student"))
    teacher = SimpleNamespace(get=lambda sample: (events.append("teacher") or torch.zeros(8, 7)))
    sample = {"episode_id": "a", "timestep": 0, "proprio": np.zeros(8, dtype=np.float32),
              "gt_action": np.ones((8, 7), dtype=np.float32), "action_valid_mask": np.ones((8, 7), dtype=bool)}
    loss, _, _ = qwen_vla_task.action_loss(model, None, None, None, "cpu", teacher=teacher)
    loss((sample, tiny_inputs(), None), 0).backward()
    assert events == ["teacher", "student"]


def test_storage_estimate_and_activation_reserve(monkeypatch):
    model = torch.nn.Linear(4, 2)
    storage = adamw_storage(model)
    assert storage["minimum_training_bytes"] == 10 * 16
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1024**3, 24 * 1024**3))
    with pytest.raises(RuntimeError, match="activation reserve"):
        qwen_vla_task.check_adamw_capacity(model, "cuda:0", reserve_gib=2)
    model.weight.requires_grad_(False)
    assert adamw_storage(model)["minimum_training_bytes"] == 8 * 4 + 2 * 16


def test_tiny_native_optimizer_seal_reload_and_resume(tmp_path):
    model = tiny_student().train()
    configure_qwen_memory(model, config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, foreach=False)
    scheduler = make_scheduler(optimizer, 2)
    inputs, proprio = tiny_inputs(), torch.randn(1, 8)
    target, valid = torch.randn(1, 8, 7), torch.ones(1, 8, 7, dtype=torch.bool)
    for _ in range(8):
        (masked_action_l1(model(inputs, proprio), target, valid) / 8).backward()
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    checkpoint = tmp_path / "step-0000001"
    model.save(checkpoint / "model")
    payload = training_state(LoopState(1, 8), optimizer, scheduler)
    payload["episode_history"] = ("a", torch.zeros(1, 56, 32))
    torch.save(payload, checkpoint / "trainer_state.pt")
    seal_checkpoint(checkpoint, 1, 8)
    (checkpoint / "COMPLETE").write_text("ok\n")
    assert verify_checkpoint(checkpoint, accumulation=8)["step"] == 1
    resumed = QwenVLA.load(checkpoint / "model").train()
    configure_qwen_memory(resumed, config())
    other = torch.optim.AdamW(resumed.parameters(), lr=2e-5, foreach=False)
    schedule = make_scheduler(other, 2)
    state = restore_training_state(torch.load(checkpoint / "trainer_state.pt", weights_only=True), other, schedule)
    assert (state.step, state.microsteps) == (1, 8)
    rng = capture_rng()
    for student, opt, sched in ((model, optimizer, scheduler), (resumed, other, schedule)):
        restore_rng(rng)
        for _ in range(8):
            (masked_action_l1(student(inputs, proprio), target, valid) / 8).backward()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    for p, q in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(p, q)


def test_l4_notebook_is_config_only_and_defaults_to_plan():
    notebook = json.loads((ROOT / "experiments/colab_l4_training.ipynb").read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    assert list(cells) == ["source", "config", "run"]
    for code in cells.values(): compile(code, "notebook", "exec")
    scope = {}
    exec(cells["config"], scope)
    assert scope["ACTION"] == "plan" and scope["PROFILE"] == "s0"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"


def test_actual_native_trainer_cpu_pilot_resume_and_identity_rejection(tmp_path, monkeypatch):
    """Real trainer/loss/backward/optimizer/serialization, with synthetic I/O and CPU placement."""
    from vlade.training import memory
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "download_manifest.json").write_text("{}")
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "manifest.json").write_text("synthetic-data")
    data = SimpleNamespace(root=data_root, manifest={"counts": {}, "normalization_sha256": "n" * 64})
    model_config = tmp_path / "model.json"
    model_config.write_text(json.dumps({"student": asdict(tiny_student().student_config)}))
    settings = {**config(), "model_config": str(model_config), "max_steps": 2, "save_steps": 1,
                "validation_queries": 1}
    settings.pop("workflow")
    monkeypatch.setattr(qwen_vla_task, "resolve_input", lambda ref: data_root if ref == settings["data"] else initial)
    monkeypatch.setattr(qwen_vla_task, "EpisodeData", lambda root: data)
    monkeypatch.setattr(qwen_vla_task, "source_identity", lambda *a, **kw: {"git_commit": "a" * 40})
    monkeypatch.setattr(QwenVLA, "from_pretrained", lambda *args, **kw: tiny_student())
    class Processor:
        def save_pretrained(self, folder):
            folder.mkdir()
            (folder / "processor.json").write_text("synthetic-processor")
    monkeypatch.setattr(qwen_vla_task, "load_processor", lambda *a, **kw: Processor())
    monkeypatch.setattr(qwen_vla_task, "observation_inputs", lambda *a: tiny_inputs())
    def batches(data, processor, cfg, split, seed, *, repeat=False, skip=0, limit=None, **kw):
        for index in range(skip, 16 if repeat else limit):
            sample = {"episode_id": "episode", "timestep": index * 8,
                      "proprio": np.zeros(8, dtype=np.float32), "gt_action": np.ones((8, 7), dtype=np.float32),
                      "action_valid_mask": np.ones((8, 7), dtype=bool)}
            yield sample, tiny_inputs(), None
    monkeypatch.setattr(qwen_vla_task, "batches", batches)
    # No GPU or HF bypass in operating code: only this test module substitutes placement/I/O.
    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
        mem_get_info=lambda device: (24 * 1024**3, 24 * 1024**3), manual_seed_all=lambda seed: None,
        get_device_name=lambda *a: "synthetic CPU placement", reset_peak_memory_stats=lambda: None,
        OutOfMemoryError=torch.cuda.OutOfMemoryError)
    proxy = SimpleNamespace(**{key: getattr(torch, key) for key in dir(torch)})
    proxy.cuda = cuda
    proxy.device = lambda *a: torch.device("cpu")
    proxy.autocast = lambda *a, **kw: nullcontext()
    monkeypatch.setattr(qwen_vla_task, "torch", proxy)
    measured = lambda device: dict(allocated_bytes=0, reserved_bytes=0, peak_allocated_bytes=0,
                                   peak_reserved_bytes=0, device_free_bytes=0, device_total_bytes=0, device_used_bytes=0)
    monkeypatch.setattr(memory, "cuda_memory", measured)
    monkeypatch.setattr(qwen_vla_task, "cuda_memory", measured)
    publications = []
    class Hub:
        def __init__(self, *args, **kw): pass
        def prepare(self, *args): pass
        def publish(self, checkpoint, *, final): publications.append((checkpoint.name, final))
        def flush(self): pass
    monkeypatch.setattr(qwen_vla_task, "HubCheckpoints", Hub)
    output = tmp_path / "S0"
    qwen_vla_task.run(settings, output, stop_after=1)
    first = output / "step-0000001"
    assert verify_checkpoint(first, accumulation=8)["microsteps"] == 8
    assert publications == [(first.name, False)]
    qwen_vla_task.run(settings, output, resume=first)
    assert verify_checkpoint(output / "step-0000002", accumulation=8)["microsteps"] == 16
    assert publications[-1] == ("step-0000002", True)
    changed = copy.deepcopy(settings)
    changed["memory_profile"]["attention_backend"] = "eager"
    with pytest.raises(ValueError, match="identity differs"):
        qwen_vla_task.run(changed, output, resume=first)


@pytest.mark.parametrize("supervised", [True, False])
def test_l4_s2_preflight_preserves_plan_supervision_without_loading_models(supervised, tmp_path, monkeypatch, capsys):
    _, variants = distillation.load_settings(CONFIGS / "qwen_vla_plan_kd_l4.json", {})
    settings = variants["gt_kd"]
    settings["distillation"]["teacher_python"] = sys.executable
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "checkpoint_manifest.json").write_text("{}")
    monkeypatch.setattr(qwen_vla_task, "source_identity", lambda *a, **kw: {"git_commit": "a" * 40})
    monkeypatch.setattr(qwen_vla_task, "resolve_input", lambda ref: tmp_path)
    monkeypatch.setattr(qwen_vla_task, "EpisodeData", lambda path: SimpleNamespace(root=path, manifest={"counts": {}}))
    monkeypatch.setattr(qwen_vla_task, "verified_s0", lambda *a: 800)
    signature = {"step": 42}
    if supervised:
        signature["plan_supervision"] = {"prompt_version": PLAN_PROMPT_VERSION, "objective": "plan_ce",
            "trained_steps": 42, "weight": 1., "annotations_manifest_sha256": "a" * 64,
            "annotations_sha256": "b" * 64, "annotations_revision": "c" * 40,
            "annotations_repo_id": "test/private"}
    monkeypatch.setattr(qwen_vla_task, "teacher_identity", lambda *a: (tmp_path, signature))
    monkeypatch.setattr(QwenVLA, "load", lambda *a: pytest.fail("model loaded during preflight"))
    output = tmp_path / "never-created"
    if supervised:
        qwen_vla_task.run(settings, output, preflight=True)
        assert json.loads(capsys.readouterr().out)["status"] == "preflight_ok"
    else:
        with pytest.raises(InvalidTeacherPlan, match="no Plan supervision"):
            qwen_vla_task.run(settings, output, preflight=True)
    assert not output.exists()


def test_bf16_replay_leaves_backbone_gradients_and_teacher_outside_autocast(monkeypatch):
    proxy = SimpleNamespace(**{key: getattr(torch, key) for key in dir(torch)})
    proxy.autocast = lambda *a, **kw: torch.autocast("cpu", dtype=torch.bfloat16)
    monkeypatch.setattr(qwen_vla_task, "torch", proxy)
    model = tiny_student().train()
    configure_qwen_memory(model, config())
    monkeypatch.setattr(qwen_vla_task, "observation_inputs", lambda *a: tiny_inputs())
    seen = []
    def plan(sample, **kw):
        seen.append(torch.is_autocast_enabled("cpu"))
        return "Move cup."
    teacher = SimpleNamespace(get_plan=plan, get_signal=lambda sample: SimpleNamespace(
        action=torch.zeros(8, 7), plan=plan(sample)))
    def query(t):
        return {"episode_id": "episode", "timestep": t, "proprio": np.zeros(8, dtype=np.float32),
                "gt_action": np.ones((8, 7), dtype=np.float32),
                "action_valid_mask": np.ones((8, 7), dtype=bool)}
    loss, _, _ = qwen_vla_task.action_loss(model, None, None, None, "cpu", plan_teacher=teacher)
    # Reconstruct history after a gap/resume before the differentiable current query.
    loss((query(8), None, query(0)), 0).backward()
    assert seen == [False, False]
    for module in (model.vision_backbone.patch_embed, model.vision_backbone.merger, model.language_model):
        assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0 for p in module.parameters())


def test_host_ram_guard_rejects_before_model_transfer(monkeypatch):
    import vlade.training.residency as residency
    meminfo = SimpleNamespace(is_file=lambda: True, read_text=lambda: "MemAvailable: 1048576 kB\n")
    monkeypatch.setattr(residency, "Path", lambda *a: meminfo)
    with pytest.raises(RuntimeError, match="high-RAM"):
        residency.require_host_memory(1024**3, purpose="synthetic offload")


def test_worker_offloads_before_emitting_paired_signal(tmp_path, monkeypatch):
    """Exercise the real worker entry point/JSON protocol with a tiny fake device policy."""
    import vlade.storage.assets as assets
    import vlade.data.episode_data as episode_data
    import vlade.training.fixed_cache as fixed_cache
    loaded = []
    def sample(t): return {"episode_id": "a", "timestep": t}
    data = SimpleNamespace(episodes=lambda split, order: iter([({}, [sample(0), sample(8)])]))
    monkeypatch.setattr(assets, "resolve_input", lambda value: tmp_path)
    monkeypatch.setattr(episode_data, "EpisodeData", lambda value: data)
    signature = {"temporal_feature_layer": -2}
    monkeypatch.setattr(teacher_inference, "teacher_identity", lambda *a: (tmp_path, signature))
    class Policy:
        def __init__(self, root, *, device, **kw):
            loaded.append(device)
            self.model = torch.nn.Linear(1, 1)
            self.head = self.proprio = self.generator = None
            self.device, self.context = torch.device(device), None
        def reset(self): self.context = None
        def predict(self, sample, data):
            self.context = torch.ones(1)
            return torch.full((8, 7), .25), None
        def plan(self, *a, **kw): return "Move cup."
    monkeypatch.setattr(fixed_cache, "SequentialTeacher", Policy)
    monkeypatch.setattr(sys, "argv", ["teacher", "--checkpoint", "hfasset://T1", "--data", "hfasset://data",
                                       "--device", "cpu", "--plan-max-new-tokens", "48", "--offload-between-requests"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"episode_id": "a", "timestep": 8,
                                                             "signal": "action_and_plan"}) + "\n"))
    wire = io.StringIO()
    monkeypatch.setattr(sys, "stdout", wire)
    teacher_inference.main()
    rows = [json.loads(line) for line in wire.getvalue().splitlines()]
    assert loaded == ["cpu"] and rows[0] == {"ready": signature}
    assert rows[1]["plan"] == "Move cup." and rows[1]["action"] == [[.25] * 7] * 8
