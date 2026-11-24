"""Regression checks; all generated fixtures belong to this task's scratch."""
from contextlib import nullcontext
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys
import subprocess
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "AGENTS.md").is_file()
sys.path.insert(0, str(ROOT / "src"))

from vlade.distillation.teacher_plan import (
    InvalidTeacherPlan, PLAN_PROMPT_VERSION, decode_plan_tokens, normalize_generated_plan,
    plan_generation_kwargs, plan_prompt, require_plan_supervision,
)
from vlade.training import fft, teacher_inference
from vlade.training.fixed_cache import SequentialTeacher
from vlade.training.teacher_plan_supervision import PlanAnnotations, attach_plan_batch, plan_ce_loss
from vlade.models.teachers.components.multimodal import MultimodalForwardMixin


def contract(step=2):
    return {"step": step, "plan_supervision": {
        "prompt_version": PLAN_PROMPT_VERSION, "objective": "plan_ce", "trained_steps": step,
        "weight": 1.0, "annotations_manifest_sha256": "a" * 64, "annotations_sha256": "b" * 64,
        "annotations_revision": "c" * 40, "annotations_repo_id": "test/private"}}


@pytest.mark.parametrize("text", ["ouch", "imas"])
def test_format_filter_cannot_establish_plan_semantics(text):
    assert normalize_generated_plan(text) == text
    with pytest.raises(InvalidTeacherPlan, match="no Plan supervision"):
        require_plan_supervision({"step": 1000})


@pytest.mark.parametrize("field,value", [("objective", "action_l1"), ("trained_steps", 0),
    ("trained_steps", 3), ("prompt_version", "wrong"), ("annotations_manifest_sha256", "x" * 64),
    ("weight", float("nan")), ("weight", 0.0)])
def test_cannot_claim_supervision_with_invalid_metadata(field, value):
    metadata = contract()
    metadata["plan_supervision"][field] = value
    with pytest.raises(InvalidTeacherPlan, match="contract"):
        require_plan_supervision(metadata)


def test_untrained_teacher_is_rejected_before_worker_or_processor(monkeypatch):
    monkeypatch.setattr(teacher_inference, "teacher_identity", lambda *args: (Path("unused"), {"step": 12}))
    monkeypatch.setattr(teacher_inference.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("worker started"))
    with pytest.raises(InvalidTeacherPlan, match="no Plan supervision"):
        teacher_inference.InferenceActions({"teacher_mode": "inference", "teacher_plan_max_new_tokens": 48,
                                           "teacher_checkpoint": "hfasset://T1"}, object())
    policy = SequentialTeacher.__new__(SequentialTeacher)
    policy.checkpoint_metadata = {"step": 12}
    with pytest.raises(InvalidTeacherPlan, match="no Plan supervision"):
        policy.plan({"instruction": "Move cup", "timestep": 0}, None, max_new_tokens=48)


def tiny_components():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    raw = Tokenizer(WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "Move": 3, "cup": 4, ".": 5}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token="<s>", eos_token="</s>",
                                        unk_token="<unk>", pad_token="</s>", model_max_length=256)

    class TinyTeacher(MultimodalForwardMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=2))
            self.vision = torch.nn.Linear(1, 16)
            self.seen_labels = None
            self.language_model.register_forward_pre_hook(self.capture, with_kwargs=True)

        def capture(self, model, args, kwargs):
            if "labels" in kwargs:
                self.seen_labels = kwargs["labels"].detach().clone()

        def get_input_embeddings(self):
            return self.language_model.get_input_embeddings()

        def get_output_embeddings(self):
            return self.language_model.get_output_embeddings()

        def _process_vision_features(self, pixels, **kwargs):
            return self.vision(pixels.float().reshape(pixels.shape[0], -1, 1))

        def forward(self, input_ids, **kwargs):
            # Minimal action forward with the exact hidden alignment used by compute_loss.
            hidden = self.get_input_embeddings()(input_ids)
            return SimpleNamespace(hidden_states=(hidden,))

    model = TinyTeacher()
    head = torch.nn.Linear(16, 1)
    head.predict_action = lambda hidden: head(hidden.float()).reshape(-1, 8, 7)
    proprio = torch.nn.Linear(8, 16)
    row = {"episode_id": "train", "timestep": 0, "instruction": "Move cup."}
    batch = {"input_ids": torch.ones(1, 58, dtype=torch.long),
             "labels": torch.tensor([[-100] + [31744] * 56 + [2]]),
             "attention_mask": torch.ones(1, 58, dtype=torch.long), "pixel_values": torch.ones(1, 2, 1),
             "proprio": torch.ones(1, 8), "actions": torch.zeros(1, 8, 7),
             "sample_metadata": [row], "plan_supervision": {"targets": ["Move cup."], "weight": 1., "tokenizer": tokenizer}}
    return model, head, proprio, batch, tokenizer


def test_eos_and_generated_only_slicing_use_real_generate():
    model, _, _, _, tokenizer = tiny_components()
    llama = model.language_model.eval()
    llama.lm_head = torch.nn.Linear(16, 8, bias=True)
    with torch.no_grad():
        llama.lm_head.weight.zero_()
        llama.lm_head.bias.fill_(-10)
        llama.lm_head.bias[2], llama.lm_head.bias[3] = 10, 9
    prefix = torch.randn(1, 7, 16)
    options = plan_generation_kwargs(tokenizer, 8, 4)
    ids = llama.generate(inputs_embeds=prefix, attention_mask=torch.ones(1, 7), **options)[0].tolist()
    assert ids == [2]  # inputs_embeds-only returns new IDs, not the seven prefix positions.
    with pytest.raises(InvalidTeacherPlan, match="empty"):
        decode_plan_tokens(ids, tokenizer)
    old = dict(options, min_new_tokens=1)
    forced = llama.generate(inputs_embeds=prefix, attention_mask=torch.ones(1, 7), **old)[0].tolist()
    assert forced == [3, 2]  # Forced text can look valid while EOS is the actual highest logit.
    assert decode_plan_tokens([3, 4, 5, 2], tokenizer) == tokenizer.decode([3, 4, 5])
    with pytest.raises(InvalidTeacherPlan, match="without EOS"):
        decode_plan_tokens([3, 4, 5], tokenizer)


def test_real_tokenizer_has_no_plan_special_tokens_and_fragments_are_whole_ids():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "checkpoints/teacher/avavla-libero-4in1", local_files_only=True)
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (1, 2, 32000)
    for text in ("ouch", "imas", "<plan>", "</plan>"):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        assert not set(ids).intersection(tokenizer.all_special_ids)
        print(json.dumps({"text": text, "ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=False)}))


def test_fft_plan_ce_trains_lm_head_and_masks_prompt_images_proprio(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    model, head, proprio, batch, tokenizer = tiny_components()
    loss = fft.compute_loss(model, head, proprio, None, batch, torch.device("cpu"), 0, 0, plan_loss_weight=1.)
    loss.backward()
    for module in (head, model.language_model.lm_head, model.vision, proprio):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    labels = model.seen_labels[0]
    target = tokenizer("Move cup.", add_special_tokens=False).input_ids + [2]
    assert labels[labels != -100].tolist() == target
    assert (labels[:-(len(target))] == -100).all()
    model.zero_grad()
    action_batch = {key: value for key, value in batch.items() if key != "plan_supervision"}
    fft.compute_loss(model, head, proprio, None, action_batch, torch.device("cpu"), 0, 0).backward()
    assert model.language_model.lm_head.weight.grad is None
    with pytest.raises(ValueError, match="missing"):
        fft.compute_loss(model, head, proprio, None, action_batch, torch.device("cpu"), 0, 0, plan_loss_weight=1.)


def test_tiny_supervised_fit_and_four_query_generation_smoke():
    model, _, proprio, batch, tokenizer = tiny_components()
    optimizer = torch.optim.AdamW([*model.parameters(), *proprio.parameters()], lr=0.02)
    initial = float(plan_ce_loss(model, proprio, batch, torch.device("cpu")).detach())
    for _ in range(40):
        optimizer.zero_grad()
        loss = plan_ce_loss(model, proprio, batch, torch.device("cpu"))
        loss.backward()
        optimizer.step()
    ending = float(loss.detach())
    assert ending < initial * .15
    model.eval()
    policy = SequentialTeacher.__new__(SequentialTeacher)
    policy.model, policy.proprio, policy.timings, policy.device = model, proprio, None, torch.device("cpu")
    policy.context = torch.randn(1, 56, 16)
    policy.checkpoint_metadata = contract(40)
    prompts = []

    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer

        def __call__(self, prompt, image, **kwargs):
            prompts.append(prompt)
            encoded = tokenizer(prompt, return_tensors="pt")
            return {**encoded, "pixel_values": torch.ones(1, 1, 1)}

    policy.processor = Processor()
    data = SimpleNamespace(image=lambda *args: None)
    old = policy.context.clone()
    outputs = []
    for split in ("train", "validation"):
        for timestep in (0, 8):
            row = dict(batch["sample_metadata"][0], episode_id=split, timestep=timestep, proprio=[1.] * 8)
            result = policy.plan(row, data, max_new_tokens=16)
            assert result == tokenizer.decode(tokenizer("Move cup.", add_special_tokens=False).input_ids)
            assert plan_prompt(row["instruction"], timestep) in prompts
            outputs.append({"split": split, "timestep": timestep, "plan": result})
    torch.testing.assert_close(old, policy.context)
    print(json.dumps({"synthetic_smoke": outputs, "initial_ce": initial, "ending_ce": ending}))


def make_annotations(tmp_path, monkeypatch):
    from vlade.storage.assets import sha256
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "manifest.json").write_text("{}", encoding="utf-8")
    data = SimpleNamespace(root=data_root, manifest={"split_sha256": "s", "normalization_sha256": "n"},
                           entries=[{"episode_id": "train", "split": "train", "queries": 2},
                                    {"episode_id": "val", "split": "validation", "queries": 2}])
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    manifest = {"schema_version": 1, "prompt_version": PLAN_PROMPT_VERSION, "annotation_source": "human_reviewed",
                "prepared_data_identity": {"manifest_sha256": sha256(data_root / "manifest.json"),
                                           **data.manifest}}
    (annotations / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [{"episode_id": "train", "split": "train", "timestep": t, "instruction": "Move cup.",
             "plan": "Move cup."} for t in (0, 8)]
    (annotations / "plans.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    calls = []
    monkeypatch.setattr("vlade.storage.assets.resolve_input_record", lambda reference: (
        calls.append(reference) or annotations, {"format": "directory", "revision": "a" * 40, "repo_id": "test/private",
                                               "files": {"manifest.json": {}, "plans.jsonl": {}}}))
    return annotations, data, calls


@pytest.mark.parametrize("reference", ["hfasset://plan_annotations", "verified/local/cache"])
def test_annotations_verify_uri_and_local_paths_and_batch_targets(tmp_path, monkeypatch, reference):
    annotations, data, calls = make_annotations(tmp_path, monkeypatch)
    records = PlanAnnotations(reference, data)
    assert calls == [reference]
    batch = {"sample_metadata": [{"episode_id": "train", "timestep": 8, "instruction": "Move cup."}]}
    attach_plan_batch(batch, records, .5, None)
    assert batch["plan_supervision"]["targets"] == ["Move cup."]
    with pytest.raises(ValueError, match="Missing"):
        records.target({"episode_id": "val", "timestep": 0, "instruction": "Move cup."})
    with pytest.raises(ValueError, match="Missing"):
        records.target({"episode_id": "train", "timestep": 0, "instruction": "different"})


@pytest.mark.parametrize("damage", ["validation", "duplicate", "missing", "data_identity"])
def test_annotation_leaks_and_incomplete_targets_are_rejected(tmp_path, monkeypatch, damage):
    annotations, data, _ = make_annotations(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (annotations / "plans.jsonl").read_text().splitlines()]
    if damage == "validation":
        rows[0].update(episode_id="val", split="validation")
    if damage == "duplicate":
        rows.append(rows[0])
    if damage == "missing":
        rows.pop()
    if damage == "data_identity":
        data.manifest["split_sha256"] = "changed"
    (annotations / "plans.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    with pytest.raises(ValueError):
        PlanAnnotations("hfasset://annotations", data)


def test_unregistered_annotation_path_is_not_accepted(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        PlanAnnotations(tmp_path / "not_registered", None)


def test_unpinned_annotation_files_are_not_accepted(tmp_path, monkeypatch):
    annotations, data, _ = make_annotations(tmp_path, monkeypatch)
    monkeypatch.setattr("vlade.storage.assets.resolve_input_record", lambda reference: (
        annotations, {"format": "directory", "files": {"other": {}}}))
    with pytest.raises(ValueError, match="both be pinned"):
        PlanAnnotations("hfasset://annotations", data)


def test_resume_refuses_adding_plan_loss_to_action_checkpoint(tmp_path, monkeypatch):
    metadata = {"step": 2, "configuration": {}}
    for name, content in (("vlade_fft.json", json.dumps(metadata)), ("COMPLETE", "ok"), ("trainer_state.pt", "fake")):
        (tmp_path / name).write_text(content)
    monkeypatch.setattr("vlade.training.run_control.verify_checkpoint", lambda *args, **kwargs: {"step": 2})
    monkeypatch.setattr("vlade.training.teacher_plan_supervision.checkpoint_plan_contract", lambda *args: contract()["plan_supervision"])
    cfg = fft.FFTConfig("ava_vla", tmp_path, tmp_path, tmp_path, resume=tmp_path,
                        plan_annotations=Path("hfasset://annotations"), plan_loss_weight=1.)
    with pytest.raises(ValueError, match="Resume Plan supervision mismatch"):
        fft.validate_resume(cfg)


def test_plan_probe_reports_validation_action_omission_explicitly(tmp_path, monkeypatch):
    old = runpy.run_path(str(ROOT / ".harness/v1/2026-11-24-t1-plan-tags/checks/test_plan_probe.py"))
    old["prepare_probe"](tmp_path, monkeypatch)
    report = teacher_inference.probe_teacher_plans({"teacher_mode": "inference", "teacher_plan_max_new_tokens": 48},
                                                  "hfasset://data", tmp_path / "probe")
    rows = [json.loads(line) for line in report.read_text().splitlines()]
    for row in rows[1:3]:
        assert row["action_shape"] == [8, 7] and row["action_status"] == "predicted"
    for row in rows[3:5]:
        assert row["action_shape"] is None and row["action_status"] == "not_requested_plan_only"


def test_supervised_teacher_json_pipe_returns_signal_and_checks_observation(tmp_path, monkeypatch):
    signature = {"step": 5, "plan_supervision": contract(5)["plan_supervision"]}
    script = tmp_path / "teacher.py"
    script.write_text("import json,sys\nprint(" + repr(json.dumps({"ready": signature})) + ",flush=True)\n"
        "for line in sys.stdin:\n"
        " row=json.loads(line)\n"
        " if row['signal']=='action_and_plan': row['action']=[[0.25]*7 for _ in range(8)]\n"
        " row['plan']='Move cup.'\n"
        " if row['timestep']==16: row['timestep']=8\n"
        " print(json.dumps(row),flush=True)\n", encoding="utf-8")
    popen = subprocess.Popen
    monkeypatch.setattr(teacher_inference, "teacher_identity", lambda *args: (tmp_path, signature))
    monkeypatch.setattr(teacher_inference.subprocess, "Popen",
                        lambda command, **kw: popen([sys.executable, "-B", str(script)], **kw))
    settings = {"teacher_checkpoint": "hfasset://T1", "teacher_python": sys.executable,
                "teacher_mode": "inference", "teacher_plan_max_new_tokens": 48, "teacher_timeout_seconds": 10}
    with closing(teacher_inference.InferenceActions(settings, SimpleNamespace(root=tmp_path))) as teacher:
        signal = teacher.get_signal({"episode_id": "train", "timestep": 0})
        assert signal.action.shape == (8, 7) and signal.plan == "Move cup."
        assert teacher.get_plan({"episode_id": "val", "timestep": 8}, split="validation") == "Move cup."
        with pytest.raises(ValueError, match="requested observation"):
            teacher.get_signal({"episode_id": "train", "timestep": 16})


def test_pinned_annotations_accept_uri_cache_and_reject_tampering(tmp_path, monkeypatch):
    from vlade.storage import assets
    helpers = runpy.run_path(str(ROOT / ".harness/v1/2026-11-11-ava-teacher-code/checks/test_hf_assets.py"))
    data_root = tmp_path / "prepared"
    data_root.mkdir()
    (data_root / "manifest.json").write_text("{}", encoding="utf-8")
    data = SimpleNamespace(root=data_root, manifest={"split_sha256": "s", "normalization_sha256": "n"},
                           entries=[{"episode_id": "train", "split": "train", "queries": 1}])
    manifest = {"schema_version": 1, "prompt_version": PLAN_PROMPT_VERSION, "annotation_source": "human_reviewed",
                "prepared_data_identity": {"manifest_sha256": assets.sha256(data_root / "manifest.json"), **data.manifest}}
    files = {"manifest.json": json.dumps(manifest).encode(), "plans.jsonl": json.dumps({"episode_id": "train",
        "split": "train", "timestep": 0, "instruction": "Move cup.", "plan": "Move cup."}).encode()}
    record = dict(repo_id="test/private", repo_type="dataset", revision="a" * 40, format="directory",
                  prefix="assets/x", local_name="annotations", files={k: helpers["row"](v) for k, v in files.items()})
    registry = helpers["registry_file"](tmp_path, record)
    helpers["mock_download"](monkeypatch, tmp_path, {"assets/x/files/" + k: v for k, v in files.items()})
    monkeypatch.setenv("VLADE_ASSET_REGISTRY", str(registry))
    monkeypatch.setenv("VLADE_ASSET_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("VLADE_LOCAL_INPUTS", raising=False)
    pinned = PlanAnnotations("hfasset://example", data)
    root = assets.resolve_input("hfasset://example")
    assert PlanAnnotations(root, data).identity == pinned.identity
    (root / "plans.jsonl").write_bytes(files["plans.jsonl"].replace(b"Move cup.", b"Move mug."))
    for reference in ("hfasset://example", root):
        with pytest.raises(ValueError, match="integrity"):
            PlanAnnotations(reference, data)


def test_plan_ce_validation_retains_action_l1_and_config_is_opt_in(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    model, head, proprio, batch, _ = tiny_components()
    model.eval()
    plain = fft.compute_loss(model, head, proprio, None, batch, torch.device("cpu"), 0, 0)
    configured = fft.compute_loss(model, head, proprio, None, batch, torch.device("cpu"), 0, 0, plan_loss_weight=1.)
    torch.testing.assert_close(plain, configured)
    assert model.seen_labels is None
    cfg = fft.FFTConfig.from_json(ROOT / "configs/experiments/libero_plus_fft.json", "ava_vla")
    assert cfg.plan_annotations is None and cfg.plan_loss_weight == 0
    for change in ({"plan_loss_weight": 1.}, {"plan_annotations": Path("hfasset://annotations")},
                   {"plan_loss_weight": float("nan")}):
        with pytest.raises(ValueError, match="Plan CE"):
            replace(cfg, **change).validate()


def test_supervised_checkpoint_seals_reloads_and_refuses_changed_annotation_bytes(tmp_path, monkeypatch):
    from vlade.training.loop import LoopState, make_scheduler
    from vlade.training.run_control import verify_checkpoint
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    annotations, data, _ = make_annotations(tmp_path, monkeypatch)
    monkeypatch.setattr("vlade.data.episode_data.EpisodeData", lambda *args: data)
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda *args: data.root)
    monkeypatch.setattr(fft, "_prepared_data_identity", lambda cfg: {"prepared": "synthetic"})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "synthetic CPU check")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(total_memory=1024))
    model, head, proprio, batch, tokenizer = tiny_components()
    model.save_pretrained = model.language_model.save_pretrained
    cfg = fft.FFTConfig("ava_vla", tmp_path / "source", tmp_path, tmp_path / "outputs", grad_accumulation_steps=1,
                        prepared_data_root=data.root, plan_annotations=Path("hfasset://annotations"),
                        plan_loss_weight=1., optimizer="adamw", save_optimizer_state=True)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters(), *proprio.parameters()], lr=.001)
    scheduler = make_scheduler(optimizer, cfg.max_steps)
    fft.compute_loss(model, head, proprio, None, batch, torch.device("cpu"), 0, 0, plan_loss_weight=1.).backward()
    optimizer.step()
    result = fft.save_checkpoint(cfg, 1, model, tokenizer, head, proprio, None, optimizer,
                                 SimpleNamespace(dataset_statistics={"synthetic": {}}),
                                 scheduler=scheduler, loop_state=LoopState(1, 1))
    metadata = json.loads((result / "vlade_fft.json").read_text())
    assert verify_checkpoint(result)["step"] == 1
    assert require_plan_supervision(metadata)["trained_steps"] == 1
    fft.validate_resume(replace(cfg, resume=result))
    restored = LlamaForCausalLM.from_pretrained(result, local_files_only=True)
    torch.testing.assert_close(restored.lm_head.weight, model.language_model.lm_head.weight)
    path = annotations / "plans.jsonl"
    path.write_text(path.read_text().replace("Move cup.", "Move mug."), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume Plan supervision mismatch"):
        fft.validate_resume(replace(cfg, resume=result))


def test_teacher_accepts_current_fft_canonical_identity_and_rejects_mismatch(tmp_path, monkeypatch):
    helpers = runpy.run_path(str(ROOT / ".harness/v1/2026-11-16-online-action-kd/checks/test_online_action_kd.py"))
    data, root = helpers["Data"](tmp_path / "data"), tmp_path / "t1"
    helpers["t1"](root, data)
    metadata = json.loads((root / "vlade_fft.json").read_text())
    data.manifest.update(canonical_manifest_sha256="c" * 64, canonical_normalization_sha256="d" * 64)
    metadata["prepared_data_identity"].update({key: data.manifest[key] for key in
        ("canonical_manifest_sha256", "canonical_normalization_sha256")})
    metadata["plan_supervision"] = contract(4)["plan_supervision"]
    (root / "vlade_fft.json").write_text(json.dumps(metadata))
    helpers["seal"](root)
    monkeypatch.setattr("vlade.storage.assets.resolve_input", lambda value: root)
    _, signature = teacher_inference.teacher_identity("hfasset://t1", data)
    assert require_plan_supervision(signature)["trained_steps"] == 4
    data.manifest["canonical_manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="matching data"):
        teacher_inference.teacher_identity("hfasset://t1", data)
