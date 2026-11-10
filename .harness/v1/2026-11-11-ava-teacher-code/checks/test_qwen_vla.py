"""Native Qwen vision/merger + AVA state, without external vision/projector."""
import os
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(os.environ["VLADE_PROJECT_ROOT"]) / "src"))
from vlade.models.students.configuration_qwen_vla import QwenVLAConfig
from vlade.models.students.qwen_vla import QwenVLA
from vlade.models.students.backbones.qwen import prepare_observation, verified_initial
from vlade.models.students.components.qwen_attention import language_forward, visual_attention_mask
from qwen_vla_smoke import tiny_inputs, tiny_student


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(42)


def test_ava_mask_matches_equation_on_noncontiguous_native_image_spans():
    omega = torch.tensor([[[1.9],[.1]]],requires_grad=True)
    visual = torch.tensor([[False,True,False,False,True,False,False]])
    logits = torch.randn(1,1,7,7)
    actual = (logits + visual_attention_mask(torch.ones(1,7),omega,visual_mask=visual)).softmax(-1)
    scores = logits.exp()*torch.ones(7,7).tril()
    weights = torch.ones(7,7)
    weights[:,1],weights[:,4] = 1.9,.1
    weights.fill_diagonal_(1.)
    expected = scores*weights
    expected = expected / expected.sum(-1,keepdim=True)
    torch.testing.assert_close(actual,expected)
    actual[0,0,6,1].backward()
    assert omega.grad.abs().sum() > 0


def test_native_vision_merger_scatter_and_mrope_match_unmodified_qwen():
    model = tiny_student().eval()
    inputs = tiny_inputs()
    with torch.no_grad():
        native = model.backbone(**inputs,use_cache=False,output_hidden_states=True,return_dict=True)
        embeddings,visual,_,mask,positions = model.encode_observation(inputs)
        final,previous = language_forward(model.language_model,embeddings,inputs["attention_mask"],
            torch.ones(1,8,1),visual_mask=mask,position_ids=positions)
    torch.testing.assert_close(final,native.last_hidden_state)
    torch.testing.assert_close(previous,native.hidden_states[-2])
    assert visual.shape == (1,8,32)
    assert not torch.equal(positions[0],positions[1])  # genuine spatial M-RoPE
    assert not hasattr(model,"projector") and not hasattr(model,"vision_asset")
    assert model.ava.sink_ids is None


def test_temporal_state_changes_actions_and_new_episode_is_reproducible():
    model = tiny_student().eval()
    inputs,proprio = tiny_inputs(),torch.randn(1,8)
    with torch.no_grad():
        first = model(inputs,proprio)
        second = model(inputs,proprio,first.recurrent_state)
        reset = model(inputs,proprio,None)
    assert first.actions.shape == (1,8,7) and first.recurrent_state.shape == (1,56,32)
    assert not torch.allclose(first.actions,second.actions)
    torch.testing.assert_close(first.actions,reset.actions)
    assert model.modulated_layers == [3]
    assert model.backbone.rope_deltas is None


def test_two_frame_backward_reaches_native_vision_merger_ava_and_previous_state():
    model = tiny_student()
    inputs,proprio = tiny_inputs(),torch.randn(1,8)
    first = model(inputs,proprio)
    first.recurrent_state.retain_grad()
    second = model(inputs,proprio,first.recurrent_state,batch_idx=1)
    second.actions.square().mean().backward()
    assert first.recurrent_state.grad.abs().sum() > 0
    for module in (model.vision_backbone.patch_embed,model.vision_backbone.merger,model.language_model,
                   model.ava,model.proprio_projector,model.action_head):
        assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
                   for p in module.parameters())


def test_image_change_reaches_action_head():
    model = tiny_student().eval()
    inputs,proprio = tiny_inputs(),torch.randn(1,8)
    changed = {**inputs,"pixel_values":inputs["pixel_values"].clone()}
    changed["pixel_values"][:16] = torch.randn_like(changed["pixel_values"][:16])*2
    with torch.no_grad():
        assert not torch.allclose(model(inputs,proprio).actions,model(changed,proprio).actions)


def test_bfloat16_checkpoint_keeps_native_vision_and_component_dtypes(tmp_path):
    model = tiny_student().to(dtype=torch.bfloat16)
    model.save(tmp_path / "bf16")
    loaded = QwenVLA.load(tmp_path / "bf16")
    assert {p.dtype for p in loaded.parameters()} == {torch.bfloat16}


def test_save_reload_and_corrupt_checkpoint_rejection(tmp_path):
    model = tiny_student().eval()
    inputs,proprio = tiny_inputs(),torch.randn(1,8)
    model.save(tmp_path / "saved")
    restored = QwenVLA.load(tmp_path / "saved").eval()
    with torch.no_grad():
        torch.testing.assert_close(model(inputs,proprio).actions,restored(inputs,proprio).actions)
    weights = torch.load(tmp_path / "saved/components.pt",weights_only=True)
    assert not any("projector.fc3" in key or "vision_backbone" in key for key in weights)
    weights.pop(next(iter(weights)))
    torch.save(weights,tmp_path / "saved/components.pt")
    with pytest.raises(ValueError,match="component keys"):
        QwenVLA.load(tmp_path / "saved")


@pytest.mark.parametrize("bad",["labels","past_key_values","position_ids","pixel_values_videos"])
def test_reject_gt_cache_position_overrides_and_video(bad):
    inputs = tiny_inputs()
    inputs[bad] = torch.zeros(1)
    with pytest.raises(ValueError,match="Expected current native"):
        tiny_student()(inputs,torch.zeros(1,8))


def test_reject_history_images_and_inconsistent_placeholders():
    model = tiny_student()
    inputs = tiny_inputs()
    bad = {**inputs,"image_grid_thw":torch.tensor([[1,4,4]]*4)}
    with pytest.raises(ValueError,match="exactly two"):
        model(bad,torch.zeros(1,8))
    bad = {**inputs,"input_ids":inputs["input_ids"].clone()}
    bad["input_ids"][0,3] = 10
    with pytest.raises(ValueError,match="run length"):
        model(bad,torch.zeros(1,8))


def test_hf_boundary_and_no_openvla_dependency(monkeypatch,tmp_path):
    import vlade.storage.assets as assets
    calls = []
    def accept(path):
        calls.append(path)
        return tmp_path
    monkeypatch.setattr(assets,"resolve_input",accept)
    config = QwenVLAConfig()
    (tmp_path / "download_manifest.json").write_text(json.dumps(
        dict(repository=config.model_id,revision=config.revision)),encoding="utf-8")
    assert verified_initial(config,"hfasset://qwen_initial") == tmp_path
    assert calls == ["hfasset://qwen_initial"]
    def reject(path):
        raise ValueError("unregistered HF input")
    monkeypatch.setattr(assets,"resolve_input",reject)
    with pytest.raises(ValueError,match="unregistered"):
        verified_initial(config,Path("unregistered-qwen"))


def test_reject_previously_saved_external_vision_schema():
    with pytest.raises(ValueError,match="schema 2"):
        replace(QwenVLAConfig(),schema_version=1).validate()


def test_native_full_checkpoint_load_preserves_vision_merger_and_language(monkeypatch,tmp_path):
    from transformers import Qwen3_5ForConditionalGeneration
    import vlade.storage.assets as assets
    model = tiny_student()
    native = Qwen3_5ForConditionalGeneration(model.backbone.config)
    native.model.load_state_dict(model.backbone.state_dict())
    native.save_pretrained(tmp_path)
    config = model.student_config
    (tmp_path / "download_manifest.json").write_text(json.dumps(
        dict(repository=config.model_id,revision=config.revision)),encoding="utf-8")
    monkeypatch.setattr(assets,"resolve_input",lambda path:tmp_path)
    loaded = QwenVLA.from_pretrained(config,qwen_initial=tmp_path)
    for key,value in model.backbone.state_dict().items():
        torch.testing.assert_close(value,loaded.backbone.state_dict()[key])


def test_processor_receives_only_two_current_cameras_and_native_template():
    from PIL import Image
    calls = []
    class ProcessorFixture:
        def apply_chat_template(self,messages,**kwargs):
            content = messages[0]["content"]
            images = [item["image"] for item in content if item["type"] == "image"]
            assert len(images) == 2
            assert images[0].getpixel((0,0)) == (255,0,0)
            assert images[1].getpixel((0,0)) == (0,0,255)
            assert kwargs["tokenize"] and kwargs["return_dict"] and not kwargs["enable_thinking"]
            calls.append(messages)
            return tiny_inputs()
    inputs = prepare_observation(ProcessorFixture(),tiny_student().student_config,
        instruction="pick cup",primary=Image.new("RGB",(8,8),"red"),wrist=Image.new("RGB",(8,8),"blue"))
    assert inputs["image_grid_thw"].shape == (2,3) and len(calls) == 1


def test_native_image_processor_pixels_reach_native_vlm_and_action_head():
    from PIL import Image
    from transformers import Qwen2VLImageProcessorPil
    processor = Qwen2VLImageProcessorPil(patch_size=2,temporal_patch_size=2,merge_size=2,
        size={"shortest_edge":64,"longest_edge":64},image_mean=[.5,.5,.5],image_std=[.5,.5,.5])
    packed = processor(images=[Image.new("RGB",(8,8),"red"),Image.new("RGB",(8,8),"blue")],
                       return_tensors="pt")
    model = tiny_student().eval()
    inputs = {**tiny_inputs(),**dict(packed)}
    with torch.no_grad():
        output = model(inputs,torch.zeros(1,8))
    assert output.actions.shape == (1,8,7)
    assert output.visual_weights.shape == (1,8,1)


def test_named_policy_and_components_are_separate_modules():
    from vlade.models.students.components.action_heads import L1RegressionActionHead
    from vlade.models.students.components.projectors import ProprioProjector
    from vlade.models.students.components.attn_weight_generator import AttentionWeightGenerator
    model = tiny_student()
    assert type(model).__name__ == "QwenVLA"
    assert type(model).__module__ == "vlade.models.students.qwen_vla"
    assert isinstance(model.action_head, L1RegressionActionHead)
    assert isinstance(model.proprio_projector, ProprioProjector)
    assert isinstance(model.ava, AttentionWeightGenerator)


def test_student_components_have_no_teacher_runtime_dependency():
    model = tiny_student()
    for module in (model.action_head, model.proprio_projector, model.ava):
        assert type(module).__module__.startswith("vlade.models.students.components.")
    import ast
    root = Path(os.environ["VLADE_PROJECT_ROOT"]) / "src/vlade/models/students"
    for file in list((root / "components").glob("*.py")) + list((root / "backbones").glob("*.py")):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("prismatic", "vlade.models.teachers", "vlade.models.components", "vlade.models.backbones"))
