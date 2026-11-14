"""Active AVA Teacher subset: upstream equivalence without weights or GPU."""

from __future__ import annotations
import os

import ast
import copy
import logging
import json
import os
import subprocess
import sys
import typing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput

ROOT = Path(os.environ["VLADE_PROJECT_ROOT"])
sys.path.insert(0, str(ROOT / "src"))
from vlade.models.teachers import action_masks, constants
from vlade.models.teachers.components.action_heads import L1RegressionActionHead
from vlade.models.teachers.components.attn_weight_generator import AttentionWeightGenerator
from vlade.models.teachers.components.projectors import PrismaticProjector, ProprioProjector
from vlade.models.teachers.configuration_ava_vla import OpenVLAConfig, PrismaticConfig

TEACHER = ROOT / "src/vlade/models/teachers"
UPSTREAM = ROOT / "third_party/ava_vla/prismatic"


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(42)


def definitions(path: Path) -> dict[str, ast.AST]:
    return {
        n.name: n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    }


def test_retained_definitions_match_pinned_source():
    pairs = [
        ("components/action_heads.py", "models/action_heads.py"),
        ("components/projectors.py", "models/projectors.py"),
        ("backbones/prismatic.py", "extern/hf/modeling_prismatic.py"),
        ("action_masks.py", "training/train_utils.py"),
        ("configuration_ava_vla.py", "extern/hf/configuration_prismatic.py"),
        ("processing_ava_vla.py", "extern/hf/processing_prismatic.py"),
        ("components/attn_weight_generator.py", "models/attn_weight_generator.py"),
        ("components/attention_utils.py", "models/attn_weight_generator.py"),
        ("components/attention_blocks.py", "models/attn_weight_generator.py"),
        ("components/attention_heads.py", "models/attn_weight_generator.py"),
        ("outputs.py", "extern/hf/modeling_prismatic.py"),
    ]
    for local, original in pairs:
        actual, expected = definitions(TEACHER / local), definitions(UPSTREAM / original)
        for name, node in actual.items():
            if name == "_load_processor":
                continue
            if name == "AttentionWeightGenerator":
                continue  # Only its head registry/unsupported-mode guard was narrowed.
            expected_node = expected.get(name)
            if name == "PrismaticProjector":
                expected_node = definitions(UPSTREAM / "extern/hf/modeling_prismatic.py")[name]
            if name == "PrismaticVisionBackbone":
                # Only defer timm imports to actual backbone construction.
                node = copy.deepcopy(node)
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        method.body = [
                            n
                            for n in method.body
                            if not isinstance(n, (ast.Import, ast.ImportFrom))
                        ]
            assert ast.dump(node) == ast.dump(expected_node), (local, name)
    assert (TEACHER / "AVA_LICENSE.txt").read_text(encoding="utf-8") == (
        ROOT / "third_party/ava_vla/LICENSE"
    ).read_text(encoding="utf-8")


def test_only_active_teacher_paths_and_libero_constants(monkeypatch):
    for path in TEACHER.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imports = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom) and node.level == 0
                else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            )
            assert not any(
                name.startswith(("prismatic", "vlade.models.students")) for name in imports
            )
        names = definitions(path)
        assert not set(names) & {
            "DiffusionActionHead",
            "STEHead",
            "GumbelSoftmaxHead",
            "SigmoidScoreHead",
        }
    assert set(AttentionWeightGenerator.head_modules) == {"softmaxscore"}
    monkeypatch.setattr(sys, "argv", ["train", "aloha", "bridge"])
    assert (constants.NUM_ACTIONS_CHUNK, constants.ACTION_DIM, constants.PROPRIO_DIM) == (8, 7, 8)
    assert list(constants.NormalizationType) == [constants.NormalizationType.BOUNDS_Q99]


def test_native_action_masks_reject_non_action_tokens():
    labels = torch.tensor([[-100, -100] + [31744] * 56 + [2]])
    current = action_masks.get_current_action_mask(labels)
    future = action_masks.get_next_actions_mask(labels)
    assert current.sum() == 7 and future.sum() == 49
    assert not (current | future)[:, :2].any()
    assert not (current | future)[:, -1].any()
    assert not action_masks.get_current_action_mask(torch.ones_like(labels)).any()


def model_class(path: Path, monkeypatch, name: str):
    """Execute actual model classes with a synthetic vision/language dependency.

    This checks Teacher assembly, attention plumbing and L1/temporal math; it
    does not load timm, torchvision, the AVA Transformers fork or pretrained VLMs.
    """
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    namespace = module.__dict__
    namespace.update(vars(typing))
    namespace.update(vars(constants))
    namespace.update(
        dict(
            __name__=name,
            torch=torch,
            nn=nn,
            np=np,
            dataclass=dataclass,
            ModelOutput=ModelOutput,
            PreTrainedModel=nn.Module,
            PretrainedConfig=PrismaticConfig,
            PrismaticConfig=PrismaticConfig,
            OpenVLAConfig=OpenVLAConfig,
            logger=logging.getLogger(name),
            PrismaticProjector=PrismaticProjector,
            get_current_action_mask=action_masks.get_current_action_mask,
            get_next_actions_mask=action_masks.get_next_actions_mask,
        )
    )
    namespace["NormalizationType"] = SimpleNamespace(
        BOUNDS="bounds", BOUNDS_Q99=constants.NormalizationType.BOUNDS_Q99
    )
    sources = (
        [
            TEACHER / "outputs.py",
            TEACHER / "components/multimodal.py",
            TEACHER / "backbones/vlm.py",
            TEACHER / "backbones/llama.py",
            TEACHER / "components/action_prediction.py",
            path,
        ]
        if path == TEACHER / "ava_vla.py"
        else [path]
    )
    selected = [
        node
        for source in sources
        for key, node in definitions(source).items()
        if key
        in {
            "PrismaticCausalLMOutputWithPast",
            "PrismaticPreTrainedModel",
            "MultimodalForwardMixin",
            "PrismaticForConditionalGeneration",
            "L1ActionPredictionMixin",
            "OpenVLAForActionPrediction",
        }
    ]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["OpenVLAForActionPrediction"]


def test_ava_generator_state_and_outputs_match_upstream():
    namespace = dict(
        ast=ast,
        np=np,
        torch=torch,
        nn=nn,
        F=nn.functional,
        partial=partial,
        ClassVar=typing.ClassVar,
        trunc_normal_=nn.init.trunc_normal_,
        DEFAULT_NORM_LAYER=partial(nn.LayerNorm, eps=1e-6),
    )
    tree = ast.Module(
        body=list(definitions(UPSTREAM / "models/attn_weight_generator.py").values()),
        type_ignores=[],
    )
    exec(compile(tree, "pinned-ava-generator", "exec"), namespace)
    kwargs = dict(
        embed_dim=128,
        image_shape=(2, 2, 32),
        action_shape=(8, 7, 32),
        text_dim=32,
        num_images=2,
        max_steps=10,
        sink_ids=[1, 5],
        sink_weight=1.0,
    )
    original, local = namespace["AttentionWeightGenerator"](**kwargs), AttentionWeightGenerator(
        **kwargs
    )
    local.load_state_dict(original.state_dict(), strict=True)
    image, state, text = torch.randn(1, 8, 32), torch.randn(1, 56, 32), torch.randn(1, 3, 32)
    torch.manual_seed(17)
    expected = original(image, state, text, batch_idx=1)
    torch.manual_seed(17)
    actual = local(image, state, text, batch_idx=1)
    torch.testing.assert_close(actual, expected)
    for output in (expected, actual):
        (output[0].square().mean() + output[1][0].square().mean()).backward()
    for (name, value), (expected_name, ref) in zip(
        local.named_parameters(), original.named_parameters()
    ):
        assert name == expected_name
        torch.testing.assert_close(value.grad, ref.grad)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 16)

    def forward(self, pixels):
        return self.projection(pixels)

    def get_num_patches(self):
        return 4

    def get_num_images_in_input(self):
        return 2


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32064, 32)
        self.mixer = nn.Linear(32, 32)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, extra_attn_weights=None, **kwargs):
        weighted = (
            inputs_embeds if extra_attn_weights is None else inputs_embeds * extra_attn_weights
        )
        hidden = inputs_embeds + self.mixer(weighted.mean(1, keepdim=True))
        return SimpleNamespace(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=(inputs_embeds, hidden, hidden * 1.1),
            attentions=None,
        )


def tiny_components(cls):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False, output_hidden_states=True, use_return_dict=True
    )
    model.vision_backbone, model.language_model = TinyVision(), TinyLanguage()
    model.projector = PrismaticProjector(True, 16, 32)
    model.llm_dim, model.extra_layer_ids = 32, [0]
    stats = dict(q01=[-1.0] * 7, q99=[1.0] * 7, min=[-1.0] * 7, max=[1.0] * 7, mask=[False] * 7)
    model.norm_stats = {"normalized": {"action": stats}}
    return nn.ModuleDict(
        dict(
            model=model,
            head=L1RegressionActionHead(32, 64, 7),
            proprio=ProprioProjector(32, 8),
            generator=AttentionWeightGenerator(
                128,
                image_shape=(2, 2, 32),
                action_shape=(8, 7, 32),
                text_dim=32,
                num_images=2,
                max_steps=10,
                sink_ids=[1, 5],
                sink_weight=1.0,
            ),
        )
    )


@pytest.fixture
def reference_and_local(monkeypatch):
    original = model_class(
        UPSTREAM / "extern/hf/modeling_prismatic.py", monkeypatch, "ava_reference"
    )
    local = model_class(TEACHER / "ava_vla.py", monkeypatch, "ava_candidate")
    reference, candidate = tiny_components(original), tiny_components(local)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    return reference, candidate


def test_two_frame_forward_gradients_and_strict_state_reload_match_upstream(
    reference_and_local, tmp_path
):
    reference, candidate = reference_and_local
    inputs = dict(
        input_ids=torch.tensor([[1, 10, 29871] + [1] * 56 + [2]]),
        attention_mask=torch.ones(1, 60, dtype=torch.long),
        pixel_values=torch.randn(1, 8, 16),
        labels=torch.tensor([[-100] * 3 + [31744] * 56 + [2]]),
        proprio=torch.randn(1, 8),
        use_cache=False,
    )
    outputs = []
    for bundle in (reference, candidate):
        torch.manual_seed(17)  # Match the retained AVA dropout masks in both executions.
        first = bundle["model"](
            **inputs,
            proprio_projector=bundle["proprio"],
            vision_attn_weight_generator=bundle["generator"],
            batch_idx=0,
        )
        state = first.hidden_states[-2][:, 12:68]
        state.retain_grad()
        second = bundle["model"](
            **inputs,
            proprio_projector=bundle["proprio"],
            vision_attn_weight_generator=bundle["generator"],
            temporal_context=state,
            batch_idx=1,
        )
        predicted = bundle["head"].predict_action(second.hidden_states[-1][:, 12:68])
        predicted.square().mean().backward()
        assert state.grad is not None and state.grad.abs().sum() > 0
        outputs.append((second.hidden_states[-1], second.vision_attn_weights, predicted))
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected)
    for (name, actual), (refname, expected) in zip(
        candidate.named_parameters(), reference.named_parameters()
    ):
        assert name == refname
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad)
    for component in candidate.values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    path = tmp_path / "teacher-components.pt"
    torch.save(candidate.state_dict(), path)
    restored = copy.deepcopy(candidate)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[name])


def test_l1_prediction_temporal_state_and_episode_reset_match_upstream(reference_and_local):
    reference, candidate = reference_and_local
    inputs = dict(
        input_ids=torch.tensor([[1, 10, 29871]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        pixel_values=torch.randn(1, 8, 16),
        proprio=np.zeros(8),
        temporal_feature_layer=-2,
        unnorm_key="normalized",
    )
    results = []
    for bundle in (reference.eval(), candidate.eval()):
        kwargs = dict(
            action_head=bundle["head"],
            proprio_projector=bundle["proprio"],
            vision_attn_weight_generator=bundle["generator"],
        )
        with torch.no_grad():
            first = bundle["model"].predict_action(**inputs, **kwargs)
            second = bundle["model"].predict_action(**inputs, **kwargs, temporal_context=first[1])
            reset = bundle["model"].predict_action(**inputs, **kwargs, temporal_context=None)
        assert first[0].shape == (8, 7) and first[1].shape == (1, 56, 32)
        np.testing.assert_allclose(first[0], reset[0])
        assert not np.allclose(first[0], second[0])
        results.append((first, second))
    for actual_step, expected_step in zip(results[1], results[0]):
        for actual, expected in zip(actual_step, expected_step):
            if isinstance(actual, np.ndarray):
                np.testing.assert_allclose(actual, expected)
            else:
                torch.testing.assert_close(actual, expected)


def test_unused_modes_fail_explicitly(reference_and_local):
    _, candidate = reference_and_local
    with pytest.raises(ValueError, match="continuous L1"):
        candidate["model"].predict_action(action_head=None)
    with pytest.raises(ValueError, match="continuous L1"):
        candidate["model"](noisy_actions=torch.zeros(1))
    with pytest.raises(ValueError, match="current camera"):
        candidate["model"](input_ids=torch.tensor([[1, 2]]), pixel_values=None)
    with pytest.raises(ValueError, match="current camera"):
        candidate["model"](past_key_values=[])


def test_config_roundtrip_keeps_active_backbone_and_upstream_defaults(tmp_path):
    config = OpenVLAConfig(
        vision_backbone_id="dinosiglip-vit-so-224px",
        llm_backbone_id="llama2-7b-pure",
        text_config=dict(
            hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64
        ),
    )
    config.save_pretrained(tmp_path)
    restored = OpenVLAConfig.from_pretrained(tmp_path, local_files_only=True)
    assert restored.vision_backbone_id == config.vision_backbone_id
    assert restored.llm_backbone_id == config.llm_backbone_id
    assert restored.text_config.hidden_size == 32
    # Original defaults are needed by HF diff serialization; do not migrate them.
    assert OpenVLAConfig().vision_backbone_id == "siglip-vit-so400m"
    assert OpenVLAConfig().llm_backbone_id == "vicuna-v15-7b"


def test_fft_and_paper_build_teacher_local_components(monkeypatch, tmp_path):
    """Exercise real training factories with weight loading replaced by a tiny fixture."""
    from vlade.models.teachers import configuration_ava_vla
    from vlade.training import fft, paper_ava

    calls = []

    class ConfigFactory:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"local_files_only": True}
            return SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2))

    class ModelFactory(nn.Module):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            assert kwargs["local_files_only"] is True
            result = cls()
            result.config = kwargs["config"]
            result.llm_dim = 512
            result.linear = nn.Linear(2, 2)
            result.vision_backbone = SimpleNamespace(
                set_num_images_in_input=lambda count: None, get_num_patches=lambda: 256
            )
            return result

    registry = SimpleNamespace(register=lambda *args: None)
    transformers = ModuleType("transformers")
    for name in ("AutoConfig", "AutoImageProcessor", "AutoProcessor", "AutoModelForVision2Seq"):
        setattr(transformers, name, registry)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(configuration_ava_vla, "OpenVLAConfig", ConfigFactory)
    model_module = ModuleType("vlade.models.teachers.ava_vla")
    model_module.OpenVLAForActionPrediction = ModelFactory
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    processor_module = ModuleType("vlade.models.teachers.processing_ava_vla")
    processor = object()
    processor_module.PrismaticImageProcessor = object
    processor_module.PrismaticProcessor = object
    processor_module._load_processor = lambda path: processor
    monkeypatch.setitem(sys.modules, processor_module.__name__, processor_module)
    loaded = []
    monkeypatch.setattr(fft, "load_component", lambda module, path, name: loaded.append(name))
    monkeypatch.setattr(fft, "_activate_upstream", lambda model: None)
    cfg = fft.FFTConfig("ava_vla", tmp_path, tmp_path, tmp_path, gradient_checkpointing=False)
    fft_bundle = fft._make_model_and_components(cfg, torch.device("cpu"))
    paper_bundle = paper_ava._model(
        dict(max_steps=10, lora_rank=2, lora_alpha=4, attention_gamma=[1.9,.1], gradient_checkpointing=False),
        tmp_path,
        torch.device("cpu"),
    )
    for bundle in (fft_bundle, paper_bundle):
        assert bundle[1] is processor
        assert isinstance(bundle[2], L1RegressionActionHead)
        assert isinstance(bundle[3], ProprioProjector)
        assert isinstance(bundle[4], AttentionWeightGenerator)
        assert bundle[4].sink_ids == [68, 75, 180, 187, 324, 331, 436, 443]
        assert bundle[4].head.scores == (1.9, 0.1)
    assert loaded == ["action_head", "proprio_projector", "vision_attn_weight_generator"]
    assert len(calls) == 2


def test_processor_loader_uses_local_classes_without_snapshot_python(monkeypatch, tmp_path):
    calls = []

    class ImageProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("image", path, kwargs))
            return "image"

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return "tokenizer"

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    tree = definitions(TEACHER / "processing_ava_vla.py")["_load_processor"]
    namespace = dict(
        Path=Path,
        PrismaticImageProcessor=ImageProcessor,
        PrismaticProcessor=lambda **kwargs: kwargs,
    )
    exec(
        compile(ast.Module(body=[tree], type_ignores=[]), "teacher-processor-loader", "exec"),
        namespace,
    )
    assert namespace["_load_processor"](tmp_path) == dict(
        image_processor="image", tokenizer="tokenizer"
    )
    assert calls == [
        ("image", tmp_path, dict(local_files_only=True)),
        ("tokenizer", tmp_path, dict(local_files_only=True, trust_remote_code=False)),
    ]


def test_teacher_runs_from_minimal_source_without_third_party(tmp_path):
    """Import the real policy in an isolated interpreter, without the repository."""
    stage = tmp_path / "isolated"
    source = stage / "src/vlade/models/teachers"
    required = (
        "ava_vla.py",
        "configuration_ava_vla.py",
        "constants.py",
        "action_masks.py",
        "outputs.py",
        "backbones/prismatic.py",
        "backbones/vlm.py",
        "backbones/llama.py",
        "components/multimodal.py",
        "components/action_prediction.py",
        "components/projectors.py",
        "components/action_heads.py",
        "components/attn_weight_generator.py",
        "components/attention_utils.py",
        "components/attention_blocks.py",
        "components/attention_heads.py",
    )
    for relative in required:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((TEACHER / relative).read_bytes())
    assert not (stage / "third_party").exists()
    helpers = definitions(Path(__file__))
    fixtures = "\n\n".join(
        ast.unparse(helpers[name]) for name in ("TinyVision", "TinyLanguage", "tiny_components")
    )
    script = stage / "check_teacher.py"
    script.write_text(
        """import importlib.abc
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
root = Path(__file__).parent
sys.path.insert(0, str(root / "src"))
class NoUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "prismatic" or fullname.startswith("prismatic."):
            raise AssertionError("Unexpected upstream import: " + fullname)
sys.meta_path.insert(0, NoUpstream())
from vlade.models.teachers.ava_vla import OpenVLAForActionPrediction
from vlade.models.teachers.components.projectors import PrismaticProjector, ProprioProjector
from vlade.models.teachers.components.action_heads import L1RegressionActionHead
from vlade.models.teachers.components.attn_weight_generator import AttentionWeightGenerator
import vlade.models.teachers.ava_vla as policy
assert Path(policy.__file__).resolve().is_relative_to(root.resolve())
assert not (root / "third_party").exists()
torch.set_num_threads(2)
torch.manual_seed(42)
"""
        + fixtures
        + """
bundle = tiny_components(OpenVLAForActionPrediction)
inputs = dict(input_ids=torch.tensor([[1, 10, 29871]+[1]*56+[2]]),
    attention_mask=torch.ones(1, 60, dtype=torch.long), pixel_values=torch.randn(1, 8, 16),
    labels=torch.tensor([[-100]*3+[31744]*56+[2]]), proprio=torch.randn(1, 8), use_cache=False)
first = bundle["model"](**inputs, proprio_projector=bundle["proprio"],
    vision_attn_weight_generator=bundle["generator"], batch_idx=0)
state = first.hidden_states[-2][:, 12:68]
state.retain_grad()
second = bundle["model"](**inputs, proprio_projector=bundle["proprio"],
    vision_attn_weight_generator=bundle["generator"], temporal_context=state, batch_idx=1)
prediction = bundle["head"].predict_action(second.hidden_states[-1][:, 12:68])
prediction.square().mean().backward()
assert prediction.shape == (1, 8, 7)
assert state.grad is not None and state.grad.abs().sum() > 0
for module in bundle.values():
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in module.parameters())
path=root/"state.pt"
torch.save(bundle.state_dict(), path)
restored=tiny_components(OpenVLAForActionPrediction)
restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
bundle.eval(); restored.eval()
args=dict(input_ids=torch.tensor([[1, 10, 29871]]), attention_mask=torch.ones(1,3,dtype=torch.long),
    pixel_values=inputs["pixel_values"], proprio=np.zeros(8), unnorm_key="normalized",
    temporal_feature_layer=-2)
outputs=[]
with torch.no_grad():
    for value in (bundle,restored):
        result=value["model"].predict_action(**args,action_head=value["head"],
            proprio_projector=value["proprio"],vision_attn_weight_generator=value["generator"])
        assert result[0].shape==(8,7) and result[1].shape==(1,56,32)
        outputs.append(result[0])
np.testing.assert_allclose(*outputs)
assert "prismatic" not in sys.modules
print(json.dumps(dict(third_party_absent=True, actual_policy_import=True,
    synthetic_vision_language=True, two_frame_backward=True, strict_reload=True,
    action_shape=[8,7], state_shape=[1,56,32])))
""",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(script)],
        cwd=stage,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["strict_reload"] is True


def test_fft_preflight_reports_remaining_upstream_dataset_dependency(monkeypatch, tmp_path):
    from vlade.training import fft

    monkeypatch.setattr(fft, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fft, "resolve_config_assets", lambda cfg, **kwargs: cfg)
    cfg = fft.FFTConfig("ava_vla", tmp_path, tmp_path, tmp_path, dataset_name="synthetic")
    with pytest.raises(FileNotFoundError, match="Missing local model implementation"):
        fft.preflight(cfg)
