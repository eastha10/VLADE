"""Data continuity, local state, real FFT wiring and resumable optimizer updates."""

from collections import Counter
from itertools import islice
from types import SimpleNamespace
import copy
import torch
import pytest

from vlade.data.sequences import windows
from vlade.training.temporal import sequence_objective, fft_objective
from vlade.training.loop import run_loop, make_scheduler, training_state, restore_training_state


class Data:
    def __init__(self):
        self.entries = [
            dict(episode_id=name, split=split, queries=size, suite=suite)
            for name, split, size, suite in [
                ("a", "train", 5, "spatial"),
                ("b", "train", 10, "spatial"),
                ("g", "train", 8, "goal"),
                ("o", "train", 7, "object"),
                ("l", "train", 6, "libero_10"),
                ("short", "train", 2, "goal"),
                ("val", "validation", 6, "goal"),
                ("test", "test", 8, "goal"),
            ]
        ]

    def episodes(self, split, *, order):
        for name in order:
            entry = next(e for e in self.entries if e["episode_id"] == name)
            assert entry["split"] == split
            yield entry, [
                dict(episode_id=name, timestep=t * 8, instruction=name)
                for t in range(entry["queries"])
            ]


def keys(stream):
    return [[(f["episode_id"], f["timestep"]) for f in seq] for seq in stream]


@pytest.mark.parametrize("sampling", ["all_windows", "episode_uniform"])
def test_windows_preserve_time_splits_and_exact_resume(sampling):
    data = Data()
    full = keys(islice(windows(data, sampling=sampling, seed=9), 120))
    assert all(
        len(set(e for e, _ in seq)) == 1
        and [t for _, t in seq] == list(range(seq[0][1], seq[0][1] + 32, 8))
        for seq in full
    )
    assert not {"val", "test", "short"} & {e for seq in full for e, _ in seq}
    assert full[73:99] == keys(windows(data, sampling=sampling, seed=9, skip=73, limit=26))
    assert len({seq[0][0] for seq in full[:12]}) > 1
    validation = keys(windows(data, split="validation", limit=2))
    assert len(validation) == 2 and all(e == "val" for seq in validation for e, _ in seq)
    with pytest.raises(ValueError):
        list(windows(data, split="test", limit=1))


def test_random_episode_selection_does_not_weight_long_episodes_more():
    data = Data()
    data.entries = [e for e in data.entries if e["episode_id"] in {"a", "b"}]
    counts = Counter(
        seq[0]["episode_id"]
        for seq in windows(data, sampling="episode_uniform", limit=1000, seed=4)
    )
    assert 400 < counts["a"] < 600 and 400 < counts["b"] < 600


def test_balanced_sampling_balances_sequences_and_restores_cursor():
    data = Data()
    options = dict(sampling="episode_uniform", balanced=True, suite_map={}, update_size=8, seed=5)
    stream = list(windows(data, limit=80, **options))
    suites = {e["episode_id"]: e["suite"] for e in data.entries}
    assert Counter(suites[s[0]["episode_id"]] for s in stream) == dict.fromkeys(
        ["spatial", "goal", "object", "libero_10"], 20
    )
    assert keys(stream[11:31]) == keys(windows(data, skip=11, limit=20, **options))
    assert all(
        len({suites[s[0]["episode_id"]] for s in stream[i : i + 4]}) == 4 for i in range(0, 80, 4)
    )


def test_tbptt_frees_segments_and_preserves_values_with_local_reset():
    values = [torch.tensor(float(i + 1), requires_grad=True) for i in range(4)]
    contexts = []

    def frame(value, context):
        if len(contexts) == 2:
            assert values[0].grad is not None
        contexts.append(context)
        hidden = value if context is None else value + context
        return hidden.square(), hidden

    loss = sequence_objective(values, frame, backward_scale=0.5)
    assert contexts[0] is None and not contexts[2].requires_grad
    assert contexts[2].item() == 3 and contexts[3].item() == 6
    assert [v.grad.item() for v in values] == [1.0, 0.75, 4.0, 2.5]
    assert loss.item() == 36.5
    seen = []
    sequence_objective([torch.tensor(2.0)], lambda x, c: (seen.append(c) or x.square(), x))
    assert seen == [None]


def test_streaming_loop_accumulation_and_resume_match_uninterrupted(tmp_path):
    def components():
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.3)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        return model, opt, make_scheduler(opt, 4)

    batches = [list(range(1, 5))] * 20

    def train(model, opt, sched, target, state=None):
        def backward(batch, step, scale):
            return sequence_objective(
                batch,
                lambda x, c: (model(torch.tensor([[float(x)]])).square().mean(), None),
                backward_scale=scale,
            )

        return run_loop(
            modules=[model],
            optimizer=opt,
            scheduler=sched,
            batches=batches,
            loss_fn=lambda *a: pytest.fail("sequence backward adapter must run"),
            backward_fn=backward,
            max_steps=target,
            accumulation_steps=2,
            state=state,
        )

    full, opt, sched = components()
    train(full, opt, sched, 4)
    part, opt, sched = components()
    state = train(part, opt, sched, 2)
    saved = tmp_path / "state.pt"
    torch.save(dict(model=part.state_dict(), trainer=training_state(state, opt, sched)), saved)
    restored, opt, sched = components()
    payload = torch.load(saved, weights_only=True)
    restored.load_state_dict(payload["model"])
    state = restore_training_state(payload["trainer"], opt, sched)
    train(restored, opt, sched, 4, state)
    torch.testing.assert_close(full.weight, restored.weight, rtol=0, atol=0)


def test_actual_teacher_fft_sequence_backward_updates_recurrence():
    from vlade.models.teachers.ava_vla import OpenVLAForActionPrediction
    from test_ava_teacher import tiny_components

    torch.set_num_threads(2)
    torch.manual_seed(2)
    bundle = tiny_components(OpenVLAForActionPrediction).to(torch.bfloat16)
    for module in bundle.values():
        module.train()
    frames = []
    for step in range(4):
        frames.append(
            dict(
                input_ids=torch.tensor([[1, 10, 29871] + [1] * 56 + [2]]),
                attention_mask=torch.ones(1, 60, dtype=torch.long),
                pixel_values=torch.randn(1, 8, 16),
                labels=torch.tensor([[-100] * 3 + [31744] * 56 + [2]]),
                proprio=torch.randn(1, 8),
                actions=torch.zeros(1, 8, 7),
                action_valid_mask=torch.ones(1, 8, 7, dtype=torch.bool),
                sample_metadata=[dict(episode_id="one", timestep=8 * step)],
            )
        )
    cfg = SimpleNamespace(sequence_length=4, detach_every=2, temporal_feature_layer=-2)
    states = []
    hook = bundle["generator"].register_forward_pre_hook(lambda m, args: states.append(args[1]))
    result = fft_objective(
        cfg,
        tuple(bundle[k] for k in ("model", "head", "proprio", "generator")),
        {"sequences": [frames]},
        torch.device("cpu"),
        9,
        0,
        backward_scale=1.0,
    )
    hook.remove()
    assert torch.isfinite(result) and not result.requires_grad
    assert states[0] is None and all(s is not None for s in states[1:])
    assert states[1].requires_grad and not states[2].requires_grad
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in bundle["generator"].action2embed_input.parameters()
    )
    bad = copy.deepcopy(frames)
    bad[2]["sample_metadata"][0]["episode_id"] = "other"
    with pytest.raises(ValueError, match="crosses an episode"):
        fft_objective(
            cfg,
            tuple(bundle[k] for k in ("model", "head", "proprio", "generator")),
            {"sequences": [bad]},
            torch.device("cpu"),
            9,
            0,
        )


def test_prepared_loader_transforms_complete_windows_and_resumes(monkeypatch, tmp_path):
    import numpy as np
    from vlade.data import episode_data
    from vlade.training.prepared_fft import make_loader

    class Prepared(Data):
        stats = {"action": {}, "proprio": {}}

        def episodes(self, split, *, order):
            for entry, samples in super().episodes(split, order=order):
                for sample in samples:
                    sample.update(
                        gt_action=np.zeros((8, 7), dtype=np.float32),
                        proprio=np.zeros(8, dtype=np.float32),
                        action_valid_mask=np.ones((8, 7), dtype=bool),
                        primary="p",
                        wrist="w",
                    )
                    sample["action_valid_mask"][-1] = False
                yield entry, samples

        def image(self, sample, camera):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    class Tokenizer:
        vocab_size = 32000
        model_max_length = 128
        pad_token_id = 0

        def decode(self, ids):
            return "x" * len(ids)

        def batch_decode(self, rows):
            return [self.decode(row) for row in rows]

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1, 10, 29871] + [31744] * 56 + [2])

    monkeypatch.setattr(episode_data, "EpisodeData", lambda root: Prepared())
    processor = SimpleNamespace(
        tokenizer=Tokenizer(),
        image_processor=SimpleNamespace(apply_transform=lambda im: torch.zeros(4, 4, 4)),
    )
    cfg = SimpleNamespace(
        sequence_length=4,
        sequence_sampling="episode_uniform",
        seed=7,
        image_aug=False,
        prepared_data_root=tmp_path,
        dataset_name="test",
        batch_size=2,
        grad_accumulation_steps=2,
        resume=None,
        max_eval_batches=2,
    )
    loader, data = make_loader(cfg, processor, train=True)
    batches = list(islice(loader, 4))
    assert len(batches[0]["sequences"]) == 2
    for frames in batches[0]["sequences"]:
        assert len(frames) == 4
        assert frames[0]["actions"].shape == (1, 8, 7)
        assert frames[0]["pixel_values"].shape == (1, 8, 4, 4)
        assert not frames[0]["action_valid_mask"][0, -1].any()
    assert data.sampling_statistics["excluded_short_episodes"] == 1
    (tmp_path / "vlade_fft.json").write_text('{"step":1}', encoding="utf-8")
    cfg.resume = tmp_path
    resumed, _ = make_loader(cfg, processor, train=True)

    def ids(batch):
        return [[f["sample_metadata"] for f in seq] for seq in batch["sequences"]]

    assert ids(next(iter(resumed))) == ids(batches[2])
    validation, _ = make_loader(cfg, processor, train=False)
    seen = [
        f["sample_metadata"][0]["episode_id"]
        for b in validation
        for seq in b["sequences"]
        for f in seq
    ]
    assert seen and set(seen) == {"val"}


def test_research_one_step_override_wins_over_base_k4(monkeypatch, tmp_path):
    from pathlib import Path
    from vlade.research.execution import dispatch
    from vlade.training import fft
    import json, os

    root = Path(os.environ["VLADE_PROJECT_ROOT"])
    received = []
    monkeypatch.setattr(fft, "train", lambda cfg: received.append(cfg))
    for profile, length in [("ava_fft", 4), ("ava_one_step", 1)]:
        config = json.loads((root / f"configs/research/{profile}.json").read_text())
        config["reload_check"] = False
        dispatch(config, root, tmp_path, tmp_path / profile)
        assert received[-1].sequence_length == length
    assert (
        fft.FFTConfig.from_json(
            root / "configs/experiments/libero_plus_fft.json", "openvla_oft"
        ).sequence_length
        == 1
    )


def test_existing_sealed_k1_checkpoint_cannot_resume_as_k4(monkeypatch, tmp_path):
    from test_fft import test_full_checkpoint_saves_model_and_all_components
    from vlade.training.fft import FFTConfig, validate_resume
    test_full_checkpoint_saves_model_and_all_components(monkeypatch, tmp_path)
    cfg = FFTConfig("ava_vla", tmp_path / "source", tmp_path, tmp_path / "out",
        save_optimizer_state=True, optimizer="adamw", sequence_length=4,
        resume=tmp_path / "out" / "step-0000001")
    with pytest.raises(ValueError, match="temporal configuration mismatch: sequence_length"):
        validate_resume(cfg)
