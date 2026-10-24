"""Real CPU autograd tests for optimizer boundaries and continuation."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vlade.training.loop import (  # noqa: E402
    LoopState, capture_rng, evaluate, make_scheduler, restore_rng,
    restore_training_state, run_loop, training_state,
)


def setup_model():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = make_scheduler(optimizer, 4, 1, "cosine")
    return model, optimizer, scheduler


def loss_for(model):
    return lambda batch, step: model(batch).square().mean()


def test_accumulation_matches_one_combined_batch():
    small, opt_a, sched_a = setup_model()
    large, opt_b, sched_b = setup_model()
    batches = [torch.tensor([[1., 2.]]), torch.tensor([[3., 4.]])]
    records = []
    result = run_loop(modules=[small], optimizer=opt_a, scheduler=sched_a,
                      batches=batches, loss_fn=loss_for(small), max_steps=1,
                      accumulation_steps=2, on_step=lambda state, metrics: records.append(metrics))
    run_loop(modules=[large], optimizer=opt_b, scheduler=sched_b,
             batches=[torch.cat(batches)], loss_fn=loss_for(large), max_steps=1)
    torch.testing.assert_close(small.weight, large.weight)
    assert result == LoopState(step=1, microsteps=2)
    assert records[0]["loss"] == pytest.approx((0.6**2 + 1.4**2) / 2)
    assert records[0]["grad_norm"] > 1.0


def test_clipping_limits_the_actual_optimizer_update():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(10.)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.)
    run_loop(modules=[model], optimizer=optimizer, scheduler=make_scheduler(optimizer, 1),
             batches=[torch.ones(1, 1)], loss_fn=loss_for(model), max_steps=1, max_grad_norm=0.5)
    assert model.weight.item() == pytest.approx(9.5)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_loss_does_not_update_or_checkpoint(bad_value):
    model, optimizer, scheduler = setup_model()
    before = model.weight.detach().clone()
    with pytest.raises(FloatingPointError):
        run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler,
                 batches=[None], loss_fn=lambda *_: model.weight.sum() * bad_value, max_steps=1,
                 on_step=lambda *_: pytest.fail("Must not checkpoint a failed step"))
    torch.testing.assert_close(model.weight, before)
    assert model.weight.grad is None


def test_finite_loss_with_infinite_gradient_is_rejected():
    model, optimizer, scheduler = setup_model()
    before = model.weight.detach().clone()

    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.sum() * 0

        @staticmethod
        def backward(ctx, gradient):
            return torch.full((1, 2), float("inf"))

    with pytest.raises(RuntimeError, match="non-finite"):
        run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler, batches=[None],
                 loss_fn=lambda *_: BadBackward.apply(model.weight), max_steps=1)
    torch.testing.assert_close(model.weight, before)


def test_truncated_accumulation_window_is_discarded():
    model, optimizer, scheduler = setup_model()
    before = model.weight.detach().clone()
    state = LoopState()
    with pytest.raises(RuntimeError, match="stream ended"):
        run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler,
                 batches=[torch.ones(1, 2)], loss_fn=loss_for(model), max_steps=1,
                 accumulation_steps=2, state=state)
    torch.testing.assert_close(model.weight, before)
    assert model.weight.grad is None and state.step == 0


def test_validation_restores_modes_and_rng_even_on_failure():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[0].eval()
    before = capture_rng()

    def fail(*_):
        assert not model.training
        assert not torch.is_grad_enabled()
        torch.rand(2)
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        evaluate([model], [None], fail, 1, 1)
    assert model.training and not model[0].training and model[1].training
    torch.testing.assert_close(torch.get_rng_state(), before["torch"])


def test_validation_is_bounded_and_averaged():
    model, _, _ = setup_model()
    batches = [torch.ones(1, 2), torch.ones(1, 2) * 2, torch.ones(1, 2) * 100]
    result = evaluate([model], batches, loss_for(model), 0, 2)
    assert result == pytest.approx((0.4**2 + 0.8**2) / 2)


def test_resume_matches_uninterrupted_with_same_remaining_data(tmp_path):
    """This controls data order, unlike the live shuffled RLDS stream."""
    torch.manual_seed(17)
    batches = [torch.ones(1, 2) * i for i in range(1, 9)]
    model, optimizer, scheduler = setup_model()
    initial_rng = capture_rng()

    def random_loss(module):
        return lambda batch, step: (module(batch) * torch.rand(())).square().mean()

    snapshot = {}

    def checkpoint(state, metrics):
        if state.step == 2:
            snapshot["weights"] = copy.deepcopy(model.state_dict())
            snapshot["trainer"] = copy.deepcopy(training_state(state, optimizer, scheduler))

    run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler, batches=batches,
             loss_fn=random_loss(model), max_steps=4, accumulation_steps=2, on_step=checkpoint)
    restored, new_optimizer, new_scheduler = setup_model()
    restored.load_state_dict(snapshot["weights"])
    path = tmp_path / "trainer.pt"
    torch.save(snapshot["trainer"], path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = restore_training_state(payload, new_optimizer, new_scheduler)
    run_loop(modules=[restored], optimizer=new_optimizer, scheduler=new_scheduler, batches=batches[4:],
             loss_fn=random_loss(restored), max_steps=4, accumulation_steps=2, state=state)
    torch.testing.assert_close(restored.weight, model.weight, rtol=0, atol=0)
    assert new_scheduler.state_dict() == scheduler.state_dict()
    assert state == LoopState(4, 8)
    restore_rng(initial_rng)


def test_rng_round_trip_covers_numpy_and_torch():
    before = capture_rng()
    expected_np, expected_torch = np.random.rand(), torch.rand(2)
    restore_rng(before)
    assert np.random.rand() == expected_np
    torch.testing.assert_close(torch.rand(2), expected_torch)


def test_finished_resume_does_not_consume_any_data():
    model, optimizer, scheduler = setup_model()

    def forbidden():
        raise AssertionError("finished run must not read data")
        yield

    run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler, batches=forbidden(),
             loss_fn=loss_for(model), max_steps=4, state=LoopState(4, 4))


def test_warmup_and_cosine_change_only_at_optimizer_steps():
    model, optimizer, _ = setup_model()
    scheduler = make_scheduler(optimizer, 4, 2, "cosine")
    lrs = []
    run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler,
             batches=[torch.ones(1, 2)] * 8, loss_fn=loss_for(model), max_steps=4,
             accumulation_steps=2, on_step=lambda _, metrics: lrs.append(metrics["lr"]))
    assert lrs == pytest.approx([0.005, 0.01, 0.01, 0.005])


def test_invalid_resume_boundary_fails():
    model, optimizer, scheduler = setup_model()
    with pytest.raises(ValueError, match="boundary"):
        run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler, batches=[],
                 loss_fn=loss_for(model), max_steps=4, accumulation_steps=2,
                 state=LoopState(1, 1))
