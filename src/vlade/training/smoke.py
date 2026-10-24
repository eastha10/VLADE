"""Small CPU action-policy smoke check; does not load a pretrained VLA."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from vlade.training.loop import make_scheduler, restore_training_state, run_loop, training_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    batches = [torch.randn(2, 8) for _ in range(8)]

    def create():
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 56))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, optimizer, make_scheduler(optimizer, 4, 1, "cosine")

    def loss_fn(model):
        return lambda batch, step: model(batch).reshape(-1, 8, 7).abs().mean()

    model, optimizer, scheduler = create()
    records = []
    checkpoint = args.output_dir / "synthetic-step-2.pt"

    def on_step(state, metrics):
        records.append({**asdict(state), **metrics})
        if state.step == 2:
            torch.save({"model": model.state_dict(),
                        "trainer": training_state(state, optimizer, scheduler)}, checkpoint)

    run_loop(modules=[model], optimizer=optimizer, scheduler=scheduler, batches=batches,
             loss_fn=loss_fn(model), max_steps=4, accumulation_steps=2, on_step=on_step)
    restored, new_optimizer, new_scheduler = create()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(payload["model"])
    state = restore_training_state(payload["trainer"], new_optimizer, new_scheduler)
    run_loop(modules=[restored], optimizer=new_optimizer, scheduler=new_scheduler,
             batches=batches[4:], loss_fn=loss_fn(restored), max_steps=4,
             accumulation_steps=2, state=state)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    report = {"status": "passed", "test_type": "synthetic_cpu_not_pretrained_vla",
              "torch": torch.__version__, "steps": records,
              "save_load_resume_exact_with_fixed_data": True}
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0
