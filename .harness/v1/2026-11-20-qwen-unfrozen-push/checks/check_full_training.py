"""GPU smoke test for native QwenVLA cached KD using tiny, synthetic inputs."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch


def main(project: Path, scratch: Path) -> None:
    sys.path.insert(0, str(project / ".harness/v1/2026-11-11-ava-teacher-code/checks"))
    sys.path.insert(0, str(project / "src"))
    from qwen_vla_smoke import tiny_inputs, tiny_student
    from vlade.training import qwen_vla_task as kd
    from vlade.training.run_control import seal_checkpoint, verify_checkpoint

    torch.manual_seed(42)
    torch.set_num_threads(2)
    scratch.mkdir(parents=True)
    data_root = scratch / "data"
    data_root.mkdir()
    (data_root / "manifest.json").write_text("{}", encoding="utf-8")
    data_sha = kd.sha256(data_root / "manifest.json")
    initial = scratch / "step-0000800"
    model = tiny_student()
    model.save(initial / "model")

    class Processor:
        def save_pretrained(self, path):
            path.mkdir()
            (path / "processor_config.json").write_text("{}", encoding="utf-8")

    Processor().save_pretrained(initial / "processor")
    (initial / "identity.json").write_text(json.dumps({"data_sha256": data_sha}), encoding="utf-8")
    seal_checkpoint(initial, 800, 6400)
    (initial / "COMPLETE").touch()
    assert kd.verified_s0(initial, model.student_config, data_sha) == 800
    try:
        kd.verified_s0(initial, model.student_config, "different")
    except ValueError as error:
        assert "same prepared data" in str(error)
    else:
        raise AssertionError("Accepted incompatible S0 data")

    config_path = scratch / "tiny.json"
    config_path.write_text(json.dumps({"student": asdict(model.student_config)}), encoding="utf-8")
    inputs = tiny_inputs()
    teacher_action = torch.full((8, 7), 0.2)
    teacher_action[-1].zero_()
    mask = np.ones((8, 7), dtype=bool)
    mask[-1] = False
    calls = []
    worker_events = []
    teacher_signature = {"checkpoint_sha256": "synthetic"}

    class Data:
        def __init__(self, root):
            assert root == data_root
            self.root = root
            self.manifest = {"normalization_sha256": "synthetic"}

    class Teacher:
        def __init__(self, root, data, *, episode_limit):
            assert root == scratch / "cache" and episode_limit == 1
            self.signature = teacher_signature

        def get(self, sample):
            assert sample["episode_id"] == "train"
            calls.append(sample["timestep"])
            target = teacher_action.clone()
            target[-1] = 1000
            return target

    class OnlineTeacher(Teacher):
        def __init__(self, settings, data, *, timings=None):
            assert settings["teacher_checkpoint"] == "hfasset://T1" and data.root == data_root
            self.signature = teacher_signature
            worker_events.append("opened")

        def close(self):
            worker_events.append("closed")

    def stream(data, processor, config, split, seed, **kwargs):
        if split == "train":
            assert kwargs["episode_limit"] in {0, 1}
        rows = [{"episode_id": split, "timestep": i * 8, "proprio": np.ones(8, dtype=np.float32),
                 "gt_action": np.zeros((8, 7), dtype=np.float32), "action_valid_mask": mask}
                for i in range(2)]
        for i in range(kwargs.get("skip", 0), 2):
            yield rows[i], inputs, rows[i - 1] if i else None

    output = scratch / "output"
    settings = {
        "trainable_scope": "full", "max_steps": 2, "accumulation_steps": 1, "save_steps": 1, "validation_queries": 1,
        "initial_checkpoint": "hfasset://student_s0_800", "data": "hfasset://data",
        "model_config": str(config_path), "seed": 42, "learning_rate": 0.001, "weight_decay": 0.0,
        "distillation": {"teacher_cache": "hfasset://teacher_cache_t1", "episode_limit": 1,
                         "task_weight": 1.0, "action_kd_weight": 0.5},
    }
    paths = {"hfasset://data": data_root, "hfasset://teacher_cache_t1": scratch / "cache"}
    paths[initial] = initial  # explicit verified fixture resolver, never a production bypass
    local_input = {"local_s0": initial, "local_s0_manifest_sha256": kd.sha256(initial / "checkpoint_manifest.json")}
    control = {key: value for key, value in settings.items() if key != "distillation"}
    control["initialization"] = "student_checkpoint"
    online = {
        **control,
        "distillation": {
            "teacher_mode": "inference", "teacher_checkpoint": "hfasset://T1",
            "teacher_python": sys.executable, "task_weight": 1.0, "action_kd_weight": 0.5,
        },
    }
    online_output = scratch / "online"
    control_output = scratch / "control"
    with (patch.object(kd, "source_identity", return_value={"git_commit": "synthetic"}),
          patch.object(kd, "resolve_input", side_effect=paths.__getitem__),
          patch.object(kd, "EpisodeData", Data), patch.object(kd, "CachedActions", Teacher),
          patch.object(kd, "InferenceActions", OnlineTeacher),
          patch.object(kd, "teacher_identity", return_value=(scratch / "t1", teacher_signature)),
          patch.object(kd, "batches", stream), patch.object(kd, "load_saved_processor", return_value=Processor()),
          patch.dict("os.environ", {"VLADE_HUB_REPO": ""})):
        kd.run(settings, output, stop_after=1, **local_input)
        first = output / "step-0000001"
        assert verify_checkpoint(first)["microsteps"] == 1
        try:
            kd.run({**settings, "trainable_scope": "policy"}, output, resume=first, **local_input)
        except ValueError as error:
            assert "identity differs" in str(error)
        else:
            raise AssertionError("Accepted a resume with a different trainable scope")
        try:
            kd.run(settings, output, resume=first, local_s0=initial, local_s0_manifest_sha256="0" * 64)
        except ValueError as error:
            assert "pinned SHA256" in str(error)
        else:
            raise AssertionError("Accepted unpinned local S0")
        kd.run(settings, output, resume=first, **local_input)
        assert calls == [0, 8] and not worker_events
        calls.clear()
        kd.run(online, online_output, stop_after=1, **local_input)
        online_first = online_output / "step-0000001"
        kd.run(online, online_output, resume=online_first, **local_input)
        assert calls == [0, 8] and worker_events == ["opened", "closed", "opened", "closed"]
        kd.run(control, control_output, stop_after=1, **local_input)
        assert calls == [0, 8] and worker_events == ["opened", "closed", "opened", "closed"]
    control_first = control_output / "step-0000001"
    control_metrics = json.loads((control_first / "metrics.json").read_text(encoding="utf-8"))
    assert control_metrics["task_l1"] == control_metrics["loss"] and "action_kd_l1" not in control_metrics
    online_identity = json.loads((online_first / "identity.json").read_text(encoding="utf-8"))
    control_identity = json.loads((control_first / "identity.json").read_text(encoding="utf-8"))
    assert online_identity["teacher_inference"] == teacher_signature and "teacher_inference" not in control_identity
    for key in ("initial_sha256", "initial_student_step", "data_sha256"):
        assert online_identity[key] == control_identity[key]
    second = output / "step-0000002"
    assert verify_checkpoint(second)["microsteps"] == 2
    online_second = online_output / "step-0000002"
    assert verify_checkpoint(online_second)["microsteps"] == 2
    for checkpoint in (first, second, online_first, online_second):
        metrics = json.loads((checkpoint / "metrics.json").read_text(encoding="utf-8"))
        assert abs(metrics["loss"] - metrics["task_l1"] - 0.5 * metrics["action_kd_l1"]) < 1e-6
        assert metrics["action_kd_l1"] < 5 and metrics["validation_l1"] >= 0
    for checkpoint in (second, online_second, control_first):
        loaded = kd.QwenVLA.load(checkpoint / "model")
        assert any(not torch.equal(initial_weight, loaded.backbone.state_dict()[name])
                   for name, initial_weight in model.backbone.state_dict().items())
        for name in ("ava", "proprio_projector", "action_head"):
            trained = dict(getattr(loaded, name).named_parameters())
            assert any(not torch.equal(parameter, trained[key]) for key, parameter in getattr(model, name).named_parameters())
    import csv
    timing_paths = list(scratch.rglob("timings-s0-*.csv"))
    assert len(timing_paths) == 5
    for timing_path in timing_paths:
        rows = list(csv.DictReader(timing_path.open()))
        assert any(row["event"] == "s0_training_step" for row in rows)
        assert any(row["event"] == "s0_validation" for row in rows)
        assert any(row["event"] == "s0_checkpoint_save" for row in rows)
    print("Passed: cached KD, simulated online targets, matched GT-only, saved S0 checks and checkpoint resume")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
