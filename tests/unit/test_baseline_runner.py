"""외부 baseline 실행 설정의 단위 테스트입니다."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts" / "evaluate" / "run_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_baseline", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class BaselineRunnerTest(unittest.TestCase):
    def test_registered_baselines_are_discoverable(self) -> None:
        self.assertEqual(RUNNER.available_baselines(), ("ava_vla", "openvla_oft"))

    def test_ava_vla_invocation_contains_temporal_arguments(self) -> None:
        config = RUNNER.load_baseline_config("ava_vla")
        invocation = RUNNER.build_invocation(config, task_suite="libero_10")

        self.assertIn("--multi_frame.temporal_strategy", invocation.command)
        self.assertIn("attn_weight", invocation.command)
        self.assertEqual(invocation.checkpoint_source, "local")
        self.assertTrue(str(invocation.local_checkpoint).endswith("avavla-libero-4in1"))

    def test_openvla_oft_uses_suite_specific_checkpoint(self) -> None:
        config = RUNNER.load_baseline_config("openvla_oft")
        invocation = RUNNER.build_invocation(config, task_suite="libero_object", use_hub_checkpoint=True)

        checkpoint_index = invocation.command.index("--pretrained_checkpoint") + 1
        self.assertEqual(
            invocation.command[checkpoint_index],
            "moojink/openvla-7b-oft-finetuned-libero-object",
        )
        self.assertNotIn("--multi_frame.temporal_strategy", invocation.command)

    def test_rejects_unknown_task_suite(self) -> None:
        config = RUNNER.load_baseline_config("ava_vla")
        with self.assertRaises(RUNNER.BaselineConfigurationError):
            RUNNER.build_invocation(config, task_suite="unknown")

    def test_main_experiment_is_libero_plus_full_fine_tuning(self) -> None:
        config_path = REPO_ROOT / "configs" / "experiments" / "libero_plus_fft.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["benchmark"]["name"], "libero_plus")
        self.assertEqual(config["training"]["method"], "full_fine_tuning")
        for baseline_path in config["baseline_inputs"]:
            self.assertTrue((REPO_ROOT / baseline_path).is_file())


if __name__ == "__main__":
    unittest.main()
