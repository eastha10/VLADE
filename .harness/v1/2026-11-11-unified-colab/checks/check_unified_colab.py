"""Exercise the unified notebook against real configs without installing or training."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
assert (ROOT / "src/vlade/research/cli.py").is_file()
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
sys.dont_write_bytecode = True

from vlade.research.cli import main
from vlade.utils.colab_notebooks import PROFILES, NOTEBOOK_NAME, write_unified_notebook

NOTEBOOK = json.loads((ROOT / "experiments" / NOTEBOOK_NAME).read_text(encoding="utf-8"))
CELLS = {c["id"]: "".join(c["source"]) for c in NOTEBOOK["cells"]}
SCRATCH = ROOT / ".harness/v1/2026-11-11-unified-colab/scratch"


class UnifiedColabChecks(unittest.TestCase):
    def controls(self):
        namespace = {"CODE": ROOT}
        exec(CELLS["config"], namespace)
        return namespace

    def plan(self, namespace):
        with redirect_stdout(io.StringIO()):
            exec(CELLS["plan"], namespace)

    def test_single_notebook_schema_and_empty_outputs(self):
        self.assertEqual([p.name for p in (ROOT / "experiments").glob("*.ipynb")], [NOTEBOOK_NAME])
        self.assertEqual(NOTEBOOK["nbformat"], 4)
        self.assertEqual(len(CELLS), len(NOTEBOOK["cells"]))
        for cell in NOTEBOOK["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        self.assertEqual(set(PROFILES), {p.stem for p in (ROOT / "configs/research").glob("*.json")})

    def test_run_all_defaults_only_plans(self):
        with patch("vlade.research.client.job") as job:
            namespace = self.controls()
            self.plan(namespace)
            with redirect_stdout(io.StringIO()):
                exec(CELLS["setup"], namespace)
                exec(CELLS["run"], namespace)
        self.assertFalse(namespace["RUN_SETUP"])
        self.assertFalse(namespace["RUN_EXPERIMENT"])
        self.assertEqual([call.kwargs["action"] for call in job.call_args_list], ["plan"])

    def test_all_18_profiles_reach_real_cli_and_keep_outputs_distinct(self):
        namespace = self.controls()
        work = SCRATCH / "never-created"
        namespace["WORKSPACE"] = str(work)
        outputs, groups = set(), set()

        def plan(config, *, action, workspace, overrides):
            self.assertEqual(action, "plan")
            args = ["--config", str(config), "--workspace", workspace, "--action", action]
            for key, value in overrides.items():
                args += ["--set", key + "=" + json.dumps(value)]
            stream = io.StringIO()
            with redirect_stdout(stream):
                self.assertEqual(main(args), 0)
            actual = json.loads(stream.getvalue())
            self.assertEqual(actual["execution"], "not_started")
            self.assertTrue(actual["config"]["run_group"].endswith("-" + namespace["PROFILE"]))
            groups.add(actual["config"]["run_group"])
            outputs.add(actual["output"])

        with patch("vlade.research.client.job", side_effect=plan):
            namespace["job"] = plan
            for profile in PROFILES:
                namespace["PROFILE"] = profile
                self.plan(namespace)
        self.assertEqual(len(outputs), 18)
        self.assertEqual(len(groups), 18)
        self.assertFalse(work.exists())

    def test_suite_phase_uses_same_group_and_dispatches_only_selected_job(self):
        with patch("vlade.research.client.job") as job:
            namespace = self.controls()
            namespace["PROFILE"] = "ava_suite"
            self.plan(namespace)
            pilot_group = namespace["RUN_GROUP"]
            namespace["OVERRIDES"] = {
                "settings.phase": "main", "train.max_steps": 1000,
                "train.warmup_steps": 100, "train.save_steps": 250,
            }
            self.plan(namespace)
            self.assertEqual(namespace["RUN_GROUP"], pilot_group)
            namespace["RUN_SETUP"] = namespace["RUN_EXPERIMENT"] = True
            exec(CELLS["setup"], namespace)
            exec(CELLS["run"], namespace)
            last = job.call_args
            self.assertEqual(Path(last.args[0]).name, "ava_suite.json")
            self.assertEqual(last.kwargs["action"], "run")
            self.assertEqual(last.kwargs["overrides"]["settings.phase"], "main")
            self.assertEqual(last.kwargs["overrides"]["run_group"], pilot_group)

    def test_changed_selection_or_nested_overrides_require_new_plan(self):
        with patch("vlade.research.client.job"):
            for key, value in [
                ("PROFILE", "qwen_kd"), ("WORKSPACE", "changed"),
                ("RUN_GROUP_OVERRIDE", "other-run"), ("EXPERIMENT_ID", "new-experiment"),
            ]:
                namespace = self.controls()
                self.plan(namespace)
                namespace[key] = value
                namespace["RUN_EXPERIMENT"] = True
                with self.assertRaisesRegex(RuntimeError, "계획"):
                    exec(CELLS["run"], namespace)
            namespace = self.controls()
            namespace["OVERRIDES"] = {"arguments": ["first"]}
            self.plan(namespace)
            namespace["OVERRIDES"]["arguments"].append("changed")
            namespace["RUN_SETUP"] = True
            with self.assertRaisesRegex(RuntimeError, "계획"):
                exec(CELLS["setup"], namespace)

    def test_explicit_resume_group_and_config_rerun_preserve_identity(self):
        with patch("vlade.research.client.job"):
            namespace = self.controls()
            experiment = namespace["EXPERIMENT_ID"]
            exec(CELLS["config"], namespace)
            self.assertEqual(namespace["EXPERIMENT_ID"], experiment)
            namespace["RUN_GROUP_OVERRIDE"] = "original-k4-run"
            namespace["OVERRIDES"] = {"resume": "/restored/step-0000002"}
            self.plan(namespace)
            self.assertEqual(namespace["RUN_GROUP"], "original-k4-run")
            self.assertEqual(namespace["_PLANNED"][2]["resume"], "/restored/step-0000002")

    def test_generation_is_deterministic_and_compatibility_does_not_restore_old_files(self):
        from vlade.utils.kd_notebook import generate_kd_notebook
        from vlade.utils.paper_notebook import generate_paper_notebook
        from vlade.utils.hf_notebooks import apply_hf_storage, refresh_suite_sources, refresh_inference_sources
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            root = Path(directory)
            path = write_unified_notebook(root / "experiments" / NOTEBOOK_NAME)
            original = path.read_bytes()
            write_unified_notebook(path)
            self.assertEqual(original, path.read_bytes())
            generate_kd_notebook(root, "unused")
            generate_paper_notebook(root, "unused")
            old = root / "experiments/colab_02_ava_fft_and_fixed_cache.ipynb"
            apply_hf_storage(old, root)
            refresh_suite_sources(old, root)
            refresh_inference_sources(old)
            self.assertEqual(list((root / "experiments").glob("*.ipynb")), [path])
            self.assertEqual(original, path.read_bytes())


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    unittest.main(verbosity=2)
