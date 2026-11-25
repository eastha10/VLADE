"""Offline checks of notebook wiring; no model download or GPU execution."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
assert ROOT.name == 'vlade' and (ROOT / 'AGENTS.md').is_file()
NOTEBOOK = ROOT / 'experiments/colab_cot_action_kd_ready.ipynb'
CELLS = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
SOURCES = [''.join(c['source']) for c in CELLS]
SCRATCH = Path(__file__).resolve().parents[1] / 'scratch'


class NotebookChecks(unittest.TestCase):
    def run_cell(self, index, env):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(SOURCES[index], f'cell-{index}', 'exec'), env)

    def config(self, replacement=None):
        source = SOURCES[1]
        if replacement:
            source = source.replace(*replacement)
        env = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, env)
        return env

    def test_syntax_and_clean_outputs(self):
        ids = []
        for i, cell in enumerate(CELLS):
            ids.append(cell['id'])
            if cell['cell_type'] == 'code':
                compile(SOURCES[i], f'cell-{i}', 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
        self.assertEqual(len(ids), len(set(ids)))
        env = {}
        self.run_cell(13, env)
        compile(env['AUDIT_SOURCE'], 'checkpoint-audit', 'exec')

    def test_default_config_and_pinned_load_settings(self):
        env = self.config()
        self.assertFalse(env['RUN_TRAIN'])
        self.assertFalse(env['RUN_PILOT'])
        self.assertEqual(env['OUTPUT'].name, 'S2')
        tree = ast.parse(subprocess.check_output(['git', 'show', env['SOURCE_COMMIT'] + ':src/vlade/research/distillation.py'], cwd=ROOT, text=True, encoding='utf-8'))
        load = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_settings')
        budget = ast.parse(subprocess.check_output(['git', 'show', env['SOURCE_COMMIT'] + ':src/vlade/distillation/teacher_plan.py'], cwd=ROOT, text=True, encoding='utf-8'))
        # Use the pinned implementation and only its token-budget function dependencies.
        scope = {'Path': Path, 'ROOT': ROOT, 'json': json, 'math': env['math'], 're': re}
        function = next(n for n in budget.body if isinstance(n, ast.FunctionDef) and n.name == 'plan_token_budget')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])), 'pinned-budget', 'exec'), scope)
        exec(compile(ast.Module(body=[load], type_ignores=[]), 'pinned-settings', 'exec'), scope)
        workflow, selected = scope['load_settings'](ROOT / env['CONFIG'], env['OVERRIDES'])
        self.assertEqual(list(selected), ['gt_kd'])
        self.assertEqual(selected['gt_kd']['distillation']['teacher_plan_max_new_tokens'], 48)
        self.assertEqual(selected['gt_kd']['initial_checkpoint'], 'hfasset://S0')
        self.assertRegex(workflow['registry_revision'], r'^[0-9a-f]{40}$')

    def test_teacher_pin_uses_separate_key(self):
        pin = {'repo_id': 'VLADE-konkuk/VLADE-checkpoints', 'revision': 'a' * 40, 'prefix': 'runs/teacher/model/checkpoints/step-0000100'}
        env = self.config(('TEACHER_CHECKPOINT_PIN = {}', 'TEACHER_CHECKPOINT_PIN = ' + repr(pin)))
        self.assertEqual(env['OVERRIDES']['distillation.teacher_checkpoint'], 'hfasset://T1_plan')
        self.assertEqual(set(env['OVERRIDES']['workflow.checkpoint_pins']), {'T1_plan'})
        with self.assertRaises(AssertionError):
            self.config(('PLAN_MAX_NEW_TOKENS = 48', 'PLAN_MAX_NEW_TOKENS = 0'))

    def test_s1_resume_rejected(self):
        with self.assertRaises(AssertionError):
            self.config(('RESUME_REVISIONS = {}', "RESUME_REVISIONS = {'S1': '" + 'a' * 40 + "'}"))

    def test_probe_accepts_both_splits_and_rejects_bad_rows(self):
        good = [{'status': 'ok', 'split': 'train', 'plan': 'Move toward the handle.', 'action_shape': [8, 7]},
                {'status': 'ok', 'split': 'validation', 'plan': 'Open the gripper.', 'action_shape': None},
                {'status': 'complete', 'queries': 2}]
        bad = copy.deepcopy(good)
        bad[0]['plan'] = ''
        for rows, accepted in [(good, True), (bad, False), (good[:-1], False)]:
            with tempfile.TemporaryDirectory(dir=SCRATCH) as temporary:
                path = Path(temporary)
                def run_action(action):
                    self.assertEqual(action, 'probe')
                    (path / 'teacher-plan-probe-test.jsonl').write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
                env = {'PREFLIGHT_OK': True, 'RUN_ROOT': path, 'json': json, 'run_action': run_action}
                if accepted:
                    self.run_cell(11, env)
                    self.assertTrue(env['PROBE_OK'])
                else:
                    with self.assertRaises(RuntimeError):
                        self.run_cell(11, env)
                    self.assertFalse(env['PROBE_OK'])

    def test_training_requires_probe(self):
        for index, flag in [(15, 'RUN_PILOT'), (17, 'RUN_TRAIN')]:
            env = {flag: True, 'PREFLIGHT_OK': True, 'PROBE_OK': False}
            with self.assertRaises(AssertionError):
                self.run_cell(index, env)

    def test_failed_preflight_clears_previous_success(self):
        function = next(n for n in ast.parse(SOURCES[5]).body if isinstance(n, ast.FunctionDef))
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temporary:
            env = self.config()
            env['WORKSPACE'] = Path(temporary)
            env['RUN_ROOT'] = Path(temporary)
            env['PREFLIGHT_OK'] = env['PROBE_OK'] = True
            contract = {'commit': env['SOURCE_COMMIT'], 'group': env['RUN_GROUP'], 'workspace': str(env['WORKSPACE']), 'config': env['CONFIG'], 'overrides': env['OVERRIDES']}
            env['_CONTRACT'] = json.dumps(contract, sort_keys=True)
            env['CODE'] = ROOT
            env['contextlib'] = contextlib
            env['Tee'] = lambda *streams: streams[0]
            def failed_job(*args, **kwargs):
                raise RuntimeError('simulated failure')
            env['job'] = failed_job
            exec(compile(ast.Module(body=[function], type_ignores=[]), 'run-action', 'exec'), env)
            with self.assertRaises(RuntimeError):
                env['run_action']('preflight')
            self.assertFalse(env['PREFLIGHT_OK'])
            self.assertFalse(env['PROBE_OK'])

    def test_report_bundle_includes_plan_logs(self):
        self.assertIn('teacher-plans-*.jsonl', SOURCES[21])
        self.assertIn('teacher-plan-probe-*.jsonl', SOURCES[21])
        self.assertNotIn('/S1', ''.join(SOURCES))


if __name__ == '__main__':
    SCRATCH.mkdir(exist_ok=True)
    unittest.main()
