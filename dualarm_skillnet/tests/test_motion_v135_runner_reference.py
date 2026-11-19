"""The native entry must consume the verified reference and preserve its status."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from skillnet.capability.oracle import Status
from skillnet.motion.base import MotionResult
from simbridge import runner
from simbridge.backend import CoppeliaBackend
from tests.test_v13_compiler import fixture

ROOT = Path(__file__).resolve().parents[1]


class RunnerReferenceTests(unittest.TestCase):
    def invoke(self, backend_type, check):
        engine, geometry, compiler = fixture('s1_parallel_place')
        engine.problem.proposal = None
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as folder:
            owner = MagicMock()
            owner.connect.return_value = (object(), object())
            owner.process.pid = 12345
            owner.process.poll.return_value = 0
            owner.log_path = Path(folder)/'native.log'
            with (patch.object(runner, 'load_problem', return_value=engine.problem),
                  patch.object(runner, 'check_environment', return_value={}),
                  patch.object(runner, 'Simulator', return_value=owner),
                  patch.object(runner, 'build', return_value=deepcopy(compiler.cfg)),
                  patch.object(runner, 'MotionPlanner') as planner,
                  patch.object(runner, 'CoppeliaBackend', backend_type),
                  patch('simbridge.executor.execute_plan', return_value={'passed': True}) as execute):
                planner.return_value.__enter__.return_value = geometry
                result = runner.run_task(ROOT/'tasks/s1_parallel_place.toml', out=folder)
            output = Path(folder)/engine.problem.name
            saved = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            check(result, output, execute, saved)
            owner.close.assert_called_once()

    def test_uses_complete_reference_instead_of_recompiling(self):
        class Captured(CoppeliaBackend):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.compiler.compile = MagicMock(side_effect=AssertionError('duplicate compile'))
        def check(result, output, execute, saved):
            self.assertTrue(result['passed'], result.get('reason'))
            self.assertTrue((output/'reference_receipt.json').is_file())
            self.assertTrue(result['reference_reused'])
            execute.assert_called_once()
            tasks = execute.call_args.args[3]
            receipt = json.loads((output/'reference_receipt.json').read_text(encoding='utf-8'))
            self.assertEqual(tasks, receipt['tasks'])
        self.invoke(Captured, check)

    def test_terminal_unknown_matches_plan_and_summary_and_never_executes(self):
        class HomeBlocked(CoppeliaBackend):
            def finalize_reference(self, schedule, graph):
                self.validation_context = {'phase': 'final_reference', 'failed_segment': 'home_B',
                    'classification': 'home_goal_collision', 'failure_scope': 'terminal',
                    'retryable': False, 'pair': ['B_Finger_pos', 'red']}
                return MotionResult(Status.UNKNOWN, 'fixed goal collides with final block')
        def check(result, output, execute, saved):
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(result['real_planning_status'], 'unknown')
            self.assertEqual(saved['status'], 'unknown')
            plan = json.loads((output/'plan.json').read_text(encoding='utf-8'))
            self.assertEqual(plan['search']['status'], 'unknown')
            execute.assert_not_called()
        self.invoke(HomeBlocked, check)


if __name__ == '__main__':
    unittest.main()
