"""The final physical jobs hand off serially to the durable late workflow."""
from pathlib import Path
import tempfile
import unittest

from tools import overnight_batch as old
from tools import v133_finish_physical as finish


class PhysicalFinishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=old.ROOT/'out')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        old.atomic_json(self.base/'physical_stage_outcome.json', {
            'finished': False, 'deadline_epoch': 10000., 'fixtures': {'kept': 'proof'}})

    def invoke(self, **overrides):
        self.calls = []
        options = dict(base=self.base, runner=lambda stage: self.calls.append(stage) or {'status': 'phase_complete'},
            preparer=lambda stage: self.calls.append('prepare:'+stage), checker=lambda: {'passed': True},
            reporter=lambda: None, recorder=lambda text: None, now=lambda: 100.,
            late=lambda: self.calls.append('late') or {'status': 'finished'})
        options.update(overrides)
        return finish.run(**options)

    def test_physical_jobs_finish_before_late_workflow_and_resume_does_not_repeat_them(self):
        result = self.invoke()
        self.assertEqual(self.calls, ['d_platform_required_final', 'prepare:d_gates', 'd_gates', 'late'])
        self.assertEqual(result['status'], 'finished')
        outcome = old.read_json(self.base/'physical_stage_outcome.json')
        self.assertTrue(outcome['finished'])
        self.assertEqual(outcome['fixtures'], {'kept': 'proof'})
        self.invoke()
        self.assertEqual(self.calls, ['late'])

    def test_global_stop_prevents_gates_and_late_workflow(self):
        result = self.invoke(runner=lambda stage: {'status': 'global_stop', 'global_stop': 'archive mismatch'})
        self.assertEqual(result['status'], 'global_stop')
        self.assertEqual(self.calls, [])
        self.assertFalse(old.read_json(self.base/'physical_stage_outcome.json')['finished'])

    def test_physical_cap_skips_gates_without_changing_task_budget(self):
        result = self.invoke(now=lambda: 9900.)
        self.assertEqual(self.calls, ['d_platform_required_final', 'late'])
        outcome = old.read_json(self.base/'physical_stage_outcome.json')
        self.assertEqual(outcome['d_gates']['status'], 'skipped_physical_cap')
        self.assertIn('1200', outcome['d_gates']['reason'])
        self.assertEqual(result['status'], 'finished')

    def test_ordinary_task_failure_does_not_stop_independent_work(self):
        def runner(stage):
            self.calls.append(stage)
            return {'status': 'phase_complete', 'passed': False}
        self.invoke(runner=runner)
        self.assertEqual(self.calls[-1], 'late')


if __name__ == '__main__':
    unittest.main()
