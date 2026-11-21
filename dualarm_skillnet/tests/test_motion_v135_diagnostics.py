"""Offline old-failure queue contracts; no solver or simulator launches."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import overnight_batch as old
from tools import motion_v135_diagnostics as queue


class DiagnosticQueueTests(unittest.TestCase):
    @contextmanager
    def prepared(self):
        with tempfile.TemporaryDirectory(dir=queue.control.BASE) as folder:
            base = Path(folder)
            root = base/'diagnostics/mock_queue'
            code = {'core': {'simbridge/backend.py': old.sha256(queue.ROOT/'simbridge/backend.py')}, 'tools': {}}
            gate = base/'normal_gate.json'
            old.atomic_json(gate, {'passed': True})
            proof = {'result_path': str(gate), 'result_sha256': old.sha256(gate), 'proof': {'evidence_files': {}}}
            with (patch.object(queue.control, 'BASE', base),
                  patch.object(queue, 'snapshot', return_value=code),
                  patch.object(queue.control, 'audit', return_value={'passed': True}),
                  patch.object(queue, '_check_gate', return_value=proof),
                  patch.object(queue, 'dependencies', return_value={'local_files': {}, 'remote_services_required': []})):
                queue.prepare('mock_queue', 'normal')
                yield root, code

    def test_prepare_preserves_fourteen_and_static_snapshot_refuses_changed_code(self):
        with self.prepared() as (root, code):
            selection = queue.verify(root)
            self.assertEqual(selection['requested'], 14)
            self.assertEqual(len(old.load_manifest(root/'manifest.json')['jobs']), 14)
            changed = deepcopy(code)
            changed['core']['simbridge/backend.py'] = 'changed'
            with patch.object(queue, 'snapshot', return_value=changed):
                with self.assertRaisesRegex(ValueError, '指纹'):
                    queue.verify(root)

    def test_changed_normal_gate_proof_is_rejected(self):
        with self.prepared() as (root, code):
            gate = Path(old.read_json(root/'selection.json')['normal_gate']['result_path'])
            old.atomic_json(gate, {'passed': False})
            with self.assertRaisesRegex(ValueError, '普通取放门'):
                queue.verify(root)

    def test_actual_execution_manifest_keeps_completed_task_and_recovers_only_interrupted(self):
        with self.prepared() as (root, code):
            original = old.load_manifest(root/'manifest.json')
            journal = old.Journal(original['state_dir'])
            first, second = original['jobs'][:2]
            journal.append({**first, 'status': 'unknown', 'passed': False})
            journal.append({**second, 'status': 'interrupted', 'passed': False})
            execution = queue.execution_manifest(root)
            self.assertEqual(execution['jobs'][0]['id'], first['id'])
            self.assertEqual(execution['jobs'][1]['id'], second['id']+'-resume1')
            self.assertNotEqual(execution['jobs'][1]['output'], second['output'])
            self.assertEqual(len(journal.load()), 2)

    def test_exact_original_fourteen_in_user_order_and_original_budgets(self):
        jobs, sources = queue.baseline_jobs()
        self.assertEqual([j['baseline_job_id'] for j in jobs], [row[1] for row in queue.ORDER])
        self.assertEqual(len(jobs), 14)
        self.assertEqual(sum(j['baseline_cohort'] == 'sim20' for j in jobs), 12)
        self.assertTrue(all(j['timeout_seconds'] == 1200. for j in jobs))
        self.assertTrue(all(j['task_sha256'] == old.sha256(j['task']) for j in jobs))
        self.assertTrue(all(not j['baseline_result']['passed'] for j in jobs))

    def test_task_sha_change_is_rejected_before_any_launch(self):
        original = old.sha256
        with patch.object(old, 'sha256', side_effect=lambda p: 'changed' if str(p).endswith('.toml') else original(p)):
            with self.assertRaisesRegex(ValueError, 'SHA'):
                queue.baseline_jobs()

    def test_same_logical_task_has_only_one_completed_attempt(self):
        jobs = [{'id': 'a'}, {'id': 'b'}]
        rows = {'a': {'id': 'a', 'status': 'interrupted'},
                'a-resume1': {'id': 'a-resume1', 'logical_job_id': 'a', 'status': 'unknown'}}
        selected = queue.canonical_rows(jobs, rows)
        self.assertEqual(selected['a']['id'], 'a-resume1')
        rows['a-resume2'] = {'id': 'a-resume2', 'logical_job_id': 'a', 'status': 'passed'}
        with self.assertRaisesRegex(ValueError, '重复|multiple'):
            queue.canonical_rows(jobs, rows)

    def test_recovered_worker_pass_requires_its_own_native_certificate(self):
        jobs = [{'id': 'a'}]
        record = {'id': 'a', 'status': 'passed', 'passed': True}
        with self.assertRaisesRegex(ValueError, '独立验证证明'):
            queue.canonical_rows(jobs, {'a': record})
        record['native_independent_passed'] = True
        self.assertEqual(queue.canonical_rows(jobs, {'a': record})['a'], record)

    def test_resume_only_uncommitted_interrupted_task_and_keeps_old_output(self):
        root = queue.control.BASE/'diagnostics/_mock_resume'
        canonical = {'jobs': [{'id': 'a', 'output': str(root/'a')}, {'id': 'b', 'output': str(root/'b')}]}
        records = {'a': {'id': 'a', 'status': 'passed', 'native_independent_passed': True},
                   'b': {'id': 'b', 'status': 'interrupted'}}
        jobs, receipts = queue.recovery_jobs(root, canonical['jobs'], records)
        self.assertEqual(jobs[0], canonical['jobs'][0])
        self.assertEqual(jobs[1]['id'], 'b-resume1')
        self.assertNotEqual(jobs[1]['output'], canonical['jobs'][1]['output'])
        self.assertEqual(receipts[0]['previous_attempts'], ['b'])

    def test_false_pass_without_independent_evidence_is_rejected(self):
        jobs, _ = queue.baseline_jobs()
        with tempfile.TemporaryDirectory(dir=queue.ROOT/'out') as folder:
            job = {**jobs[0], 'output': folder}
            result = queue.check_native(job, {'status': 'passed', 'passed': True})
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'missing_evidence')

    def test_ordinary_unknown_retains_classification_and_does_not_become_pass(self):
        jobs, _ = queue.baseline_jobs()
        result = queue.check_native(jobs[0], {'status': 'unknown', 'passed': False, 'reason': 'finite candidates'})
        self.assertEqual(result['status'], 'unknown')
        self.assertFalse(result['passed'])

    def test_progress_distinguishes_committed_counts_from_running_case(self):
        jobs, _ = queue.baseline_jobs()
        manifest = {'jobs': jobs}
        selected = {jobs[0]['id']: {'id': jobs[0]['id'], 'status': 'unknown', 'passed': False}}
        counts = queue.counts(manifest, selected)
        self.assertEqual(counts['total']['pending'], 13)
        self.assertEqual(counts['total']['failed'], 1)
        self.assertEqual(counts['sim20']['pending'], 11)
        self.assertEqual(counts['walls3']['pending'], 2)

    def test_startup_streak_cannot_lose_three_failures(self):
        records = {str(i): {'id': str(i), 'status': 'failed', 'passed': False,
                           'failure_phase': 'simulator_connect'} for i in range(3)}
        self.assertEqual(queue.startup_streak(records.values()), 3)
        records['ok'] = {'id': 'ok', 'status': 'unknown', 'passed': False, 'failure_phase': 'real_planning'}
        self.assertEqual(queue.startup_streak(records.values()), 0)

    def test_status_read_does_not_prepare_or_launch_queue(self):
        with tempfile.TemporaryDirectory(dir=queue.control.BASE) as folder:
            with patch.object(queue, 'queue_root', return_value=Path(folder)), patch.object(old, 'run_phase') as run:
                result = queue.status('absent')
            self.assertEqual(result['status'], 'not_prepared')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
