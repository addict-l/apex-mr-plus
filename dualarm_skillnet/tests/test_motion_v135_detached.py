"""Independent Windows supervisor launch contracts; all workers are mocked."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import motion_v135_detached as detached


class DetachedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=detached.ROOT/'out')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.target = self.folder/'target.py'
        self.target.write_text('raise SystemExit(0)\n', encoding='utf-8')
        self.base = self.folder/'results'
        self.base_patch = patch.object(detached, 'BASE', self.base)
        self.base_patch.start()
        self.addCleanup(self.base_patch.stop)
        self.process = Mock(pid=100)

    def receipt(self, spec):
        return {'pid': 100, 'birth': 'born100', 'is_in_job': False,
                'detached_from_launcher_job': True,
                'name': spec['name'], 'command': spec['command'],
                'python_executable': spec['python_executable'],
                'target_sha256': spec['target_sha256'],
                'helper_sha256': spec['helper_sha256'], 'status': 'running',
                'started_at': '2026-11-18T00:00:00+00:00'}

    def launch(self, name='unit'):
        return detached.launch(name, self.folder/'unit.log', [str(self.target), 'arg with space'])

    def test_explicit_breakaway_detach_group_real_logs_and_no_handle_leak(self):
        with patch.object(detached.subprocess, 'Popen', return_value=self.process) as popen, \
                patch.object(detached, '_wait_receipt', side_effect=lambda proc, path, spec, timeout: self.receipt(spec)), \
                patch.object(detached, 'process_birth', return_value='born100'):
            result = self.launch()
        call = popen.call_args
        self.assertEqual(call.kwargs['creationflags'], 0x01000208)
        self.assertTrue(call.kwargs['close_fds'])
        self.assertEqual(call.kwargs['stdin'], detached.subprocess.DEVNULL)
        self.assertEqual(Path(call.kwargs['cwd']), detached.ROOT)
        self.assertTrue(call.kwargs['stdout'].closed)
        self.assertTrue(call.kwargs['stderr'].closed)
        self.assertEqual(result['status'], 'started')
        self.assertEqual(result['pid'], 100)
        self.assertEqual(result['birth'], 'born100')

    def test_live_same_identity_is_adopted_without_second_process(self):
        with patch.object(detached.subprocess, 'Popen', return_value=self.process), \
                patch.object(detached, '_wait_receipt', side_effect=lambda proc, path, spec, timeout: self.receipt(spec)), \
                patch.object(detached, 'process_birth', return_value='born100'):
            self.launch()
        path = self.base/'launches/unit'
        spec = json.loads((path/'spec.json').read_text(encoding='utf-8'))
        (path/'receipt.json').write_text(json.dumps(self.receipt(spec)), encoding='utf-8')
        with patch.object(detached.subprocess, 'Popen') as popen, \
                patch.object(detached, 'process_birth', return_value='born100'):
            answer = self.launch()
        self.assertEqual(answer['status'], 'already_running')
        popen.assert_not_called()

    def test_pid_reuse_or_changed_command_refuses_to_reset_saved_attempt(self):
        with patch.object(detached.subprocess, 'Popen', return_value=self.process), \
                patch.object(detached, '_wait_receipt', side_effect=lambda proc, path, spec, timeout: self.receipt(spec)), \
                patch.object(detached, 'process_birth', return_value='born100'):
            self.launch()
        path = self.base/'launches/unit'
        spec = json.loads((path/'spec.json').read_text(encoding='utf-8'))
        (path/'receipt.json').write_text(json.dumps(self.receipt(spec)), encoding='utf-8')
        with patch.object(detached.subprocess, 'Popen') as popen, \
                patch.object(detached, 'process_birth', return_value='reused'):
            with self.assertRaisesRegex(RuntimeError, '保留|preserv|finished|identity'):
                self.launch()
        popen.assert_not_called()
        self.target.write_text('raise SystemExit(1)\n', encoding='utf-8')
        with patch.object(detached.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, '改变|mismatch|版本'):
                self.launch()
        popen.assert_not_called()

    def test_outside_project_paths_and_bad_names_are_rejected_before_popen(self):
        for name, log, command in [('..', self.folder/'x.log', [str(self.target)]),
                ('unit', detached.ROOT.parent/'bad.log', [str(self.target)]),
                ('unit', self.folder/'x.log', [str(detached.ROOT.parent/'bad.py')])]:
            with self.subTest(name=name, log=log), patch.object(detached.subprocess, 'Popen') as popen:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    detached.launch(name, log, command)
                popen.assert_not_called()

    def test_child_checks_own_job_before_running_target_and_keeps_same_pid(self):
        with patch.object(detached.subprocess, 'Popen', return_value=self.process), \
                patch.object(detached, '_wait_receipt', side_effect=lambda proc, path, spec, timeout: self.receipt(spec)), \
                patch.object(detached, 'process_birth', return_value='born100'):
            self.launch()
        spec = self.base/'launches/unit/spec.json'
        (spec.parent/'approval.json').write_text(json.dumps({'pid': 234, 'birth': 'actual-birth',
            'detached_from_launcher_job': True}), encoding='utf-8')
        with patch.object(detached, '_inside_job', return_value=False), \
                patch.object(detached, 'process_birth', return_value='actual-birth'), \
                patch.object(detached.os, 'getpid', return_value=234), \
                patch.object(detached.runpy, 'run_path', side_effect=SystemExit(0)) as run:
            code = detached.worker(spec)
        self.assertEqual(code, 0)
        receipt = json.loads((spec.parent/'receipt.json').read_text(encoding='utf-8'))
        self.assertEqual(receipt['pid'], 234)
        self.assertFalse(receipt['is_in_job'])
        run.assert_called_once_with(str(self.target.resolve()), run_name='__main__')
        self.assertTrue((spec.parent/'exit.json').is_file())

    def test_child_still_in_launcher_kill_job_is_not_approved(self):
        with patch.object(detached, '_inside_job', return_value=True), \
                patch.object(detached, '_job_evidence', return_value={
                    'query_ok': True, 'pids': [100], 'kill_on_close': True}):
            with self.assertRaisesRegex(RuntimeError, 'Job'):
                detached._confirm_detached({'pid': 100, 'birth': 'born100'}, {})

    def test_outer_job_is_allowed_when_child_is_outside_launcher_private_job(self):
        folder = self.base/'launches/test'
        folder.mkdir(parents=True)
        with patch.object(detached, '_inside_job', return_value=True), \
                patch.object(detached, '_job_evidence', return_value={
                    'query_ok': True, 'pids': [20, 21], 'kill_on_close': True}):
            proof = detached._confirm_detached({'pid': 100, 'birth': 'born100', 'in_any_job': True},
                {'receipt_path': str(folder/'receipt.json')})
        self.assertTrue(proof['detached_from_launcher_job'])
        self.assertNotIn(100, proof['launcher_job_after_spawn']['pids'])

    def test_pending_old_receipt_is_not_adopted_or_restarted(self):
        with patch.object(detached.subprocess, 'Popen', return_value=self.process), \
                patch.object(detached, '_wait_receipt', side_effect=lambda proc, path, spec, timeout: self.receipt(spec)), \
                patch.object(detached, 'process_birth', return_value='born100'):
            self.launch()
        folder = self.base/'launches/unit'
        spec = json.loads((folder/'spec.json').read_text(encoding='utf-8'))
        receipt = self.receipt(spec)
        receipt['detached_from_launcher_job'] = False
        (folder/'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
        with patch.object(detached, 'process_birth', return_value='born100'), \
                patch.object(detached.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, '未完成|确认'):
                self.launch()
        popen.assert_not_called()

    def test_wmi_option_uses_service_creator_and_same_receipt_checks(self):
        with patch.object(detached, '_spawn_wmi', return_value=self.process) as creator, \
                patch.object(detached.subprocess, 'Popen') as direct, \
                patch.object(detached, '_wait_receipt', side_effect=lambda proc, path, spec, timeout: self.receipt(spec)), \
                patch.object(detached, 'process_birth', return_value='born100'):
            result = detached.launch('wmiunit', self.folder/'unit.log', [str(self.target)], via='wmi')
        self.assertEqual(result['status'], 'started')
        creator.assert_called_once()
        direct.assert_not_called()


if __name__ == '__main__':
    unittest.main()
