"""The full optimized profile and bounded legacy diagnostic have clear budgets."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from tools import motion_v135_m5_chain as chain


class M5ProfilePolicyTests(unittest.TestCase):
    def test_full_optimized_profile_uses_original_budget_and_complete_rollout(self):
        command, hard = chain.profile_command('00', 'optimized', Path('out/example'), True)
        self.assertEqual(hard, 1200.)
        self.assertEqual(command[command.index('--seconds')+1], '1180')
        self.assertEqual(command[command.index('--until')+1], 'rollout')
        self.assertIn('--fast-idle', command)
        self.assertNotIn('--legacy-sample-reset', command)

    def test_legacy_is_explicit_short_diagnostic_and_keeps_original_dispatch(self):
        command, hard = chain.profile_command('02', 'legacy', Path('out/example'), True)
        self.assertEqual(hard, 220.)
        self.assertEqual(command[command.index('--seconds')+1], '120')
        self.assertIn('--legacy-sample-reset', command)
        self.assertNotIn('--fast-idle', command)
        self.assertEqual(command[command.index('--until')+1], 'rollout')

    def test_original_short_profile_mode_is_compatible(self):
        command, hard = chain.profile_command('00', 'optimized', Path('out/example'), False)
        self.assertEqual(hard, 220.)
        self.assertEqual(command[command.index('--seconds')+1], '120')
        self.assertNotIn('--fast-idle', command)

    def test_first_full_attempt_persists_mode_before_any_live_launch(self):
        with tempfile.TemporaryDirectory(dir=chain.ROOT/'out') as folder:
            with (patch.object(chain, 'root_for', return_value=Path(folder)),
                  patch.object(chain, 'wait_gate', side_effect=RuntimeError('mock boundary')),
                  patch.object(chain, 'rebuild'), patch.object(chain.control, 'record'),
                  patch.object(chain.control, 'run_job') as live):
                result = chain.run('new_full', 'new_gate', complete_optimized=True)
            live.assert_not_called()
            self.assertTrue(result['complete_optimized'])
            self.assertEqual(result['status'], 'interrupted')


if __name__ == '__main__':
    unittest.main()
