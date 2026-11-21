"""Controlled TPG profiles retain exact idle restoration and source scope."""
from contextlib import contextmanager
import unittest

from tools import motion_v135_tpg_profile as profile


class FastProfileScopeTests(unittest.TestCase):
    def test_disabled_mode_keeps_the_original_profile_compatible(self):
        class Untouched:
            def fast_idle_stopped(self, **kwargs):
                raise AssertionError('disabled profile must not change idle settings')
        report = {}
        with profile.geometry_mode(Untouched(), report, 42., enabled=False):
            self.assertFalse(report['fast_idle_enabled'])

    def test_enabled_mode_restores_after_a_tpg_failure_and_records_original_deadline(self):
        calls = []
        class Owner:
            @contextmanager
            def fast_idle_stopped(self, *, deadline):
                calls.append(deadline)
                receipt = {'enabled': True, 'restored': False}
                try:
                    yield receipt
                finally:
                    receipt['restored'] = True
        report = {}
        with self.assertRaisesRegex(ValueError, 'geometry failed'):
            with profile.geometry_mode(Owner(), report, 42., enabled=True):
                raise ValueError('geometry failed')
        self.assertEqual(calls, [42.])
        self.assertTrue(report['fast_idle_scope']['restored'])

    def test_unverified_restoration_cannot_leave_a_completed_profile(self):
        class Owner:
            @contextmanager
            def fast_idle_stopped(self, *, deadline):
                yield {'enabled': True, 'restored': False}
        with self.assertRaisesRegex(RuntimeError, 'restor'):
            with profile.geometry_mode(Owner(), {}, 42., enabled=True):
                pass


if __name__ == '__main__':
    unittest.main()
