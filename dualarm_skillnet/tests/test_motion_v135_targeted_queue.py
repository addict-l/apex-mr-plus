"""Representative reruns preserve source identities and fixed budgets."""
import unittest

from tools import motion_v135_diagnostics as queue


class TargetedQueueTests(unittest.TestCase):
    def test_select_explicit_representatives_preserves_order_and_original_sha(self):
        jobs, _ = queue.baseline_jobs()
        ids = ['v134-fixed20-00', 'v134-fixed20-12', 'v134-fixed20-08', 'v134-fixed20-16']
        selected = queue.select_jobs(jobs, ids)
        self.assertEqual([row['baseline_job_id'] for row in selected], ids)
        self.assertTrue(all(row['timeout_seconds'] == 1200. for row in selected))
        self.assertTrue(all(row in jobs for row in selected))

    def test_implicit_original_diagnostic_still_has_exact_fourteen(self):
        jobs, _ = queue.baseline_jobs()
        self.assertEqual(queue.select_jobs(jobs, None), jobs)
        self.assertEqual(len(jobs), 14)

    def test_empty_duplicate_and_non_original_representatives_are_rejected(self):
        jobs, _ = queue.baseline_jobs()
        for ids in ([], ['v134-fixed20-00']*2, ['v134-fixed20-01']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                queue.select_jobs(jobs, ids)


if __name__ == '__main__':
    unittest.main()
