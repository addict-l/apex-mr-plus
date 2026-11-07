"""Retry exhaustion must not bypass a checked withdrawal or successful result."""
import unittest
from recovery.retry import RetryPolicy


class SkillRetryTests(unittest.TestCase):
    def test_third_failure_reassigns_only_after_retreat(self):
        policy=RetryPolicy(3)
        for attempt in (1,2):
            self.assertEqual(policy.decision(attempt,False),'retreat')
            self.assertEqual(policy.decision(attempt,False,safely_retreated=True),'retry')
        self.assertEqual(policy.decision(3,False),'retreat')
        self.assertEqual(policy.decision(3,False,safely_retreated=True),'reassign')

    def test_success_at_any_attempt_continues_original_assignment(self):
        for attempt in (1,2,3):
            self.assertEqual(RetryPolicy(3).decision(attempt,True),'continue')

    def test_emulator_tracks_configured_budget_including_one_attempt(self):
        self.assertEqual([RetryPolicy(3,'transient').injected_failure(a) for a in (1,2,3)],[True,True,False])
        self.assertFalse(RetryPolicy(1,'transient').injected_failure(1))
        self.assertEqual(RetryPolicy(1).decision(1,False,safely_retreated=True),'reassign')

    def test_invalid_budget_and_out_of_range_attempts_are_rejected(self):
        for count in (0,-1,True,2.5):
            with self.subTest(count=count),self.assertRaises(ValueError):RetryPolicy(count)
        for attempt in (0,4):
            with self.subTest(attempt=attempt),self.assertRaises(ValueError):RetryPolicy().decision(attempt,False)


if __name__=='__main__':unittest.main()
