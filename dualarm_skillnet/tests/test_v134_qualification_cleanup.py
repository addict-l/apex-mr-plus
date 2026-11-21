"""Public qualification reserves cleanup inside its unchanged hard budget."""
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from simbridge import qualifier
from tests import test_v133_qualifier as old_tests


class QualificationCleanupTests(unittest.TestCase):
    def setUp(self):
        self.env = old_tests.QualifierTests()
        self.env.setUp()
        self.addCleanup(self.env.doCleanups)

    def slow_qualification(self, total, *, external=None, planner_exit_seconds=0., close_seconds=2.):
        clock = [100.]
        hard_deadline = 100.+(total if external is None else external)
        scale = (hard_deadline-100.)/60.
        original_qualify = self.env.backend.qualify
        observed = []
        stages = []
        def late_query(fact):
            observed.append(self.env.backend.compiler.deadline)
            answer = original_qualify(fact)
            clock[0] = self.env.backend.compiler.deadline+.01*scale
            stages.append("query")
            return answer
        self.env.backend.qualify = late_query
        original_json, original_text = qualifier.atomic_json, qualifier.atomic_text
        def slow_json(*args, **kwargs):
            original_json(*args, **kwargs)
            clock[0] += .1*scale
        def slow_text(*args, **kwargs):
            original_text(*args, **kwargs)
            clock[0] += .1*scale
        def slow_close():
            stages.append("simulator_close")
            clock[0] += close_seconds*scale
        def slow_planner_exit(*args):
            stages.append("planner_exit")
            clock[0] += planner_exit_seconds*scale
            return False
        planner = MagicMock()
        planner.__exit__.side_effect = slow_planner_exit
        self.env.owner.close.side_effect = slow_close
        with self.env.mocks(), patch.object(qualifier.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(qualifier, "atomic_json", side_effect=slow_json), \
                patch.object(qualifier, "atomic_text", side_effect=slow_text), \
                patch.object(qualifier, "_planner", return_value=planner):
            result = qualifier.qualify_task(self.env.task, out=self.env.output, timeout_wall_s=total,
                **({} if external is None else {"deadline": hard_deadline}))
        self.assertLess(clock[0], hard_deadline, "query used cleanup time and crossed the unchanged hard deadline")
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["simulator_closed"])
        self.assertEqual(result["counts"]["UNKNOWN"], result["total"])
        self.assertEqual(result["checked"], 1)
        self.assertTrue(result["remaining_unchecked"])
        self.env.owner.close.assert_called_once()
        self.assertEqual(stages, ["query", "planner_exit", "simulator_close"])
        log = json.loads(Path(result["log_file"]).read_text(encoding="utf-8"))
        self.assertEqual(log["status"], "partial")
        self.assertTrue(log["simulator_closed"])
        self.assertAlmostEqual(log["outer_deadline_monotonic"], hard_deadline)
        self.assertEqual(observed, [log["query_deadline_monotonic"]])
        self.assertAlmostEqual(log["cleanup_reserve_seconds"], min(15., (hard_deadline-100.)*.25))
        self.assertIsNone(self.env.backend.compiler.deadline)
        return log

    def test_query_stops_before_slow_close_and_log_within_sixty_seconds(self):
        self.slow_qualification(60.)

    def test_small_total_budget_uses_proportional_reserve(self):
        self.slow_qualification(2.)

    def test_earlier_external_deadline_is_preserved(self):
        self.slow_qualification(60., external=40.)

    def test_planner_exit_then_native_close_fit_twelve_second_cleanup_inside_sixty(self):
        self.slow_qualification(60., planner_exit_seconds=8., close_seconds=4.)


if __name__ == "__main__":
    unittest.main()
