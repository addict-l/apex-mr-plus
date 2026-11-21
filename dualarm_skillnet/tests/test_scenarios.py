"""Acceptance tests for the five basic block tasks."""

import unittest

from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.planning.search import search


class BasicScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        cls.engines = {}
        for name in ("s1_parallel_place", "s1_nearest_arm", "s1_reach_handover", "s1_far_handover", "s2_tower"):
            engine = Engine(load_problem(f"tasks/{name}.toml"))
            cls.engines[name] = engine
            cls.results[name] = search(engine)

    def test_all_basic_tasks_are_solved(self):
        for name, result in self.results.items():
            with self.subTest(task=name):
                self.assertEqual(result.status, "solved")
                self.assertGreater(len(result.steps), 0)
                self.assertAlmostEqual(result.makespan, max(step.end for step in result.steps))

    def test_independent_arms_overlap_and_save_elapsed_time(self):
        result = self.results["s1_parallel_place"]
        self.assertTrue(any(
            set(first.action.occupies).isdisjoint(second.action.occupies)
            and max(first.start, second.start) < min(first.end, second.end)
            for first in result.steps for second in result.steps
        ))
        serial_duration = sum(step.end - step.start for step in result.steps)
        self.assertLess(result.makespan, serial_duration)
        self.assertEqual({step.action.name for step in result.steps}, {"pick", "place"})

    def test_nearest_arm_performs_pick(self):
        picks = [step for step in self.results["s1_nearest_arm"].steps if step.action.name == "pick"]
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0].action.args[0], "A")

    def test_near_handover_is_selected(self):
        actions = [step.action for step in self.results["s1_reach_handover"].steps]
        self.assertIn(("handover", ("B", "A", "red", "hand_mid")), [action.key for action in actions])
        self.assertFalse(any(action.name == "place" and action.args[-1] == "buffer_mid" for action in actions))

    def test_far_handover_loses_to_buffer(self):
        result = self.results["s1_far_handover"]
        self.assertNotIn("handover", [step.action.name for step in result.steps])
        self.assertIn(("place", ("B", "red", "buffer_mid")), [step.action.key for step in result.steps])
        self.assertIn(("pick", ("A", "red", "buffer_mid")), [step.action.key for step in result.steps])
        self.assertGreater(result.makespan, self.results["s1_reach_handover"].makespan)

    def test_tower_installs_bottom_to_top(self):
        inserts = sorted((step for step in self.results["s2_tower"].steps if step.action.name == "insert"), key=lambda step: step.start)
        self.assertEqual([step.action.args[-1] for step in inserts], ["t1", "t2", "t3"])
        for lower, upper in zip(inserts, inserts[1:]):
            self.assertLessEqual(lower.end, upper.start)

    def test_moving_destination_farther_increases_makespan(self):
        problem = load_problem("tasks/s1_parallel_place.toml")
        problem.hooks.positions["board_l1"] = (-2.15, 0.20, 0.00)
        result = search(Engine(problem))
        self.assertEqual(result.status, "solved")
        self.assertGreater(result.makespan, self.results["s1_parallel_place"].makespan)


if __name__ == "__main__":
    unittest.main()
