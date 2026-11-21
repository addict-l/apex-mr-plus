"""End-to-end acceptance tests for stable temporary support strategies."""

import unittest

from skillnet.core.engine import Engine
from skillnet.core.logic import Fact
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from tests.test_replay import replay


class SupportScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for name in ("s3a_cantilever", "s3a_platform", "s3a_platform_demo_needs_support"):
            cls.results[name] = run(load_problem(f"tasks/{name}.toml"))

    def test_all_support_scenarios_solve_and_replay_safely(self):
        for name, result in self.results.items():
            with self.subTest(task=name):
                self.assertEqual(result.status, "solved", result.feedback)
                self.assertIsNotNone(result.graph)
                self.assertEqual(len(result.motion), len(result.schedule.steps))
                replay(self, Engine(result.problem), result.schedule)

    def test_cantilever_uses_arm_support_until_counterweight_is_installed(self):
        result = self.results["s3a_cantilever"]
        actions = [step.action for step in result.schedule.steps]
        self.assertIn("support_on", [action.name for action in actions])
        self.assertIn("support_off", [action.name for action in actions])
        self.assertFalse(any(action.name in {"insert", "detach"} and action.args[-1] in {"tp1", "tp2"}
                             for action in actions))
        counterweight = next(step for step in result.schedule.steps
                             if step.action.name == "insert" and step.action.args[-1] == "cw")
        releases = [step for step in result.schedule.steps if step.action.name == "support_off"]
        for release in releases:
            self.assertGreaterEqual(release.start, counterweight.end)

    def test_platform_is_built_used_and_dismantled_with_free_arms(self):
        result = self.results["s3a_platform"]
        steps = result.schedule.steps
        self.assertNotIn("support_on", [step.action.name for step in steps])
        beam_insert = next(step for step in steps if step.action.name == "insert" and step.action.args[-1] == "beam")
        self.assertTrue(result.problem.hooks.evaluate("has_support", ("beam",), beam_insert.state))
        for slot in ("tp1", "tp2"):
            installed = [step for step in steps if step.action.name == "insert" and step.action.args[-1] == slot]
            detached = [step for step in steps if step.action.name == "detach" and step.action.args[-1] == slot]
            self.assertTrue(installed)
            self.assertTrue(detached)
            self.assertLessEqual(installed[0].end, beam_insert.start)
            self.assertGreaterEqual(detached[-1].start, beam_insert.end)
        final_state = replay(self, Engine(result.problem), result.schedule)
        for slot in ("tp1", "tp2"):
            self.assertNotIn(Fact("filled", (slot,)), final_state)
        for arm in result.problem.objects["arm"]:
            self.assertIn(Fact("free", (arm,)), final_state)

    def test_needs_support_during_insertion_and_causal_graph(self):
        result = self.results["s3a_platform_demo_needs_support"]
        cap = next(step for step in result.schedule.steps if step.action.name == "insert" and step.action.args[-1] == "cap")
        self.assertTrue(result.problem.hooks.evaluate("has_support", ("beam",), cap.state))
        self.assertIn(Fact("supporting", ("A", "beam")), cap.state)
        for release in result.schedule.steps:
            if release.action.name == "support_off":
                self.assertGreaterEqual(release.start, cap.end)
        self.assertTrue(any(edge.kind == "causal" and edge.label == "supporting(A, beam)"
                            for edge in result.graph.edges))


if __name__ == "__main__":
    unittest.main()
