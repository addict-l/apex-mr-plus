"""Verify the opt-in and certificate scope at motion feedback boundaries."""
import json
from time import monotonic
import unittest
from unittest.mock import patch

from skillnet.pipeline import run
from simbridge import runner
from simbridge.backend import CoppeliaBackend
from tests import test_motion_v135_runner_reference as runner_fixture
from tests.test_coppelia_backend import backend_fixture
from tests.test_motion_v135_feedback import NoProposal, schedule
from skillnet.motion.base import NullBackend


def search_events(rows):
    return [row for row in rows
            if row["query"] == "symbolic_search" and row["phase"] == "after"]


class OnlineEarlyStopWiringTests(unittest.TestCase):
    def test_runner_enables_early_stop_only_for_real_planning(self):
        def check(result, output, execute, saved):
            self.assertTrue(saved["passed"], saved.get("reason"))
            events = search_events([
                json.loads(line)["event"]
                for line in (output / "motion_live.jsonl").read_text(encoding="utf-8").splitlines()
            ])
            self.assertTrue(events)
            for row in events:
                self.assertTrue(row["early_stop_enabled"])
                self.assertEqual(row["post_first_solution_limit"], 2.0)
                self.assertIn("early_stop_triggered", row)
                self.assertIn("early_stop_reason", row)
                self.assertIn("first_solution_seconds", row)

        with patch.object(runner, "run", wraps=run) as dispatch:
            runner_fixture.RunnerReferenceTests().invoke(CoppeliaBackend, check)
        initial, online = dispatch.call_args_list
        self.assertNotIn("backend", initial.kwargs)
        self.assertNotIn("post_first_solution_limit", initial.kwargs)
        self.assertIsInstance(online.kwargs["backend"], CoppeliaBackend)
        self.assertEqual(online.kwargs["post_first_solution_limit"], 2.0)

    def test_online_relevance_search_bound_is_scoped_to_the_allowed_graph(self):
        engine, _, _ = backend_fixture()
        action = engine.parse_action("pick(A, red, stock_l1)")
        answer = schedule("solved", action, engine.initial)
        answer.makespan, answer.lower_bound, answer.suboptimality_bound = 10.0, 10.0, 1.0
        events = []
        backend = NullBackend()
        backend.progress_callback = events.append
        with patch("skillnet.pipeline.search", return_value=answer) as solve:
            result = run(engine.problem, decomposer=NoProposal(), backend=backend,
                         post_first_solution_limit=2.0)
        self.assertEqual(solve.call_args.kwargs["post_first_solution_limit"], 2.0)
        event, = search_events(events)
        self.assertTrue(event["pruned"])
        self.assertEqual(event["certificate_scope"], "restricted")
        self.assertEqual(event["lower_bound"], 10.0)
        self.assertIsNone(result.schedule.lower_bound)
        self.assertIsNone(result.schedule.suboptimality_bound)

    def test_expired_pipeline_deadline_preserves_opt_in_without_claiming_a_first_solution(self):
        engine, _, _ = backend_fixture()
        events = []
        backend = NullBackend()
        backend.progress_callback = events.append
        with patch("skillnet.pipeline.search") as solve:
            result = run(engine.problem, decomposer=NoProposal(), backend=backend, prune=False,
                         deadline=monotonic()-1., post_first_solution_limit=2.0)
        solve.assert_not_called()
        self.assertEqual(result.status, "budget")
        event, = search_events(events)
        self.assertTrue(event["early_stop_enabled"])
        self.assertFalse(event["early_stop_triggered"])
        self.assertIsNone(event["first_solution_seconds"])
        self.assertIsNone(event["early_stop_reason"])
        self.assertEqual(event["post_first_solution_limit"], 2.0)


if __name__ == "__main__":
    unittest.main()
