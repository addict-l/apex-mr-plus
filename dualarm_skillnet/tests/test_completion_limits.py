"""Repair budgets may discard proposals, never constrain optimal scheduling."""
import unittest
from unittest.mock import patch

from skillnet.core.engine import Engine
from skillnet.core.problem import Proposal
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from skillnet.planning.completion import CompletionReport, complete
from skillnet.planning.search import SearchResult, search


def failed_search(status="budget", elapsed=5.0):
    return SearchResult(status, [], 0.0, 50_000, 50_001, {}, frozenset(), elapsed)


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class CompletionLimitTests(unittest.TestCase):
    def test_subsearch_budget_abandons_without_another_goal_repair(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        with patch("skillnet.planning.completion.search", return_value=failed_search()) as solve:
            report = complete(engine, ["place(A, red, board_c)"])
        self.assertTrue(report.abandoned)
        self.assertIn("补全子搜索", report.abandon_reason)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(solve.call_args.kwargs["max_expansions"], 50_000)
        self.assertLessEqual(solve.call_args.kwargs["time_limit"], 5.0)
        self.assertFalse(report.feedback_for_model)

    def test_total_deadline_caps_the_last_subsearch(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        clock = Clock()
        calls = []

        def solve(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) < 5:
                clock.value += 4.1
                return failed_search("no_plan", 4.1)
            clock.value += kwargs["time_limit"]
            return failed_search("budget", kwargs["time_limit"])

        cycle = ["place(B, red, board_c)", "pick(A, red, stock_c)", "place(A, red, stock_c)"]
        with patch("skillnet.planning.completion.monotonic", clock, create=True), \
                patch("skillnet.planning.completion.search", side_effect=solve):
            report = complete(engine, cycle * 6)
        self.assertTrue(report.abandoned)
        self.assertIn("20", report.abandon_reason)
        self.assertEqual(len(calls), 5)
        self.assertLess(calls[-1]["time_limit"], 5.0)
        self.assertAlmostEqual(report.elapsed, 20.0)
        self.assertLess(report.deleted_steps, report.processed_steps / 2)

    def test_bad_quality_stops_after_four_original_steps(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        with patch("skillnet.planning.completion.search") as solve:
            report = complete(engine, ["broken", "unknown(A)", "pick(A)", "bad", "also_bad"])
        self.assertTrue(report.abandoned)
        self.assertIn("质量", report.abandon_reason)
        self.assertEqual(report.processed_steps, 4)
        self.assertEqual(report.deleted_steps, 4)
        self.assertEqual(len(report.entries), 4)
        solve.assert_not_called()

    def test_abandonment_skips_model_feedback_and_clears_every_search_hint(self):
        class BadModel:
            calls = []

            def decompose(self, problem, feedback=None):
                self.calls.append(feedback)
                return Proposal("", ["at(red, board_c)"],
                                ["unknown(A)"] * 3 + ["pick(A, red, stock_c)"], "模拟模型")

        model = BadModel()
        baseline = run(load_problem("tasks/s1_nearest_arm.toml"))
        with patch("skillnet.pipeline.search", wraps=search) as solve:
            result = run(load_problem("tasks/s1_nearest_arm.toml"), decomposer=model)
        self.assertTrue(result.completion.abandoned)
        self.assertEqual(len(result.completion.plan), 1)
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(all(call.kwargs["hint"] is None for call in solve.call_args_list))
        self.assertEqual(result.status, "solved")
        self.assertAlmostEqual(result.schedule.makespan, baseline.schedule.makespan, places=6)
        self.assertTrue(any("放弃" in entry for entry in result.feedback))
        self.assertGreaterEqual(result.completion_elapsed, 0)

    def test_exactly_half_deleted_does_not_trigger_quality_abandonment(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        report = complete(engine, ["unknown(A)", "pick(A, red, stock_c)",
                                   "place(A, red, stock_c)", "unknown(B)"])
        self.assertEqual(report.processed_steps, 4)
        self.assertEqual(report.deleted_steps, 2)
        self.assertFalse(report.abandoned)
        self.assertTrue(report.goal_reached)

    def test_model_rounds_share_one_completion_time_allowance(self):
        class RetryModel:
            count = 0

            def decompose(self, problem, feedback=None):
                self.count += 1
                return Proposal("", ["at(red, board_c)"], [], "模拟模型")

        model = RetryModel()
        reports = [CompletionReport(unresolved=["at(red, board_c)"], elapsed=12.0),
                   CompletionReport(unresolved=["at(red, board_c)"], elapsed=8.0)]
        with patch("skillnet.pipeline.complete", side_effect=reports) as repair:
            result = run(load_problem("tasks/s1_nearest_arm.toml"), decomposer=model)
        self.assertEqual(repair.call_count, 2)
        self.assertAlmostEqual(repair.call_args_list[0].kwargs["time_limit"], 20.0)
        self.assertLessEqual(repair.call_args_list[1].kwargs["time_limit"], 8.0)
        self.assertTrue(result.completion.abandoned)
        self.assertEqual(model.count, 2)
        self.assertGreaterEqual(result.completion_elapsed, 20.0)
        self.assertEqual(result.status, "solved")
