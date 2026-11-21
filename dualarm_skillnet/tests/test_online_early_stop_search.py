"""Offline contracts for the opt-in, per-search anytime stopping window."""

from dataclasses import asdict
import unittest
from unittest.mock import patch

from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.planning.search import search, weighted_anytime
from tests.test_bounded_search import bounded_engine
from tests.test_search import work_engine


class ExpansionClock:
    """Advance only at specified search expansions, independent of CPU speed."""

    def __init__(self, root_delays=()):
        self.now = 0.
        self.root_delays = iter(root_delays)
        self.roots = 0

    def __call__(self):
        return self.now

    def observe(self, state, running, elapsed):
        if not running and elapsed == 0.:
            self.roots += 1
            self.now += next(self.root_delays, 0.)


class OnlineEarlyStopSearchTests(unittest.TestCase):
    def test_default_and_explicit_none_return_the_same_exact_plan(self):
        with patch("skillnet.planning.search.monotonic", ExpansionClock()):
            original = search(bounded_engine(shared_goal=True))
            explicit = search(bounded_engine(shared_goal=True), post_first_solution_limit=None)
        self.assertEqual(asdict(original), asdict(explicit))
        self.assertEqual((explicit.status, explicit.weight, explicit.suboptimality_bound), ("solved", 1., 1.))
        self.assertFalse(explicit.early_stop_enabled)
        self.assertFalse(explicit.early_stop_triggered)

    def test_slow_first_solution_keeps_the_original_pre_solution_budget(self):
        clock = ExpansionClock([23.])
        with patch("skillnet.planning.search.monotonic", clock):
            answer = weighted_anytime(work_engine(), observer=clock.observe, post_first_solution_limit=2.)
        self.assertEqual((answer.status, answer.first_solution_seconds), ("solved", 23.))
        self.assertEqual(answer.stages[0]["time_budget"], 60.)
        self.assertTrue(answer.early_stop_enabled)
        self.assertFalse(answer.early_stop_triggered)

    def test_reference_probe_uses_incumbents_deadline(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        clock = ExpansionClock([4., 2.])
        with patch("skillnet.planning.search.monotonic", clock):
            result = weighted_anytime(engine, observer=clock.observe, post_first_solution_limit=2.)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.first_solution_seconds, 4.)
        self.assertEqual(result.elapsed, 6.)
        self.assertEqual([(s["phase"], s["status"]) for s in result.stages],
                         [("incumbent", "solved"), ("reference", "budget")])
        self.assertEqual(result.stages[-1]["time_budget"], 2.)
        self.assertTrue(result.early_stop_triggered)
        self.assertEqual(result.early_stop_reason, "post_first_solution_limit")
        self.assertIsNone(result.stages[0]["lower_bound"])
        self.assertIsNone(result.stages[1]["suboptimality_bound"])

    def test_weighted_improvement_does_not_extend_reference_triggered_window(self):
        engine = bounded_engine(shared_goal=True)

        class ArmCost:
            def estimate(self, action, state):
                return 1. if action.args[0] == "A" else 3.

        engine.cost = ArmCost()
        engine.hooks.reference_actions = lambda actions, initial, goal: {("work", ("B",))}
        clock = ExpansionClock([4., .5, 1.5])
        with patch("skillnet.planning.search.monotonic", clock):
            result = weighted_anytime(engine, observer=clock.observe, post_first_solution_limit=2.)
        self.assertEqual((result.status, result.makespan, result.first_solution_seconds), ("solved", 1., 4.))
        self.assertEqual(result.elapsed, 6.)
        self.assertEqual([(s["phase"], s["weight"], s["status"]) for s in result.stages],
                         [("reference", 3., "solved"), ("weighted", 3., "solved"), ("weighted", 2., "budget")])
        self.assertEqual(result.stages[-1]["time_budget"], 1.5)
        self.assertTrue(result.early_stop_triggered)

    def test_weight_one_can_prove_the_optimum_before_the_window_expires(self):
        clock = ExpansionClock([0., .25, .25, .25])
        with patch("skillnet.planning.search.monotonic", clock):
            result = weighted_anytime(work_engine(), observer=clock.observe, post_first_solution_limit=2.)
        self.assertEqual((result.status, result.weight, result.lower_bound, result.suboptimality_bound),
                         ("solved", 1., 2., 1.))
        self.assertEqual(result.elapsed, .75)
        self.assertEqual(result.early_stop_reason, "optimality_proven")
        self.assertFalse(result.early_stop_triggered)

    def test_expiry_does_not_manufacture_a_solution_or_consume_unknown_status(self):
        for expected, engine, options in (("budget", work_engine(), {"max_expansions": 1}),
                                          ("unknown", work_engine(status="UNKNOWN"), {})):
            with self.subTest(expected=expected), patch("skillnet.planning.search.monotonic", ExpansionClock()):
                result = search(engine, post_first_solution_limit=2., **options)
                self.assertEqual(result.status, expected)
                self.assertIsNone(result.first_solution_seconds)
                self.assertIsNone(result.early_stop_reason)
                self.assertTrue(result.early_stop_enabled)
                self.assertFalse(result.early_stop_triggered)

    def test_original_deadline_wins_when_first_solution_arrives_late(self):
        clock = ExpansionClock([59., 1.])
        with patch("skillnet.planning.search.monotonic", clock):
            result = weighted_anytime(work_engine(), observer=clock.observe, post_first_solution_limit=2.)
        self.assertEqual((result.status, result.first_solution_seconds, result.elapsed), ("solved", 59., 60.))
        self.assertEqual(result.stages[-1]["time_budget"], 1.)
        self.assertFalse(result.early_stop_triggered)

