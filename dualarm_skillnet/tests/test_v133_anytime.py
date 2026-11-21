"""Anytime search certificates on small, independently worked event graphs."""
import unittest
import importlib
from unittest.mock import patch

from skillnet.capability.oracle import CapabilityOracle
from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data
from skillnet.core.logic import parse_condition, parse_fact
from skillnet.core.problem import Problem
from skillnet.planning.search import _search
from tests.test_search import TestHooks, work_engine

search_module = importlib.import_module("skillnet.planning.search")


def two_route_engine():
    def action(name, source, target, duration):
        return {"name": name, "params": {"arm": "arm"}, "pre": [f"{source}(?arm)"],
            "delete": [f"{source}(?arm)"], "add": [f"{target}(?arm)"],
            "occupies": ["?arm"], "op_time": duration}
    skills = [action("expensive", "start", "done", 9.),
              action("cheap_prefix", "start", "middle", 1.),
              action("cheap_finish", "middle", "done", 4.)]
    problem = Problem("two_routes", "最优5秒，首解9秒", {"arm": ("A",), "block": (), "loc": (), "slot": ()},
        frozenset(), frozenset({parse_fact("start(A)")}), (parse_condition("done(A)"),),
        library_from_data({"skill": skills}), TestHooks(), CapabilityOracle())
    return Engine(problem)


def admissible_two_route_bound(state, running=()):
    if parse_fact("done(A)") in state:
        return 0.
    for key, remaining in running:
        if key[0] == "cheap_prefix":
            return remaining+4.
        if key[0] == "cheap_finish":
            return remaining
        if key[0] == "expensive":
            return 0.  # A deliberately weaker, still admissible branch estimate.
    return 4. if parse_fact("middle(A)") in state else 5.


class AnytimeCertificateTests(unittest.TestCase):
    def test_weighted_first_solution_nine_has_unweighted_frontier_five_and_bound_one_point_eight(self):
        answer = _search(two_route_engine(), bound=admissible_two_route_bound,
                         weight=3., max_expansions=3)
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 9.)
        self.assertEqual(answer.lower_bound, 5.)
        self.assertEqual(answer.suboptimality_bound, 1.8)
        self.assertEqual(answer.expanded, 3)

    def test_anytime_keeps_first_solution_when_shared_expansion_budget_is_used(self):
        answer = search_module.weighted_anytime(two_route_engine(), bound=admissible_two_route_bound,
                                               max_expansions=3)
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 9.)
        self.assertEqual(answer.suboptimality_bound, 1.8)
        self.assertEqual(answer.expanded, 3)
        self.assertEqual([stage["weight"] for stage in answer.stages], [3.])

    def test_weight_one_finishes_with_the_original_exact_schedule(self):
        engine = two_route_engine()
        exact = _search(engine, bound=admissible_two_route_bound)
        answer = search_module.weighted_anytime(engine, bound=admissible_two_route_bound,
                                               max_expansions=100)
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 5.)
        self.assertEqual(answer.steps, exact.steps)
        self.assertEqual(answer.weight, 1.)
        self.assertEqual(answer.suboptimality_bound, 1.)
        self.assertEqual([stage["weight"] for stage in answer.stages], [3., 2., 1.5, 1.])
        self.assertEqual(answer.expanded, sum(stage["expanded"] for stage in answer.stages))

    def test_same_cost_incumbent_cannot_replace_exact_hint_tie_order(self):
        engine = two_route_engine()
        from dataclasses import replace
        library = engine.problem.library
        library.skills = tuple(replace(skill, op_time=5.) if skill.name == "expensive" else skill for skill in library.skills)
        engine = Engine(engine.problem)
        hint = {("cheap_prefix", ("A",)), ("cheap_finish", ("A",))}
        exact = _search(engine, bound=admissible_two_route_bound, hint=hint)
        self.assertEqual([step.action.name for step in exact.steps], ["cheap_prefix", "cheap_finish"])
        answer = search_module.weighted_anytime(engine, bound=admissible_two_route_bound, hint=hint)
        self.assertEqual(answer.stages[0]["makespan"], 5.)
        self.assertEqual(answer.steps, exact.steps)
        self.assertEqual(answer.weight, 1.)

    def test_time_budget_is_shared_across_stages(self):
        clock = [0.]
        def observe(state, running, elapsed):
            clock[0] += .25
        with patch.object(search_module, "monotonic", side_effect=lambda: clock[0]):
            answer = search_module.weighted_anytime(two_route_engine(), bound=admissible_two_route_bound,
                time_limit=.75, observer=observe)
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 9.)
        self.assertEqual(answer.elapsed, .75)
        self.assertEqual(len(answer.stages), 1)
        self.assertEqual(answer.suboptimality_bound, 1.8)

    def test_closed_root_is_not_an_open_lower_bound(self):
        def bound(state, running=()):
            return 0. if parse_fact("start(A)") in state else admissible_two_route_bound(state, running)
        answer = _search(two_route_engine(), bound=bound, weight=3., max_expansions=3)
        self.assertEqual(answer.lower_bound, 5.)
        self.assertEqual(answer.suboptimality_bound, 1.8)

    def test_dominated_stale_entry_is_not_an_open_lower_bound(self):
        engine = two_route_engine()
        def action(name, source, target, duration):
            return {"name": name, "params": {"arm": "arm"}, "pre": [f"{source}(?arm)"],
                    "delete": [f"{source}(?arm)"], "add": [f"{target}(?arm)"],
                    "occupies": ["?arm"], "op_time": duration}
        engine.problem.library = library_from_data({"skill": [
            action("old_long", "start", "x", 9.), action("new_short", "start", "x", 5.),
            action("other", "start", "y", 10.), action("finish_x", "x", "done", 15.),
            action("finish_y", "y", "done", 100.)]})
        engine = Engine(engine.problem)
        def bound(state, running=()):
            if running:
                return 0.
            return 5. if parse_fact("x(A)") in state or parse_fact("y(A)") in state else 0.
        answer = _search(engine, bound=bound, weight=3., max_expansions=7)
        self.assertEqual(answer.makespan, 20.)
        self.assertEqual(answer.lower_bound, 15.)
        self.assertEqual(answer.suboptimality_bound, 20./15.)

    def test_reference_incumbent_with_zero_full_frontier_has_no_finite_certificate(self):
        engine = two_route_engine()
        engine.hooks.reference_actions = lambda actions, initial, goal: frozenset(a.key for a in actions)
        clock, calls = [0.], [0]
        def observe(state, running, elapsed):
            calls[0] += 1
            clock[0] += .1 if calls[0] <= 3 else 3.7
        with patch.object(search_module, "monotonic", side_effect=lambda: clock[0]):
            answer = search_module.weighted_anytime(engine, bound=lambda state, running=(): 0.,
                hint={('expensive', ('A',))}, observer=observe, time_limit=4.)
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 9.)
        self.assertEqual(answer.lower_bound, 0.)
        self.assertIsNone(answer.suboptimality_bound)
        self.assertEqual(answer.expanded, 4)
        self.assertEqual(answer.elapsed, 4.)

    def test_no_incumbent_budget_does_not_fabricate_a_solution_or_bound(self):
        answer = search_module.weighted_anytime(two_route_engine(), bound=admissible_two_route_bound,
                                               max_expansions=2)
        self.assertEqual(answer.status, "budget")
        self.assertFalse(answer.steps)
        self.assertIsNone(answer.suboptimality_bound)
        self.assertEqual(answer.expanded, 2)

    def test_running_effects_and_search_restrictions_survive_every_restart(self):
        engine = work_engine(status="UNKNOWN")
        action = engine.parse_action("work(A)")
        answer = search_module.weighted_anytime(engine, start=engine.start(action, engine.initial),
            goal=[parse_condition("done(A)")], allowed=iter([]), protected=iter([parse_fact("free(A)")]),
            initial_running=iter([(action.key, 1.25)]), bound=lambda state, running=(): max((t for _, t in running), default=0.))
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 1.25)
        self.assertEqual(answer.weight, 1.)
        self.assertEqual(answer.suboptimality_bound, 1.)
        self.assertFalse(answer.unknown)

    def test_search_result_positional_construction_and_unknown_zero_goal_metadata(self):
        legacy = search_module.SearchResult("budget", [], 0., 0, 1, {}, frozenset(), 0.)
        self.assertIsNone(legacy.suboptimality_bound)
        unknown = search_module.search(work_engine(status="UNKNOWN"))
        self.assertEqual(unknown.status, "unknown")
        self.assertIsNone(unknown.suboptimality_bound)
        empty = search_module.search(work_engine(), goal=[])
        self.assertEqual(empty.status, "solved")
        self.assertEqual(empty.suboptimality_bound, 1.)

    def test_plan_serialization_exposes_bound_and_weight_stages(self):
        from skillnet.pipeline import run
        from skillnet.report import plan_to_dict
        from skillnet.domains.blocks import load_problem
        result = run(load_problem("tasks/s1_nearest_arm.toml"))
        saved = plan_to_dict(result)
        self.assertEqual(saved["suboptimality_bound"], 1.)
        self.assertEqual(saved["search"]["suboptimality_bound"], 1.)
        self.assertEqual(saved["search"]["stages"][-1]["weight"], 1.)


if __name__ == "__main__":
    unittest.main()
