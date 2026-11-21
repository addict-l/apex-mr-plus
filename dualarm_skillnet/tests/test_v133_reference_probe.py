"""Goal/domain reference probes on worked toy graphs, never task witnesses."""
import importlib
import unittest
from unittest.mock import patch

from skillnet.core.logic import parse_condition, parse_fact
from tests.test_search import work_engine
from tests.test_v11_ordering import graph_engine
from tests.test_v133_anytime import two_route_engine, admissible_two_route_bound
from tests.test_v133_unknown_relaxation import tiny_engine, skill


search_module = importlib.import_module("skillnet.planning.search")


def with_direct_goal_filter(engine):
    def reference_actions(actions, initial, goal):
        # This generic hook knows only the current grounded actions and goal.
        positive = {condition.fact for condition in goal if not condition.negated}
        return frozenset(action.key for action in actions if positive.intersection(action.add))
    engine.hooks.reference_actions = reference_actions
    return engine


class GoalDomainReferenceProbeTests(unittest.TestCase):
    def test_no_hint_probe_uses_weight_three_and_the_existing_bound(self):
        engine = two_route_engine()
        engine.hooks.reference_actions = lambda actions, initial, goal: frozenset(action.key for action in actions)
        answer = search_module.weighted_anytime(engine, bound=admissible_two_route_bound,
                                                max_expansions=1000)
        probe = answer.stages[0]
        self.assertEqual(probe["phase"], "reference")
        self.assertEqual(probe["weight"], 3.)
        self.assertEqual(probe["action_scope"], "goal_domain")
        self.assertEqual(probe["makespan"], 9.)
        self.assertTrue(probe["incumbent_only"])
        self.assertIsNone(probe["lower_bound"])
        self.assertIsNone(probe["suboptimality_bound"])
        self.assertEqual([stage["weight"] for stage in answer.stages[1:]], [3., 2., 1.5, 1.])
        self.assertEqual(answer.makespan, 5.)
        self.assertEqual(answer.weight, 1.)

    def test_only_a_probe_solution_uses_the_full_root_bound_instead_of_its_subset_bound(self):
        engine = with_direct_goal_filter(two_route_engine())
        clock, observed = [0.], [0]
        def observer(state, running, elapsed):
            observed[0] += 1
            clock[0] = 4. if observed[0] == 3 else clock[0]+.1
        with patch.object(search_module, "monotonic", side_effect=lambda: clock[0]):
            answer = search_module.weighted_anytime(engine, bound=admissible_two_route_bound,
                max_expansions=1000, time_limit=4., observer=observer)
        self.assertEqual([stage["phase"] for stage in answer.stages], ["reference"])
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 9.)
        self.assertEqual(answer.lower_bound, 5.)
        self.assertEqual(answer.suboptimality_bound, 1.8)
        self.assertIsNone(answer.stages[0]["lower_bound"])

    def test_probe_closure_is_not_a_global_conclusion_when_no_full_stage_can_start(self):
        for status in ("no_plan", "unknown"):
            with self.subTest(probe_status=status):
                if status == "unknown":
                    engine = tiny_engine([
                        skill("unknown_route", pre=["free(?arm)"], add=["done(?arm)"], capability=["gate(?arm)"]),
                        skill("known_route", pre=["free(?arm)"], add=["done(?arm)"])])
                    names = {"unknown_route"}
                else:
                    engine, names = two_route_engine(), set()
                clock, armed = [0.], [False]
                def reference_actions(actions, initial, goal):
                    armed[0] = True
                    return frozenset(action.key for action in actions if action.name in names)
                def now():
                    if armed[0]:
                        clock[0] += 1.
                    return clock[0]
                engine.hooks.reference_actions = reference_actions
                with patch.object(search_module, "monotonic", side_effect=now):
                    answer = search_module.weighted_anytime(engine, bound=lambda state, running=(): 0.,
                                                            time_limit=4., max_expansions=1000)
                self.assertEqual([stage["phase"] for stage in answer.stages], ["reference"])
                self.assertEqual(answer.stages[0]["status"], status)
                self.assertEqual(answer.status, "budget")
                self.assertFalse(answer.steps)
                self.assertIsNone(answer.suboptimality_bound)
                self.assertEqual(answer.generated, sum(stage["generated"] for stage in answer.stages))

    def test_reference_incumbent_survives_shared_clock_budget_without_a_fake_subset_bound(self):
        engine = with_direct_goal_filter(two_route_engine())
        clock, observed = [0.], [0]
        def observer(state, running, elapsed):
            observed[0] += 1
            clock[0] += .1 if observed[0] <= 3 else 3.7
        with patch.object(search_module, "monotonic", side_effect=lambda: clock[0]):
            answer = search_module.weighted_anytime(engine, bound=lambda state, running=(): 0.,
                max_expansions=1000, time_limit=4., observer=observer)
        self.assertEqual(answer.stages[0]["phase"], "reference")
        self.assertEqual(answer.stages[0]["status"], "solved")
        self.assertEqual(answer.stages[1]["status"], "budget")
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 9.)
        self.assertEqual(answer.expanded, 4)
        self.assertEqual(answer.elapsed, 4.)
        self.assertEqual(answer.lower_bound, 0.)
        self.assertIsNone(answer.suboptimality_bound)
        self.assertAlmostEqual(answer.stages[0]["elapsed"], .3)
        self.assertEqual(answer.stages[0]["time_budget"], 1.)

    def test_probe_no_plan_does_not_replace_the_full_action_conclusion(self):
        engine = two_route_engine()
        engine.hooks.reference_actions = lambda actions, initial, goal: frozenset()
        answer = search_module.weighted_anytime(engine, bound=admissible_two_route_bound,
                                                max_expansions=1000)
        self.assertEqual(answer.stages[0]["phase"], "reference")
        self.assertEqual(answer.stages[0]["status"], "no_plan")
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 5.)
        self.assertEqual(answer.weight, 1.)

    def test_probe_unknown_does_not_replace_the_full_valid_route(self):
        engine = tiny_engine([
            skill("unknown_route", pre=["free(?arm)"], add=["done(?arm)"], capability=["gate(?arm)"]),
            skill("known_route", pre=["free(?arm)"], add=["done(?arm)"])])
        engine.hooks.reference_actions = lambda actions, initial, goal: frozenset(
            action.key for action in actions if action.name == "unknown_route")
        answer = search_module.weighted_anytime(engine, bound=lambda state, running=(): 0.,
                                                max_expansions=1000)
        self.assertEqual(answer.stages[0]["phase"], "reference")
        self.assertEqual(answer.stages[0]["status"], "unknown")
        self.assertEqual(answer.status, "solved")
        self.assertEqual([step.action.name for step in answer.steps], ["known_route"])

    def test_probe_expansion_cap_and_all_restarts_share_the_original_budget(self):
        edges = [(f"chain_{index}", "s" if index == 0 else f"n{index-1}",
                  "g" if index == 149 else f"n{index}", 1.) for index in range(150)]
        edges.append(("shortcut", "s", "g", 5.))
        engine, _ = graph_engine(edges)
        engine.hooks.reference_actions = lambda actions, initial, goal: frozenset(
            action.key for action in actions if action.name != "shortcut")
        answer = search_module.weighted_anytime(engine, bound=lambda state, running=(): 0.,
                                                max_expansions=1000)
        probe = answer.stages[0]
        self.assertEqual(probe["phase"], "reference")
        self.assertEqual(probe["status"], "budget")
        self.assertEqual(probe["expanded"], 200)
        self.assertEqual(probe["expansion_budget"], 200)
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 5.)
        self.assertLessEqual(answer.expanded, 1000)
        self.assertEqual(answer.expanded, sum(stage["expanded"] for stage in answer.stages))

    def test_supplied_hint_still_intersects_the_domain_probe_filter(self):
        engine = two_route_engine()
        engine.hooks.reference_actions = lambda actions, initial, goal: frozenset(action.key for action in actions)
        hint = {("cheap_prefix", ("A",)), ("cheap_finish", ("A",))}
        exact = search_module._search(engine, hint=hint, bound=admissible_two_route_bound)
        answer = search_module.weighted_anytime(engine, hint=hint, bound=admissible_two_route_bound,
                                                max_expansions=1000)
        probe = answer.stages[0]
        self.assertEqual(probe["weight"], 3.)
        self.assertEqual(probe["action_scope"], "goal_domain_and_hint")
        self.assertEqual(probe["allowed_actions"], 2)
        self.assertEqual(probe["makespan"], 5.)
        self.assertEqual(answer.steps, exact.steps)
        self.assertEqual(answer.weight, 1.)

    def test_running_effects_and_caller_restrictions_are_kept_in_the_probe(self):
        engine = work_engine(status="UNKNOWN")
        action = engine.parse_action("work(A)")
        engine.hooks.reference_actions = lambda actions, initial, goal: frozenset(a.key for a in actions)
        answer = search_module.weighted_anytime(engine, start=engine.start(action, engine.initial),
            goal=[parse_condition("done(A)")], allowed=iter([]), protected=iter([parse_fact("free(A)")]),
            initial_running=iter([(action.key, 1.25)]),
            bound=lambda state, running=(): max((remaining for _, remaining in running), default=0.),
            max_expansions=1000)
        self.assertEqual(answer.stages[0]["phase"], "reference")
        self.assertEqual(answer.stages[0]["allowed_actions"], 0)
        self.assertEqual(answer.status, "solved")
        self.assertEqual(answer.makespan, 1.25)
        self.assertEqual(answer.weight, 1.)
        self.assertFalse(answer.unknown)


if __name__ == "__main__":
    unittest.main()
