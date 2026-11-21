"""Optional reference bounds preserve the exact baseline result and limits."""

import unittest

from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data
from skillnet.core.logic import parse_condition, parse_fact
from skillnet.domains.blocks import load_problem
from skillnet.planning.bounds import HMaxBound
from skillnet.planning.heuristic import build_heuristic
from skillnet.planning.search import _search, search
from tests.test_search import work_engine


class LargeActionMap(dict):
    def __len__(self):
        return 128


def bounded_engine(**kwargs):
    engine = work_engine(**kwargs)
    engine.actions = LargeActionMap(engine.actions)
    engine.hooks.reference_actions = lambda actions, initial, goal: frozenset(a.key for a in actions)
    return engine


class BoundedSearchTests(unittest.TestCase):
    def test_reference_wrapper_preserves_pure_ucs_optimum(self):
        engine = bounded_engine()
        pure = _search(engine)
        for hint in (None, frozenset(engine.actions)):
            with self.subTest(hint=hint):
                bounded = search(engine, hint=hint)
                self.assertEqual(bounded.status, pure.status)
                self.assertEqual(bounded.makespan, pure.makespan)
                self.assertGreater(bounded.expanded, 0)
                self.assertLessEqual(bounded.expanded, 200_000)

    def test_no_hint_search_uses_a_nonzero_bound_with_a_domain_reference_probe(self):
        engine = bounded_engine()
        calls = []

        def domain_reference_probe(actions, initial, goal):
            calls.append((initial, goal))
            return frozenset(action.key for action in actions)

        engine.hooks.reference_actions = domain_reference_probe
        self.assertGreater(build_heuristic(engine)(engine.initial), 0.0)
        result = search(engine)
        self.assertEqual(calls, [(engine.initial, engine.goal)])
        self.assertEqual(result.stages[0]["phase"], "reference")
        self.assertEqual(result.stages[0]["action_scope"], "goal_domain")
        self.assertEqual(result.stages[0]["weight"], 3.)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 2.0)

    def test_slower_hint_does_not_lock_out_faster_arm_or_custom_cost(self):
        class ArmCost:
            def estimate(self, action, state):
                return 1.0 if action.args[0] == "A" else 3.0

        engine = bounded_engine(shared_goal=True)
        engine.cost = ArmCost()
        hint = {("work", ("B",))}
        pure = _search(engine, hint=hint)
        bounded = search(engine, hint=hint)
        self.assertEqual(bounded.status, "solved")
        self.assertEqual(bounded.makespan, pure.makespan)
        self.assertEqual(bounded.makespan, 1.0)
        self.assertEqual(bounded.steps[0].action.args, ("A",))

    def test_infeasible_reference_falls_back_to_full_action_set(self):
        engine = bounded_engine()
        engine.hooks.reference_actions = lambda actions, initial, goal: frozenset()
        result = search(engine, hint=frozenset(engine.actions))
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 2.0)

    def test_cut_protected_and_allowed_restrictions_survive_probe(self):
        for restriction in ("cuts", "protected", "allowed"):
            with self.subTest(restriction=restriction):
                engine = bounded_engine(shared_goal=True)
                key = ("work", ("A",))
                arguments = {"allowed": {key}}
                if restriction == "cuts":
                    arguments["cuts"] = {(engine.initial, key)}
                elif restriction == "protected":
                    arguments["protected"] = {parse_fact("free(A)")}
                else:
                    arguments["allowed"] = set()
                pure = _search(engine, **arguments)
                result = search(engine, **arguments)
                self.assertEqual(result.status, pure.status)
                self.assertEqual(result.status, "no_plan")

    def test_small_and_expired_budgets_are_preserved(self):
        engine = bounded_engine()
        for options in ({"max_expansions": 0}, {"max_expansions": 1}, {"time_limit": 0.0}):
            with self.subTest(options=options):
                result = search(engine, **options)
                self.assertEqual(result.status, "budget")
                if "max_expansions" in options:
                    self.assertLessEqual(result.expanded, options["max_expansions"])

    def test_iterable_restrictions_are_not_consumed_by_reference_probe(self):
        for restriction in ("cuts", "protected", "allowed"):
            with self.subTest(restriction=restriction):
                engine = bounded_engine(shared_goal=True)
                key = ("work", ("A",))
                arguments = {"allowed": {key}}
                expected = "no_plan"
                if restriction == "cuts":
                    arguments["cuts"] = iter([(engine.initial, key)])
                elif restriction == "protected":
                    arguments["protected"] = iter([parse_fact("free(A)")])
                else:
                    arguments["allowed"] = iter([key])
                    expected = "solved"
                self.assertEqual(search(engine, **arguments).status, expected)

    def test_probe_expansions_count_against_total_budget(self):
        engine = bounded_engine()
        skills = []
        for index in range(400):
            skills.append({"name": f"step_{index}", "params": {"arm": "arm"},
                           "pre": ["free(?arm)", f"phase_{index}(?arm)"],
                           "delete": ["free(?arm)", f"phase_{index}(?arm)"],
                           "add": ["free(?arm)", f"phase_{index + 1}(?arm)"],
                           "occupies": ["?arm"], "op_time": 1.0})
        engine.problem.library = library_from_data({"skill": skills})
        engine.problem.objects["arm"] = ("A",)
        engine.problem.initial |= {parse_fact("phase_0(A)")}
        engine.problem.goal = (parse_condition("phase_400(A)"),)
        engine = Engine(engine.problem)
        # A long valid hint lets reference work consume part of the budget.
        # The observable total must still include that preparatory search.
        result = search(engine, hint=frozenset(engine.actions), max_expansions=1000)
        self.assertIn(result.status, ("solved", "budget"))
        self.assertLessEqual(result.expanded, 1000)
        if result.status == "solved":
            self.assertEqual(result.makespan, 400.0)

    def test_coarse_bound_does_not_overestimate_running_effects(self):
        engine = bounded_engine()
        first = engine.parse_action("work(A)")
        started = engine.start(first, engine.initial)
        bound = HMaxBound(engine, coarse=True)
        self.assertAlmostEqual(bound(started, ((first.key, 1.0),)), 2.0)
        self.assertEqual(bound(started, ((first.key, 7.0),)), 7.0)

    def test_rounded_durations_do_not_make_bound_exclude_reference_solution(self):
        engine = bounded_engine()
        skills = []
        for index in range(30):
            previous = f"phase_{index}(?arm)"
            following = f"phase_{index + 1}(?arm)"
            skills.append({"name": f"step_{index}", "params": {"arm": "arm"},
                           "pre": ["free(?arm)", previous],
                           "delete": ["free(?arm)", previous],
                           "add": ["free(?arm)", following], "occupies": ["?arm"],
                           "op_time": 1.0000000004})
        engine.problem.library = library_from_data({"skill": skills})
        engine.problem.objects["arm"] = ("A",)
        engine.problem.initial |= {parse_fact("phase_0(A)")}
        engine.problem.goal = (parse_condition("phase_30(A)"),)
        engine = Engine(engine.problem)
        engine.actions = LargeActionMap(engine.actions)
        pure = _search(engine)
        self.assertEqual(pure.status, "solved")
        bounded = search(engine)
        self.assertEqual(bounded.status, "solved")
        self.assertEqual(bounded.makespan, pure.makespan)

    def test_running_move_already_removes_negative_arm_position_goal(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        goal = (parse_condition("not arm_at(A, home_A)"),)
        action = engine.parse_action("pick(A, red, stock_c)")
        started = engine.start(action, engine.initial)
        for coarse in (False, True):
            with self.subTest(coarse=coarse):
                bound = HMaxBound(engine, goal=goal, coarse=coarse)
                self.assertEqual(bound(started, ((action.key, 0.5),)), 0.5)
        engine.actions = LargeActionMap(engine.actions)
        pure = _search(engine, goal=goal)
        bounded = search(engine, goal=goal)
        self.assertEqual(bounded.status, "solved")
        self.assertEqual(bounded.makespan, pure.makespan)


if __name__ == "__main__":
    unittest.main()
