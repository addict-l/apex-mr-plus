"""Event-state UCS and distance-cost boundary checks."""

from copy import deepcopy
import unittest

from skillnet.capability.oracle import CapabilityOracle
from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data
from skillnet.core.logic import Fact, parse_condition, parse_fact
from skillnet.core.problem import Problem
from skillnet.cost.model import DistanceCost
from skillnet.planning.search import search


WORK_SKILL = {
    "name": "work",
    "params": {"arm": "arm"},
    "pre": ["free(?arm)", "not done(?arm)"],
    "capability": ["able(?arm)"],
    "delete": ["free(?arm)"],
    "add": ["done(?arm)", "free(?arm)"],
    "occupies": ["?arm"],
    "op_time": 2.0,
}


class TestHooks:
    derived_predicates = frozenset()
    derived_basis = {}
    resource_predicates = frozenset({"free", "arm_at"})

    def __init__(self):
        self.positions = {"home_A": (0., 0., 0.), "home_B": (0., 6., 0.),
                          "near": (0., 2., 0.), "far": (0., 8., 0.)}
        self.speeds = {"A": 2.0, "B": 1.0}

    def invariant(self, state):
        return None

    def position(self, place):
        return self.positions[place]

    def speed(self, arm):
        return self.speeds[arm]


def work_engine(*, structural=False, status="VALID", hooks=None, shared_goal=False):
    raw = deepcopy(WORK_SKILL)
    raw["structural"] = structural
    goal = ["done(A)", "done(B)"]
    if shared_goal:
        raw["pre"][-1] = "not done()"
        raw["add"][0] = "done()"
        goal = ["done()"]
    hooks = hooks or TestHooks()
    problem = Problem(
        "events", "事件搜索测试", {"arm": ("A", "B"), "block": (), "loc": (), "slot": ()},
        frozenset(), frozenset(map(parse_fact, ["free(A)", "free(B)",
                                              "arm_at(A, home_A)", "arm_at(B, home_B)"])),
        tuple(map(parse_condition, goal)), library_from_data({"skill": [raw]}), hooks,
        CapabilityOracle({"able": {"default": status}}),
    )
    return Engine(problem, DistanceCost(hooks))


class SearchBoundaryTests(unittest.TestCase):
    def test_parallel_work_and_structural_mutex(self):
        parallel = search(work_engine())
        self.assertEqual(parallel.status, "solved")
        self.assertEqual(parallel.makespan, 2.0)
        self.assertEqual(len(parallel.steps), 2)
        self.assertEqual({step.start for step in parallel.steps}, {0.0})
        self.assertEqual({step.end for step in parallel.steps}, {2.0})
        serial = search(work_engine(structural=True))
        self.assertEqual(serial.status, "solved")
        self.assertEqual(serial.makespan, 4.0)
        self.assertEqual([step.start for step in serial.steps], [0.0, 2.0])

    def test_atomic_simultaneous_finish(self):
        class BothOrNeither(TestHooks):
            def invariant(self, state):
                done_a = parse_fact("done(A)") in state
                done_b = parse_fact("done(B)") in state
                return "两个结果必须同时出现" if done_a != done_b else None

        result = search(work_engine(hooks=BothOrNeither()))
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 2.0)

    def test_unknown_invalid_and_explicit_budget(self):
        unknown = search(work_engine(status="UNKNOWN"))
        self.assertEqual(unknown.status, "unknown")
        self.assertEqual(set(unknown.unknown), {parse_fact("able(A)"), parse_fact("able(B)")})
        self.assertGreater(unknown.rejected["capability_unknown"], 0)
        invalid = search(work_engine(status="INVALID"))
        self.assertEqual(invalid.status, "no_plan")
        self.assertGreater(invalid.rejected["capability_invalid"], 0)
        self.assertEqual(search(work_engine(), max_expansions=0).status, "budget")
        self.assertEqual(search(work_engine(), time_limit=0).status, "budget")

    def test_cuts_protected_and_allowed(self):
        engine = work_engine()
        goal = [parse_condition("done(A)")]
        key = ("work", ("A",))
        common = {"goal": goal, "allowed": {key}}
        cut = search(engine, cuts={(engine.initial, key)}, **common)
        self.assertEqual(cut.status, "no_plan")
        self.assertGreater(cut.rejected["cut"], 0)
        protected = search(engine, protected={parse_fact("free(A)")}, **common)
        self.assertEqual(protected.status, "no_plan")
        self.assertGreater(protected.rejected["protected"], 0)
        self.assertEqual(search(engine, allowed=set()).status, "no_plan")

    def test_hint_only_breaks_equal_cost_ties(self):
        engine = work_engine(shared_goal=True)
        result = search(engine, hint={("work", ("B",))})
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.steps[0].action.key, ("work", ("B",)))

        class UnequalCost:
            def estimate(self, action, state):
                return 1.0 if action.args[0] == "A" else 2.0

        engine.cost = UnequalCost()
        faster = search(engine, hint={("work", ("B",))})
        self.assertEqual(faster.makespan, 1.0)
        self.assertEqual(faster.steps[0].action.key, ("work", ("A",)))

    def test_busy_arm_cannot_start_another_action_without_free_precondition(self):
        engine = work_engine()
        skill_a = deepcopy(WORK_SKILL)
        skill_b = deepcopy(WORK_SKILL)
        skill_a.update(name="first", pre=["not first_done(?arm)"], delete=[],
                       add=["first_done(?arm)"])
        skill_b.update(name="second", pre=["not second_done(?arm)"], delete=[],
                       add=["second_done(?arm)"])
        engine.problem.library = library_from_data({"skill": [skill_a, skill_b]})
        engine.problem.objects["arm"] = ("A",)
        engine.problem.goal = tuple(map(parse_condition, ["first_done(A)", "second_done(A)"]))
        engine = Engine(engine.problem, engine.cost)
        result = search(engine)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 4.0)
        self.assertEqual([step.start for step in result.steps], [0.0, 2.0])

    def test_action_cost_uses_position_at_actual_start(self):
        engine = work_engine()
        near = deepcopy(WORK_SKILL)
        near.update(name="approach", params={"arm": "arm", "destination": "loc"},
                    pre=["free(?arm)", "near_target(?destination)", "not reached(?arm)"],
                    add=["reached(?arm)", "free(?arm)"], move_to={"?arm": "?destination"})
        far = deepcopy(near)
        far.update(name="finish", pre=["free(?arm)", "reached(?arm)",
                                       "far_target(?destination)", "not done(?arm)"],
                   add=["done(?arm)", "free(?arm)"])
        engine.problem.library = library_from_data({"skill": [near, far]})
        engine.problem.objects["arm"] = ("A",)
        engine.problem.objects["loc"] = ("near", "far")
        engine.problem.static = frozenset(map(parse_fact, ["near_target(near)", "far_target(far)"]))
        engine.problem.goal = (parse_condition("done(A)"),)
        engine = Engine(engine.problem, engine.cost)
        result = search(engine)
        self.assertEqual(result.status, "solved")
        self.assertEqual([(step.start, step.end) for step in result.steps], [(0., 3.), (3., 8.)])
        self.assertIn(parse_fact("arm_at(A, near)"), result.steps[1].state)

    def test_start_invariant_and_initial_goal(self):
        class KeepArmsFree(TestHooks):
            def invariant(self, state):
                return None if parse_fact("free(A)") in state else "测试中 A 必须空闲"

        result = search(work_engine(hooks=KeepArmsFree()))
        self.assertEqual(result.status, "no_plan")
        self.assertGreater(result.rejected["invariant"], 0)
        engine = work_engine()
        result = search(engine, goal=[])
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 0)
        self.assertEqual(result.steps, [])


class DistanceCostTests(unittest.TestCase):
    def test_position_dependent_single_and_two_arm_motion(self):
        raw = deepcopy(WORK_SKILL)
        raw["params"]["destination"] = "loc"
        raw["move_to"] = {"?arm": "?destination"}
        action = library_from_data({"skill": [raw]}).skills[0].ground(("A", "far"))
        hooks = TestHooks()
        cost = DistanceCost(hooks)
        state = frozenset({parse_fact("arm_at(A, home_A)")})
        self.assertEqual(cost.estimate(action, state), 6.0)
        closer = frozenset({parse_fact("arm_at(A, near)")})
        self.assertEqual(cost.estimate(action, closer), 5.0)
        self.assertIsNone(cost.evaluate(action, state))
        raw["params"]["helper"] = "arm"
        raw["occupies"].append("?helper")
        raw["move_to"]["?helper"] = "?destination"
        pair = library_from_data({"skill": [raw]}).skills[0].ground(("A", "far", "B"))
        pair_state = state | {parse_fact("arm_at(B, home_B)")}
        self.assertEqual(cost.estimate(pair, pair_state), 6.0)
        hooks.speeds["B"] = 0.25
        self.assertEqual(cost.estimate(pair, pair_state), 10.0)

    def test_stationary_duration_and_invalid_position(self):
        action = library_from_data({"skill": [deepcopy(WORK_SKILL)]}).skills[0].ground(("A",))
        cost = DistanceCost(TestHooks())
        self.assertEqual(cost.estimate(action, frozenset()), 2.0)
        raw = deepcopy(WORK_SKILL)
        raw["params"]["destination"] = "loc"
        raw["move_to"] = {"?arm": "?destination"}
        moving = library_from_data({"skill": [raw]}).skills[0].ground(("A", "far"))
        with self.assertRaisesRegex(ValueError, "arm_at"):
            cost.estimate(moving, frozenset())


if __name__ == "__main__":
    unittest.main()
