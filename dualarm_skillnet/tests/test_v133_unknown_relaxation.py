"""Conservative early UNKNOWN decisions at the public search boundary."""
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

from skillnet.capability.oracle import CapabilityOracle
from skillnet.core.engine import Engine
from skillnet.core.library import library_from_data
from skillnet.core.logic import Fact, parse_condition, parse_fact
from skillnet.core.problem import Problem
from skillnet.domains.blocks import load_problem
from skillnet.planning.search import _search, search
from tests.test_search import TestHooks, work_engine


ROOT = Path(__file__).resolve().parents[1]


def tiny_engine(skills, *, initial=("free(A)",), goal=("done(A)",), capabilities=None, hooks=None):
    problem = Problem("relaxed", "松弛可达性反例", {"arm": ("A",), "block": (), "loc": (), "slot": ()},
        frozenset(), frozenset(map(parse_fact, initial)), tuple(map(parse_condition, goal)),
        library_from_data({"skill": skills}), hooks or TestHooks(), CapabilityOracle(capabilities))
    return Engine(problem)


def skill(name, *, pre=(), add=(), delete=(), capability=()):
    return {"name": name, "params": {"arm": "arm"}, "pre": list(pre), "add": ["free(?arm)", *add],
            "delete": list(delete), "capability": list(capability), "occupies": ["?arm"], "op_time": 1.}


class UnknownRelaxationTests(unittest.TestCase):
    def test_old_generated_unknown_goal_is_decided_before_two_seconds(self):
        script = """import json
from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.planning.search import search
engine = Engine(load_problem('results/sim_v1_3/overnight/generated/gen_f1_s20261115_000.toml'))
answer = search(engine)
print(json.dumps({'status': answer.status, 'unknown': sorted(map(str, answer.unknown)), 'expanded': answer.expanded, 'elapsed': answer.elapsed}))
"""
        began = time.monotonic()
        try:
            process = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=ROOT,
                capture_output=True, text=True, timeout=2., check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            self.fail("旧生成任务未能在 2 秒内返回 UNKNOWN")
        answer = json.loads(process.stdout)
        self.assertEqual(answer["status"], "unknown")
        self.assertIn("reach(B, board_2)", answer["unknown"])
        self.assertEqual(answer["expanded"], 0)
        self.assertLess(time.monotonic()-began, 2.)

    def test_only_unknowns_supporting_the_goal_are_reported(self):
        engine = tiny_engine([
            skill("unlock", pre=["free(?arm)"], add=["token(?arm)"], capability=["gate(?arm)"]),
            skill("finish", pre=["token(?arm)"], add=["done(?arm)"], capability=["destination(?arm)"]),
            skill("irrelevant", add=["noise(?arm)"], capability=["irrelevant(?arm)"])])
        result = search(engine)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.unknown, frozenset(map(parse_fact, ("gate(A)", "destination(A)"))))
        self.assertEqual(result.expanded, 0)

    def test_invalid_conjunct_cannot_be_restored_by_unknown_relaxation(self):
        engine = tiny_engine([skill("finish", add=["done(?arm)"], capability=["maybe(?arm)", "blocked(?arm)"])],
            capabilities={"blocked": {"default": "INVALID"}})
        result = search(engine)
        self.assertEqual(result.status, "no_plan")
        self.assertFalse(result.unknown)
        self.assertEqual(result.expanded, 0)

    def test_custom_start_and_fact_goal_ignore_unneeded_unknown_arm(self):
        engine = work_engine(status="UNKNOWN")
        result = search(engine, goal=[Fact("done", ("A",))])
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.unknown, frozenset({parse_fact("able(A)")}))
        started = engine.initial | {parse_fact("done(A)")}
        result = search(engine, start=started, goal=[parse_condition("done(A)")])
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 0.)

    def test_allowed_iterators_and_protected_deletes_are_respected(self):
        engine = work_engine(status="UNKNOWN")
        goal = [parse_condition("done(B)")]
        self.assertEqual(search(engine, goal=goal, allowed=iter([("work", ("A",))])).status, "no_plan")
        result = search(engine, goal=goal, allowed=iter([("work", ("B",))]))
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.unknown, frozenset({parse_fact("able(B)")}))
        result = search(engine, goal=goal, protected=iter([parse_fact("free(B)")]))
        self.assertEqual(result.status, "no_plan")
        self.assertGreater(result.rejected["protected"], 0)

    def test_running_effect_does_not_need_capability_allowed_or_protected_again(self):
        engine = work_engine(status="UNKNOWN")
        action = engine.parse_action("work(A)")
        result = _search(engine, start=engine.start(action, engine.initial),
            goal=[parse_condition("done(A)")], initial_running=((action.key, 1.25),),
            allowed=set(), protected={parse_fact("free(A)")})
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 1.25)
        self.assertFalse(result.unknown)

    def test_running_move_supplies_positive_and_negative_implicit_arm_effects(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        action = engine.parse_action("pick(A, red, stock_c)")
        for goal in (("arm_at(A, stock_c)",), ("not arm_at(A, home_A)",)):
            with self.subTest(goal=goal):
                result = _search(engine, start=engine.start(action, engine.initial),
                    goal=map(parse_condition, goal), initial_running=((action.key, .5),), allowed=set())
                self.assertEqual(result.status, "solved")
                self.assertEqual(result.makespan, .5)

    def test_negative_goal_supports_unknown_delete_and_concurrent_positive_effect(self):
        engine = tiny_engine([skill("remove", pre=["present(?arm)"], delete=["present(?arm)"],
            add=["done(?arm)"], capability=["able(?arm)"])], initial=["present(A)"],
            goal=["not present(A)", "done(A)"])
        result = search(engine)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.unknown, frozenset({parse_fact("able(A)")}))
        engine.oracle.update(parse_fact("able(A)"), "VALID", "测试资格确认")
        result = search(engine)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 1.)

    def test_arbitrary_nonmonotone_derived_conditions_are_fully_relaxed(self):
        class Nonmonotone(TestHooks):
            derived_predicates = frozenset({"clear"})
            def evaluate(self, predicate, args, state):
                return parse_fact("present(A)") not in state
            def expand(self, fact):
                return []
        engine = tiny_engine([
            skill("remove", delete=["present(?arm)"]),
            skill("finish", pre=["clear(?arm)"], add=["done(?arm)"])],
            initial=["present(A)"], hooks=Nonmonotone())
        result = search(engine)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 2.)

    def test_state_specific_cut_does_not_remove_action_in_other_states(self):
        engine = tiny_engine([
            skill("prepare", add=["prepared(?arm)"]),
            skill("finish", add=["done(?arm)"])])
        key = engine.parse_action("finish(A)").key
        result = search(engine, cuts={(engine.initial, key)})
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 2.)
        self.assertEqual([step.action.name for step in result.steps], ["prepare", "finish"])

    def test_negative_preconditions_can_be_enabled_by_a_valid_delete(self):
        engine = tiny_engine([
            skill("remove", delete=["present(?arm)"]),
            skill("finish", pre=["not present(?arm)"], add=["done(?arm)"])], initial=["present(A)"])
        result = search(engine)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.makespan, 2.)

    def test_impossible_goal_is_no_plan_despite_irrelevant_unknowns(self):
        engine = tiny_engine([skill("irrelevant", add=["noise(?arm)"], capability=["maybe(?arm)"])])
        result = search(engine)
        self.assertEqual(result.status, "no_plan")
        self.assertFalse(result.unknown)

    def test_zero_budgets_still_take_precedence_over_unknown_precheck(self):
        engine = work_engine(status="UNKNOWN")
        self.assertEqual(search(engine, max_expansions=0).status, "budget")
        self.assertEqual(search(engine, time_limit=0).status, "budget")


if __name__ == "__main__":
    unittest.main()
