"""Regressions for capability invalidation, nominal reuse and full search."""

import unittest

from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from skillnet.domains.blocks import load_problem
from skillnet.planning.recovery import RecoveryPlanner
from skillnet.planning.search import weighted_anytime
from tests.test_bounded_search import bounded_engine


class IncrementalRecoveryTests(unittest.TestCase):
    def test_insert_failure_reassigns_from_the_measured_state(self):
        problem = load_problem("tasks/s2_tower.toml")
        problem.oracle.update(parse_fact("reach(A, stock_r1)"), Status.VALID, "test")
        nominal = [
            "pick(A, b1, stock_l1)", "insert(A, b1, t1)",
            "pick(A, b2, stock_l2)", "insert(A, b2, t2)",
            "pick(B, b3, stock_r1)", "insert(B, b3, t3)",
        ]
        planner = RecoveryPlanner(problem, nominal)
        state = problem.initial
        for step in nominal[:2]:
            state = planner.engine.execute(planner.engine.parse_action(step), state)
        kwargs = dict(disabled_skills=[("B", "insert")], completed_steps=[0, 1])
        recovered = planner.replan(state, **kwargs)
        cold = planner.replan(state, incremental=False, **kwargs)

        self.assertEqual(recovered.schedule.status, "solved")
        self.assertEqual(recovered.schedule.makespan, cold.schedule.makespan)
        self.assertEqual(recovered.schedule.suboptimality_bound, 1.)
        self.assertEqual(recovered.schedule.stages[0]["phase"], "repair")
        self.assertEqual(recovered.schedule.stages[0]["status"], "solved")
        self.assertTrue(recovered.excluded)
        self.assertTrue(all(name == "insert" and args[0] == "B"
                            for name, args in recovered.excluded))
        self.assertTrue(all(step.action.args[-1] != "t1" for step in recovered.schedule.steps))
        self.assertIn(parse_fact("filled(t1)"), recovered.schedule.steps[0].state)
        self.assertIsNotNone(recovered.graph)

    def test_exact_fact_updates_can_disable_and_enable_candidates(self):
        problem = load_problem("tasks/s1_nearest_arm.toml")
        a_board = parse_fact("reach(A, board_c)")
        b_board = parse_fact("reach(B, board_c)")
        problem.oracle.update(b_board, Status.INVALID, "test")
        planner = RecoveryPlanner(problem, ["pick(A, red, stock_c)", "place(A, red, board_c)"])
        result = planner.replan(problem.initial,
                                capabilities={a_board: Status.INVALID, b_board: Status.VALID})
        self.assertEqual(result.schedule.status, "solved")
        self.assertIn(("place", ("A", "red", "board_c")), result.excluded)
        self.assertIn("place(B, red, board_c)", {str(s.action) for s in result.schedule.steps})
        self.assertEqual(problem.oracle.check(a_board).status, Status.VALID)
        self.assertEqual(problem.oracle.check(b_board).status, Status.INVALID)

    def test_unknown_capability_is_not_pruned_as_invalid(self):
        problem = load_problem("tasks/s1_nearest_arm.toml")
        planner = RecoveryPlanner(problem, ["pick(A, red, stock_c)", "place(A, red, board_c)"])
        facts = [parse_fact("reach(A, board_c)"), parse_fact("reach(B, board_c)")]
        result = planner.replan(problem.initial, capabilities=dict.fromkeys(facts, Status.UNKNOWN))
        self.assertEqual(result.schedule.status, "unknown")
        self.assertEqual(result.excluded, frozenset())
        self.assertTrue(result.schedule.unknown)
        self.assertLessEqual(result.schedule.unknown, set(facts))

    def test_repair_is_only_an_upper_bound_and_can_be_improved(self):
        engine = bounded_engine(shared_goal=True)

        class ArmCost:
            def estimate(self, action, state):
                return 1. if action.args[0] == "A" else 3.

        engine.cost = ArmCost()
        result = weighted_anytime(engine, repair_actions={("work", ("B",))})
        self.assertEqual(result.stages[0]["phase"], "repair")
        self.assertEqual(result.stages[0]["makespan"], 3.)
        self.assertEqual(result.makespan, 1.)
        self.assertEqual(result.suboptimality_bound, 1.)

    def test_failed_repair_scope_falls_back_to_all_enabled_actions(self):
        engine = bounded_engine(shared_goal=True)
        result = weighted_anytime(engine, repair_actions={("work", ("B",))},
                                  cuts={(engine.initial, ("work", ("B",)))})
        self.assertEqual(result.stages[0]["phase"], "repair")
        self.assertNotEqual(result.stages[0]["status"], "solved")
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.steps[0].action.args, ("A",))
