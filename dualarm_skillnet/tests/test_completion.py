import unittest

from skillnet.core.engine import Engine
from skillnet.core.logic import parse_fact
from skillnet.domains.blocks import load_problem
from skillnet.planning.completion import complete


class CompletionTests(unittest.TestCase):
    def replay(self, engine, report):
        state = engine.initial
        for action in report.plan:
            self.assertFalse(engine.failed(action, state), str(action))
            state = engine.execute(action, state)
        return state

    def test_handover_proposal_deletes_unreachable_place_and_repairs_goal(self):
        engine = Engine(load_problem("tasks/s1_reach_handover.toml"))
        report = complete(engine, engine.problem.proposal.steps)
        rejected = [entry for entry in report.entries if entry.kind == "删除"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].action_text, "place(B, red, board_l1)")
        self.assertIn("reach(B, board_l1)", rejected[0].reason)
        self.assertIn("INVALID", rejected[0].reason)
        self.assertTrue(report.goal_reached)
        self.assertFalse(report.unresolved)
        self.assertFalse(report.feedback_for_model)
        self.assertTrue(any(entry.kind == "补出" and "handover" in entry.action_text for entry in report.entries))
        self.assertTrue(engine.satisfied(engine.goal, self.replay(engine, report)))

    def test_tower_repairs_lower_layers_then_marks_original_steps_redundant(self):
        engine = Engine(load_problem("tasks/s2_tower.toml"))
        report = complete(engine, engine.problem.proposal.steps)
        kept_upper = [entry for entry in report.entries if entry.kind == "保留"
                      and entry.action_text == "insert(B, b3, t3)"]
        self.assertEqual(len(kept_upper), 1)
        upper_index = report.entries.index(kept_upper[0])
        inserted_before = [entry.action_text for entry in report.entries[:upper_index] if entry.kind == "补出"]
        self.assertIn("insert(A, b1, t1)", inserted_before)
        self.assertIn("insert(A, b2, t2)", inserted_before)
        redundant = [entry.action_text for entry in report.entries if entry.kind == "多余"]
        self.assertEqual(redundant, engine.problem.proposal.steps[2:])
        self.assertTrue(report.goal_reached)
        self.assertTrue(engine.satisfied(engine.goal, self.replay(engine, report)))

    def test_invalid_steps_report_errors_without_corrupting_plan(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        proposal = ["broken", "unknown(A)", "pick(A)", "pick(A, ghost, stock_c)",
                    "pick(stock_c, red, stock_c)"]
        for text in proposal:
            with self.subTest(action=text):
                report = complete(engine, [text])
                self.assertEqual(report.entries[0].kind, "删除")
                self.assertFalse(report.abandoned)
                self.assertTrue(report.goal_reached)
                self.assertTrue(engine.satisfied(engine.goal, self.replay(engine, report)))

    def test_unreachable_goal_produces_model_feedback(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        problem.oracle.update(parse_fact("reach(A, board_l1)"), "INVALID", "测试")
        engine = Engine(problem)
        report = complete(engine, problem.proposal.steps)
        self.assertFalse(report.goal_reached)
        self.assertIn("at(red, board_l1)", report.unresolved)
        self.assertTrue(any("第 2 步" in text and "INVALID" in text for text in report.feedback_for_model))

    def test_unknown_capability_deletes_step_with_evidence(self):
        problem = load_problem("tasks/s1_nearest_arm.toml")
        problem.oracle.update(parse_fact("reach(A, stock_c)"), "UNKNOWN", "测试未验证")
        report = complete(Engine(problem), ["pick(A, red, stock_c)"])
        self.assertEqual(report.entries[0].kind, "删除")
        self.assertIn("UNKNOWN", report.entries[0].reason)
        self.assertIn("reach(A, stock_c)", report.entries[0].reason)
        self.assertTrue(report.goal_reached)

    def test_missing_precondition_is_filled_before_original_step(self):
        engine = Engine(load_problem("tasks/s1_nearest_arm.toml"))
        report = complete(engine, ["place(A, red, board_c)"])
        self.assertEqual([entry.kind for entry in report.entries], ["补出", "保留"])
        self.assertEqual(report.plan[0].name, "pick")
        self.assertTrue(report.goal_reached)


if __name__ == "__main__":
    unittest.main()
