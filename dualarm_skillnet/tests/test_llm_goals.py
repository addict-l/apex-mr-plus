"""Natural task instructions and diagnostic-only model goal comparison."""

from pathlib import Path
import tomllib
import unittest

from skillnet.core.logic import parse_condition
from skillnet.core.problem import Proposal
from skillnet.decompose.prompt import build_prompt
from skillnet.domains.blocks import load_problem, problem_from_data


def task_data(name="s1_reach_handover"):
    with open(f"tasks/{name}.toml", "rb") as source:
        return tomllib.load(source)


class TaskInstructionTests(unittest.TestCase):
    def test_loader_keeps_natural_instruction_and_explicit_goals_separate(self):
        data = task_data()
        data["instruction"] = "把红色积木放到底板左边那个位置。"
        problem = problem_from_data(data)
        self.assertEqual(problem.instruction, data["instruction"])
        self.assertEqual(problem.explicit_goal, (parse_condition("at(red, board_l1)"),))
        self.assertIn(parse_condition("free(A)"), problem.goal)
        self.assertNotIn(parse_condition("free(A)"), problem.explicit_goal)

    def test_instruction_has_string_validation_and_legacy_default(self):
        data = task_data()
        data.pop("instruction", None)
        self.assertEqual(problem_from_data(data).instruction, "")
        data["instruction"] = ["不是字符串"]
        with self.assertRaisesRegex(ValueError, "instruction.*字符串"):
            problem_from_data(data)

    def test_prompt_uses_instruction_without_description_or_proposal_hints(self):
        data = task_data()
        data["instruction"] = "把红色积木放到底板左边那个位置。"
        data["description"] = "泄漏信息：B能够取到，A才能放下"
        data["proposal"]["instruction"] = "泄漏方案：先交接再放下"
        problem = problem_from_data(data)
        for mode in ("minimal", "with_capabilities"):
            with self.subTest(mode=mode):
                prompt = build_prompt(problem, mode)
                self.assertIn(data["instruction"], prompt)
                self.assertNotIn(data["description"], prompt)
                self.assertNotIn(data["proposal"]["instruction"], prompt)

    def test_missing_instruction_falls_back_to_description(self):
        data = task_data()
        data.pop("instruction", None)
        data["description"] = "把红色积木搬到底板左区"
        data["proposal"]["instruction"] = "不应使用这条旧提案指令"
        prompt = build_prompt(problem_from_data(data), "minimal")
        self.assertIn(data["description"], prompt)
        self.assertNotIn(data["proposal"]["instruction"], prompt)

    def test_every_task_has_a_goal_only_instruction_and_matching_pair(self):
        for path in sorted(Path("tasks").glob("*.toml")):
            with self.subTest(task=path.name), path.open("rb") as source:
                data = tomllib.load(source)
                instruction = data.get("instruction")
                self.assertIsInstance(instruction, str)
                self.assertTrue(instruction.strip())
                for solution in ("够得着", "够不着", "能到达", "可达", "交接", "托住", "搭台", "搭建", "支撑"):
                    self.assertNotIn(solution, instruction)
                if path.stem.endswith("_with_proposal"):
                    base = task_data(path.stem.removesuffix("_with_proposal"))
                    self.assertEqual(instruction, base["instruction"])


class ModelGoalComparisonTests(unittest.TestCase):
    def compare(self, problem, goals):
        from skillnet.decompose.goals import compare_model_goals
        return compare_model_goals(problem, Proposal(goal=goals, source="测试模型"))

    def test_coverage_normalizes_and_ignores_implicit_goals_as_extras(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        original = problem.goal
        result = self.compare(problem, [" at( red,board_l1 ) ", "free(A)", "free(B)"])
        self.assertTrue(result["covers_explicit"])
        self.assertEqual(result["expected_explicit"], ["at(red, board_l1)"])
        self.assertEqual(result["covered"], ["at(red, board_l1)", "free(A)", "free(B)"])
        self.assertEqual(result["explicit_covered"], ["at(red, board_l1)"])
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["extra"], [])
        self.assertEqual(result["errors"], [])
        self.assertIs(problem.goal, original)

    def test_missing_goal_and_additional_valid_goal_are_reported(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        result = self.compare(problem, ["at(red, stock_r1)"])
        self.assertFalse(result["covers_explicit"])
        self.assertEqual(result["missing"], ["at(red, board_l1)", "free(A)", "free(B)"])
        self.assertEqual(result["explicit_missing"], ["at(red, board_l1)"])
        self.assertEqual(result["extra"], ["at(red, stock_r1)"])
        self.assertEqual(result["errors"], [])

    def test_syntax_unknown_objects_predicates_and_parameter_types_are_errors(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        cases = [("plain text", "格式"), ("at(red, nowhere)", "不存在"),
                 ("teleported(red)", "未知谓词"), ("at(A, board_l1)", "类型"),
                 ("free(A, B)", "参数个数"), ("at(?block, board_l1)", "变量")]
        for goal, fragment in cases:
            with self.subTest(goal=goal):
                result = self.compare(problem, [goal])
                self.assertTrue(any(fragment in error for error in result["errors"]), result)
                self.assertFalse(result["covers_explicit"])

    def test_opposite_task_goal_and_model_internal_contradiction_are_errors(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        result = self.compare(problem, ["not at(red, board_l1)"])
        self.assertTrue(any("相反" in error for error in result["errors"]))
        self.assertEqual(result["extra"], ["not at(red, board_l1)"])
        self.assertEqual(result["goal_invalid"], 0)
        result = self.compare(problem, ["at(red, stock_r1)", "not at(red, stock_r1)"])
        self.assertTrue(any("互相矛盾" in error for error in result["errors"]))

    def test_empty_explicit_targets_still_show_automatic_goals_and_negative_cleanup(self):
        problem = load_problem("tasks/s3a_platform.toml")
        result = self.compare(problem, ["filled(p1)", "not filled(tp1)", "filled(tp2)"])
        self.assertEqual(result["expected_explicit"], [])
        self.assertTrue(result["covers_explicit"])
        self.assertIn("not filled(tp1)", result["expected_task"])
        self.assertTrue(any("filled(tp2)" in error and "相反" in error for error in result["errors"]))
        self.assertEqual(result["extra"], ["filled(tp2)"])
        self.assertAlmostEqual(result["goal_recall"], 2 / 7)

    def test_absent_proposal_has_no_goal_comparison(self):
        from skillnet.decompose.goals import compare_model_goals
        self.assertIsNone(compare_model_goals(load_problem("tasks/s1_nearest_arm.toml"), None))

    def test_static_and_installed_semantic_errors_remain_extra_predictions(self):
        problem = load_problem("tasks/s2_tower.toml")
        result = self.compare(problem, ["installed(b1, t3)", "fits(b1, t3)",
                                        "not fits(b1, t1)"])
        self.assertEqual(result["extra"], ["fits(b1, t3)", "installed(b1, t3)", "not fits(b1, t1)"])
        self.assertEqual(len(result["errors"]), 3)
        self.assertTrue(any("积木与槽位" in message for message in result["errors"]))
        self.assertTrue(any("静态条件" in message for message in result["errors"]))

    def test_home_positions_and_derivations_use_their_actual_argument_types(self):
        problem = load_problem("tasks/s2_tower.toml")
        result = self.compare(problem, ["arm_at(A, home_A)", "slot_ready(t2)",
                                        "has_support(t1)", "arm_at(b1, home_A)"])
        self.assertEqual(result["extra"], ["arm_at(A, home_A)", "arm_at(b1, home_A)",
                                          "has_support(t1)", "slot_ready(t2)"])
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("参数类型", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
