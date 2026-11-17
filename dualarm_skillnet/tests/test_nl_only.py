"""Goal-blind natural-language prompts and full-goal understanding metrics."""

from pathlib import Path
import unittest

from skillnet.core.logic import parse_condition
from skillnet.core.problem import Proposal
from skillnet.decompose.goals import compare_model_goals
from skillnet.decompose.prompt import build_prompt
from skillnet.domains.blocks import load_problem


class NaturalLanguageOnlyPromptTests(unittest.TestCase):
    def test_nl_only_keeps_scene_skills_and_capabilities_without_concrete_goals(self):
        for path in sorted(Path("tasks").glob("*.toml")):
            with self.subTest(task=path.stem):
                problem = load_problem(path)
                prompt = build_prompt(problem, "nl_only")
                self.assertIn(problem.instruction, prompt)
                self.assertIn("场景信息", prompt)
                self.assertIn("能力摘要", prompt)
                self.assertIn("技能清单", prompt)
                self.assertIn("goal", prompt)
                self.assertIn("自行", prompt)
                for skill in problem.library.skills:
                    self.assertIn(skill.signature, prompt)
                for condition in problem.goal:
                    self.assertNotIn(str(condition), prompt)
                for label in ("任务目标：", "默认目标", "fill_all_slots"):
                    self.assertNotIn(label, prompt)
                self.assertNotIn('"at(red, board_l1)"', prompt)

    def test_changing_formal_goals_does_not_change_nl_only_prompt(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        before = build_prompt(problem, "nl_only")
        problem.goal = (parse_condition("at(red, buffer_mid)"),)
        problem.explicit_goal = problem.goal
        self.assertEqual(build_prompt(problem, "nl_only"), before)

    def test_control_prompts_still_supply_formal_goals(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        for mode in ("minimal", "with_capabilities"):
            with self.subTest(mode=mode):
                prompt = build_prompt(problem, mode)
                self.assertIn("任务目标：", prompt)
                for condition in problem.goal:
                    self.assertIn(str(condition), prompt)
                self.assertEqual("能力摘要" in prompt, mode == "with_capabilities")


class FullGoalMetricTests(unittest.TestCase):
    def compare(self, name, goals):
        problem = load_problem(f"tasks/{name}.toml")
        original = problem.goal
        comparison = compare_model_goals(problem, Proposal(goal=goals))
        self.assertIs(problem.goal, original)
        return comparison

    def test_missing_default_free_goals_reduces_recall(self):
        result = self.compare("s1_reach_handover", ["at(red, board_l1)"])
        self.assertAlmostEqual(result["goal_recall"], 1 / 3)
        self.assertEqual(result["goal_precision"], 1.0)
        self.assertEqual(result["goal_invalid"], 0)
        self.assertEqual(result["covered"], ["at(red, board_l1)"])
        self.assertEqual(result["missing"], ["free(A)", "free(B)"])
        self.assertTrue(result["covers_explicit"])

    def test_automatic_goals_are_measured_even_without_explicit_goal(self):
        result = self.compare("s3a_platform", ["filled(p1)"])
        self.assertEqual(result["expected_explicit"], [])
        self.assertAlmostEqual(result["goal_recall"], 1 / 7)
        self.assertEqual(result["goal_precision"], 1.0)
        self.assertIn("not filled(tp1)", result["missing"])
        self.assertIn("not filled(tp2)", result["missing"])

    def test_extra_and_invalid_predictions_reduce_precision(self):
        result = self.compare("s1_reach_handover", [
            "at(red, board_l1)", "free(A)", "at(red, stock_r1)",
            "not a fact", "imaginary(red)", "at(red, missing_place)",
        ])
        self.assertAlmostEqual(result["goal_recall"], 2 / 3)
        self.assertAlmostEqual(result["goal_precision"], 2 / 6)
        self.assertEqual(result["goal_invalid"], 3)
        self.assertEqual(result["extra"], ["at(red, missing_place)", "at(red, stock_r1)",
                                          "imaginary(red)", "not a fact"])
        self.assertEqual(result["missing"], ["free(B)"])

    def test_duplicates_are_set_predictions_and_invalid_entries_still_count(self):
        result = self.compare("s1_reach_handover", [
            "at(red, board_l1)", " at( red, board_l1 ) ",
            "at(red, stock_r1)", "broken", "broken",
        ])
        # Canonical valid facts are deduplicated; malformed entries count each.
        self.assertAlmostEqual(result["goal_recall"], 1 / 3)
        self.assertAlmostEqual(result["goal_precision"], 1 / 4)
        self.assertEqual(result["goal_invalid"], 2)

    def test_type_and_contradiction_errors_remain_visible_without_false_coverage(self):
        result = self.compare("s1_reach_handover", [
            "not at(red, board_l1)", "free(red)",
        ])
        self.assertEqual(result["goal_recall"], 0)
        self.assertEqual(result["goal_precision"], 0)
        # The requested invalid counter is syntax/predicate/object existence.
        self.assertEqual(result["goal_invalid"], 0)
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(result["extra"], ["free(red)", "not at(red, board_l1)"])

    def test_empty_prediction_scores_zero_for_nonempty_task(self):
        result = self.compare("s1_reach_handover", [])
        self.assertEqual(result["goal_recall"], 0)
        self.assertEqual(result["goal_precision"], 0)
        self.assertEqual(result["goal_invalid"], 0)
        self.assertEqual(set(result["missing"]), {"free(A)", "free(B)", "at(red, board_l1)"})

    def test_empty_task_and_prediction_are_vacuously_exact(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        problem.goal = ()
        problem.explicit_goal = ()
        result = compare_model_goals(problem, Proposal(goal=[]))
        self.assertEqual(result["goal_recall"], 1)
        self.assertEqual(result["goal_precision"], 1)
