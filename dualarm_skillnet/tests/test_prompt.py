import subprocess
import sys
import unittest

from skillnet.core.logic import parse_fact
from skillnet.decompose.prompt import build_feedback, build_prompt, parse_model_output
from skillnet.domains.blocks import load_problem
from skillnet.planning.completion import CompletionEntry, CompletionReport


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.problem = load_problem("tasks/s1_reach_handover.toml")

    def test_minimal_contains_scene_and_complete_skill_library(self):
        prompt = build_prompt(self.problem, "minimal")
        self.assertNotIn("能力摘要", prompt)
        self.assertNotIn("A 能到达", prompt)
        self.assertIn("积木 red", prompt)
        self.assertIn("位置 stock_r1", prompt)
        self.assertIn("前置条件", prompt)
        self.assertIn("开始时删除", prompt)
        self.assertIn("结束时添加", prompt)
        for skill in self.problem.library.skills:
            self.assertIn(skill.signature, prompt)
        tower = build_prompt(load_problem("tasks/s2_tower.toml"), "minimal")
        self.assertIn("槽位 t2", tower)
        self.assertIn("下接 t1", tower)

    def test_default_prompt_includes_live_capability_evidence(self):
        self.problem.oracle.update(parse_fact("reach(A, hand_mid)"), "UNKNOWN", "测试")
        prompt = build_prompt(self.problem)
        self.assertIn("能力摘要", prompt)
        self.assertIn("A 可达性未验证：hand_mid", prompt)
        self.assertIn("A 不能到达：stock_r1", prompt)
        self.assertIn("B 能到达：stock_r1", prompt)
        self.assertIn("需要两只臂拿的积木：无", prompt)
        tower = build_prompt(load_problem("tasks/s2_tower.toml"))
        self.assertTrue(any("能到达" in line and "t3" in line for line in tower.splitlines()))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "提示词模式"):
            build_prompt(self.problem, "other")

    def test_parse_first_json_object_inside_surrounding_text(self):
        proposal = parse_model_output('下面是方案\n```json\n{"goal":["not filled(t1)"],'
                                      '"steps":["detach(A, red, t1)"], "note":"brace } inside string"}'
                                      '\n```\n结束 {"goal":[],"steps":[]}')
        self.assertEqual(proposal.goal, ["not filled(t1)"])
        self.assertEqual(proposal.steps, ["detach(A, red, t1)"])
        self.assertEqual(proposal.source, "模型输出")

    def test_bad_first_object_is_not_skipped(self):
        with self.assertRaisesRegex(ValueError, "第一个 JSON 对象格式错误"):
            parse_model_output('前文 {bad} 后文 {"goal": [], "steps": []}')

    def test_schema_errors_have_chinese_messages(self):
        invalid = ["纯文本", '{"goal":[]}', '{"goal":[],"steps":"pick(A)"}',
                   '{"goal":[3],"steps":[]}', '{"goal":[],"steps":[null]}',
                   '{"goal":["not a condition"],"steps":[]}']
        for text in invalid:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "模型输出"):
                parse_model_output(text)

    def test_feedback_only_when_goals_remain_unresolved(self):
        entry = CompletionEntry("删除", "place(B, red, board_l1)", "reach(B, board_l1) = INVALID", 2)
        report = CompletionReport(entries=[entry], unresolved=["at(red, board_l1)"])
        feedback = build_feedback(report)
        self.assertTrue(any("第 2 步" in message and "INVALID" in message for message in feedback))
        self.assertTrue(any("at(red, board_l1)" in message for message in feedback))
        report.unresolved.clear()
        self.assertEqual(build_feedback(report), [])

    def test_prompt_cli_uses_the_requested_mode(self):
        for mode in ("minimal", "with_capabilities"):
            with self.subTest(mode=mode):
                result = subprocess.run([sys.executable, "-X", "utf8", "-m", "skillnet", "prompt",
                                         "tasks/s1_reach_handover.toml", "--mode", mode],
                                        capture_output=True, text=True, encoding="utf-8", check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("任务目标", result.stdout)
                self.assertEqual("能力摘要" in result.stdout, mode == "with_capabilities")


if __name__ == "__main__":
    unittest.main()
