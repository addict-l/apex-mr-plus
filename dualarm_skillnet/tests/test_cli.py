"""Normal command-line entry points and generated report artifacts."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from skillnet.__main__ import main
from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from skillnet.report import plan_to_dict, render_report


class CliTests(unittest.TestCase):
    output = Path("out/test_cli")

    def invoke(self, arguments):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_network_writes_automatic_derived_edges(self):
        code, text, error = self.invoke(["network", "--out", str(self.output)])
        self.assertEqual(code, 0, error)
        self.assertIn("抓取", text)
        self.assertIn("pick → place", text)
        self.assertIn("slot_ready（由 filled 推出）", text)
        graph = (self.output / "skill_network.mmd").read_text(encoding="utf-8")
        self.assertTrue(graph.startswith("flowchart TD"))
        self.assertIn("support_on -->", graph)

    def test_run_writes_all_four_reports(self):
        code, text, error = self.invoke(["run", "tasks/s1_reach_handover.toml", "--out", str(self.output)])
        self.assertEqual(code, 0, error)
        directory = self.output / "s1_reach_handover"
        for name in ("report.md", "plan.json", "taskgraph.mmd", "taskgraph.json"):
            self.assertTrue((directory / name).is_file())
        report = (directory / "report.md").read_text(encoding="utf-8")
        for marker in ("① 初始状态", "② 小模型方案", "③ 校验与补全", "④ 搜索结果", "⑤ 运动验证与反馈", "⑥ 任务顺序图"):
            self.assertIn(marker, report)
        self.assertIn("删除 place(B, red, board_l1)", report)
        self.assertIn("reach(B, board_l1) = INVALID", report)
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["status"], "solved")
        self.assertTrue(plan["schedule"])
        self.assertTrue(plan["completion"]["entries"])
        self.assertIn("state_before", plan["schedule"][0])
        self.assertTrue(json.loads((directory / "taskgraph.json").read_text(encoding="utf-8"))["nodes"])
        self.assertIn("占位后端", text)

    def test_run_flags_skip_proposal_and_pruning(self):
        code, _, error = self.invoke(["run", "tasks/s1_reach_handover.toml", "--out", str(self.output / "flags"), "--no-proposal", "--no-prune"])
        self.assertEqual(code, 0, error)
        plan = json.loads((self.output / "flags/s1_reach_handover/plan.json").read_text(encoding="utf-8"))
        self.assertIsNone(plan["proposal"])
        self.assertIsNone(plan["completion"])
        self.assertEqual(plan["status"], "solved")

    def test_unknown_capability_is_present_in_report_and_json(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        fact = parse_fact("reach(A, hand_mid)")
        problem.oracle.update(fact, Status.UNKNOWN, "测试")
        result = run(problem)
        self.assertEqual(result.status, "solved")
        self.assertIn("reach(A, hand_mid) = UNKNOWN", render_report(result))
        self.assertIn(str(fact), plan_to_dict(result)["unknown"])

    def test_all_prints_summary(self):
        result = run(load_problem("tasks/s1_nearest_arm.toml"))
        with patch("skillnet.__main__._run_task", return_value=result) as execute:
            code, text, error = self.invoke(["all", "--out", str(self.output)])
        self.assertEqual(code, 0, error)
        self.assertEqual(execute.call_count, len(list(Path("tasks").glob("*.toml"))))
        self.assertIn("任务汇总", text)
        self.assertIn("扩展状态数", text)

    def test_missing_task_has_chinese_error_and_nonzero_code(self):
        code, _, error = self.invoke(["run", "tasks/does_not_exist.toml"])
        self.assertEqual(code, 2)
        self.assertIn("错误：", error)

    def test_python_module_entry_point_and_chinese_help(self):
        process = subprocess.run([sys.executable, "-X", "utf8", "-m", "skillnet", "--help"],
                                 cwd=Path.cwd(), capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("双臂积木技能网络规划器", process.stdout)
        self.assertIn("用法", process.stdout)
        self.assertIn("prompt", process.stdout)


if __name__ == "__main__":
    unittest.main()
