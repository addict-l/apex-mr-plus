"""Compact valid-only reports and proposal provenance in command summaries."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

from skillnet.__main__ import main
from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from skillnet.report import plan_to_dict, render_report


class RevisionReportingTests(unittest.TestCase):
    def test_human_report_groups_valid_actions_and_json_keeps_full_evidence(self):
        problem = load_problem("tasks/s1_reach_handover.toml")
        problem.oracle.update(parse_fact("reach(A, hand_mid)"), Status.UNKNOWN, "修订验收")
        result = run(problem)
        self.assertEqual(result.status, "solved")
        report = render_report(result)
        summary = report.split("相关动作的用途：", 1)[1].split("未验证的能力：", 1)[0]
        self.assertIn("抓取（pick）", summary)
        self.assertIn("放置（place）", summary)
        self.assertIn("能力有效动作", summary)
        self.assertNotIn("place(B, red, board_l1)", summary)
        self.assertNotIn("handover(B, A, red, hand_mid)", summary)
        self.assertNotIn("交接（handover）", summary)
        self.assertNotIn("pick(B, red, stock_r1)", summary)
        items = {item["action"]: item for item in plan_to_dict(result)["relevance"]}
        invalid = items["place(B, red, board_l1)"]
        self.assertEqual(invalid["status"], "INVALID")
        self.assertEqual(invalid["skill"], "place")
        self.assertTrue(any(item["fact"] == "reach(B, board_l1)" and item["status"] == "INVALID"
                            for item in invalid["evidence"]))
        unknown = items["handover(B, A, red, hand_mid)"]
        self.assertEqual(unknown["status"], "UNKNOWN")
        self.assertTrue(unknown["reasons"])
        self.assertEqual(items["pick(B, red, stock_r1)"]["status"], "VALID")

    def test_all_summary_reports_actual_proposal_presence(self):
        without = run(load_problem("tasks/s1_nearest_arm.toml"))
        with_proposal = run(load_problem("tasks/s1_reach_handover.toml"))
        stdout = StringIO()
        stderr = StringIO()
        paths = [Path("tasks/s1_nearest_arm.toml"), Path("tasks/s1_reach_handover.toml")]
        with patch("skillnet.__main__.Path.glob", return_value=iter(paths)), \
                patch("skillnet.__main__._run_task", side_effect=[without, with_proposal]), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["all", "--out", "out/test_v11_reporting"])
        self.assertEqual(code, 0, stderr.getvalue())
        text = stdout.getvalue()
        self.assertIn("是否使用 proposal", text)
        self.assertIn("s1_nearest_arm | solved | 否 |", text)
        self.assertIn("s1_reach_handover | solved | 是 |", text)
