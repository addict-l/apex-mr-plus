"""The revised experiment exposes goal understanding and bounded repairs."""
from contextlib import redirect_stderr, redirect_stdout
from inspect import signature
from io import StringIO
import csv
from pathlib import Path
import unittest
from unittest.mock import patch

from skillnet.__main__ import main
from skillnet.domains.blocks import load_problem
from skillnet.experiment import run_experiment
from skillnet.pipeline import run
from skillnet.planning.completion import CompletionReport
from skillnet.report import plan_to_dict, render_report
from tests.test_experiment import FixedModel


class RevisedExperimentTests(unittest.TestCase):
    def test_four_settings_record_complete_goal_and_repair_metrics(self):
        FixedModel.instances.clear()
        result = run_experiment(tasks=["s1_nearest_arm"],
                                out="out/test_v121_experiment/four_settings",
                                decomposer_factory=FixedModel)
        self.assertTrue(result.success)
        self.assertEqual([row["setting"] for row in result.rows],
                         ["none", "minimal", "with_capabilities", "nl_only"])
        self.assertEqual([model.mode for model in FixedModel.instances],
                         ["minimal", "with_capabilities", "nl_only"])
        for row in result.rows:
            for key in ("repair_abandoned", "repair_abandon_reason", "goal_recall",
                        "goal_precision", "goal_invalid", "completion_elapsed",
                        "timing_limit", "timing_within_budget"):
                self.assertIn(key, row)
            self.assertTrue(row["timing_within_budget"])
        for row in result.rows[1:]:
            self.assertAlmostEqual(row["goal_recall"], 1 / 3)
            self.assertEqual(row["goal_precision"], 1)
            self.assertEqual(row["goal_invalid"], 0)
        summary = (result.directory / "summary.md").read_text(encoding="utf-8")
        for label in ("nl_only", "召回率", "精确率", "无效条数", "放弃修补", "v1.2 耗时对比"):
            self.assertIn(label, summary)

    def test_reports_keep_abandonment_reason_and_elapsed_time(self):
        result = run(load_problem("tasks/s1_nearest_arm.toml"))
        report = CompletionReport(goal_reached=False, unresolved=["at(red, board_c)"])
        report.abandoned = True
        report.abandon_reason = "已处理四步，删除比例超过一半"
        report.elapsed = 4.25
        report.processed_steps = 4
        report.deleted_steps = 3
        result.completion = report
        result.completion_elapsed = 4.25
        result.model_goal_comparison = {"covers_explicit": False, "covered": [],
                                       "missing": ["free(A)", "free(B)"], "extra": [],
                                       "errors": [], "goal_recall": 1 / 3,
                                       "goal_precision": 1.0, "goal_invalid": 0}
        value = plan_to_dict(result)
        self.assertTrue(value["completion"]["abandoned"])
        self.assertEqual(value["completion"]["abandon_reason"], report.abandon_reason)
        self.assertEqual(value["completion_elapsed"], 4.25)
        text = render_report(result)
        for fragment in ("放弃修补", report.abandon_reason, "4.250", "召回率", "精确率"):
            self.assertIn(fragment, text)

    def test_time_limit_failure_is_not_hidden_by_solved_and_matching_cost(self):
        values = [0, 1, 1, 34, 34, 35, 35, 36]
        with patch("skillnet.experiment.monotonic", side_effect=values):
            result = run_experiment(tasks=["s1_nearest_arm"],
                                    out="out/test_v121_experiment/time_limit",
                                    decomposer_factory=FixedModel)
        self.assertFalse(result.success)
        self.assertTrue(all(row["matches_none"] for row in result.rows))
        self.assertFalse(result.rows[1]["timing_within_budget"])
        self.assertAlmostEqual(result.rows[1]["timing_limit"], 31.25)
        self.assertIn("耗时", (result.directory / "summary.md").read_text(encoding="utf-8"))

    def test_default_destination_is_new_and_old_results_are_protected(self):
        self.assertEqual(signature(run_experiment).parameters["out"].default,
                         "results/llm_v1_2_1")
        with patch("skillnet.experiment._checkpoint", side_effect=AssertionError("不得写旧结果")):
            with self.assertRaisesRegex(ValueError, "旧|v1.2|覆盖"):
                run_experiment(tasks=["s1_nearest_arm"], out="results/llm_v1_2",
                               decomposer_factory=FixedModel)

    def test_summary_compares_previous_measurements_without_modifying_them(self):
        fixture = Path("out/test_v121_experiment/previous.csv")
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text("task,setting,total_elapsed\ns1_nearest_arm,minimal,100.0\n",
                           encoding="utf-8")
        before = fixture.read_bytes()
        with patch("skillnet.experiment.PREVIOUS_CSV", fixture, create=True):
            result = run_experiment(tasks=["s1_nearest_arm"],
                                    out="out/test_v121_experiment/previous_comparison",
                                    decomposer_factory=FixedModel)
        summary = (result.directory / "summary.md").read_text(encoding="utf-8")
        self.assertIn("v1.2 耗时对比", summary)
        self.assertIn("100.000", summary)
        self.assertEqual(before, fixture.read_bytes())

    def test_natural_language_mode_is_available_in_prompt_and_run_commands(self):
        out, error = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(error):
            code = main(["prompt", "tasks/s1_nearest_arm.toml", "--mode", "nl_only"])
        self.assertEqual(code, 0, error.getvalue())
        self.assertNotIn("任务目标：", out.getvalue())
        out, error = StringIO(), StringIO()
        with patch("skillnet.decompose.ollama.OllamaDecomposer", FixedModel), \
                redirect_stdout(out), redirect_stderr(error):
            code = main(["run", "tasks/s1_nearest_arm.toml", "--model", "fake-7b",
                         "--mode", "nl_only", "--host", "http://127.0.0.1:12345",
                         "--out", "out/test_v121_experiment/cli_nl_only"])
        self.assertEqual(code, 0, error.getvalue())


if __name__ == "__main__":
    unittest.main()
