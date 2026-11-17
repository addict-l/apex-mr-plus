"""Experiment records and commands without a live model dependency."""

from contextlib import redirect_stderr, redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from skillnet.__main__ import main
from skillnet.core.problem import Proposal
from skillnet.domains.blocks import load_problem
from skillnet.pipeline import run
from skillnet.report import plan_to_dict, render_report


class FixedModel:
    instances = []

    def __init__(self, *, model, host, mode):
        self.model = model
        self.host = host
        self.mode = mode
        self.calls = []
        self.instances.append(self)

    def decompose(self, problem, feedback=None):
        proposal = Proposal("测试指令", ["at(red, board_c)"],
                            ["pick(A, red, stock_c)", "place(A, red, board_c)"],
                            "模拟 Ollama")
        self.calls.append({"model": self.model, "mode": self.mode,
                           "prompt": "本地模拟提示词", "prompt_length": 7,
                           "raw_output": json.dumps({"goal": proposal.goal, "steps": proposal.steps}),
                           "raw_api_response": "模拟接口响应", "elapsed": 0.25,
                           "json_valid": True, "retry": 0, "error": None})
        return proposal


class ExperimentTests(unittest.TestCase):
    def test_default_tasks_are_the_nine_base_tasks_and_names_accept_paths(self):
        from skillnet.experiment import task_paths

        selected = task_paths()
        self.assertEqual(len(selected), 9)
        self.assertFalse(any(path.stem.endswith("_with_proposal") for path in selected))
        self.assertIn("s2_tower_same_type", {path.stem for path in selected})
        named = task_paths(["s1_nearest_arm"])
        explicit = task_paths(["tasks/s1_nearest_arm.toml"])
        self.assertEqual(named, explicit)
        with self.assertRaisesRegex(ValueError, "任务"):
            task_paths(["does_not_exist"])

    def test_four_settings_preserve_optimum_and_write_all_metrics_and_calls(self):
        from skillnet.experiment import run_experiment

        FixedModel.instances.clear()
        directory = Path("out/test_experiment/three_settings")
        result = run_experiment(tasks=["s1_nearest_arm"], out=directory,
                                model="fake-7b", decomposer_factory=FixedModel)
        self.assertTrue(result.success)
        self.assertEqual([row["setting"] for row in result.rows],
                         ["none", "minimal", "with_capabilities", "nl_only"])
        self.assertEqual([model.mode for model in FixedModel.instances],
                         ["minimal", "with_capabilities", "nl_only"])
        baseline = result.rows[0]["makespan"]
        for row in result.rows:
            self.assertEqual(row["status"], "solved")
            self.assertAlmostEqual(row["makespan"], baseline, places=6)
            self.assertTrue(row["matches_none"])
            for field in ("json_valid", "model_success", "parse_retries", "model_steps", "kept", "added",
                          "deleted", "redundant", "capability_invalid_deletions",
                          "goal_covers_explicit", "goal_missing", "goal_extra", "goal_errors",
                          "completion_goal_reached", "expanded", "search_elapsed",
                          "inference_elapsed"):
                self.assertIn(field, row)
        self.assertEqual(result.rows[0]["model_calls"], 0)
        self.assertIsNone(result.rows[0]["json_valid"])
        self.assertIsNone(result.rows[0]["model_success"])
        for row in result.rows[1:]:
            self.assertTrue(row["model_success"])
            self.assertEqual(row["model_steps"], 2)
            self.assertEqual(row["kept"], 2)
            self.assertEqual(row["parse_retries"], 0)
            self.assertEqual(row["inference_elapsed"], 0.25)
        with (directory / "experiment.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 4)
        self.assertIn("四种设置", (directory / "summary.md").read_text(encoding="utf-8"))
        calls = directory / "minimal/s1_nearest_arm/calls"
        self.assertEqual((calls / "001_prompt.txt").read_text(encoding="utf-8"), "本地模拟提示词")
        self.assertTrue((calls / "001_response.txt").is_file())
        self.assertTrue((calls / "001_record.json").is_file())
        self.assertTrue((directory / "none/s1_nearest_arm/plan.json").is_file())

    def test_none_setting_ignores_a_task_file_proposal(self):
        from skillnet.experiment import run_experiment

        problem = load_problem("tasks/s1_nearest_arm.toml")
        problem.proposal = Proposal("错误输入", [], ["unknown(A)"], "任务文件")
        with patch("skillnet.experiment.load_problem", return_value=problem):
            result = run_experiment(tasks=["s1_nearest_arm"],
                                    out="out/test_experiment/ignore_proposal",
                                    decomposer_factory=FixedModel)
        self.assertEqual(result.rows[0]["model_calls"], 0)
        self.assertEqual(result.rows[0]["model_steps"], 0)
        self.assertEqual(result.rows[0]["status"], "solved")

    def test_capability_rejection_and_repair_are_counted_in_the_csv(self):
        from skillnet.experiment import run_experiment

        class WrongReachModel(FixedModel):
            def decompose(self, problem, feedback=None):
                proposal = super().decompose(problem, feedback)
                proposal.goal = ["at(red, board_l1)"]
                proposal.steps = ["pick(B, red, stock_r1)", "place(B, red, board_l1)"]
                self.calls[-1]["raw_output"] = json.dumps({"goal": proposal.goal, "steps": proposal.steps})
                return proposal

        result = run_experiment(tasks=["s1_reach_handover"],
                                out="out/test_experiment/capability_rejection",
                                decomposer_factory=WrongReachModel)
        self.assertTrue(result.success)
        for row in result.rows[1:]:
            self.assertEqual(row["model_steps"], 2)
            self.assertEqual(row["kept"], 1)
            self.assertEqual(row["deleted"], 1)
            self.assertEqual(row["added"], 2)
            self.assertEqual(row["capability_invalid_deletions"], 1)
            self.assertTrue(row["completion_goal_reached"])

    def test_model_failure_keeps_retry_evidence_and_reports_direct_search_fallback(self):
        from skillnet.experiment import run_experiment

        class BrokenModel(FixedModel):
            def decompose(self, problem, feedback=None):
                self.calls.extend({"model": self.model, "prompt": "反馈后的提示词",
                                   "raw_output": "不是 JSON", "elapsed": 0.1,
                                   "json_valid": False, "retry": retry,
                                   "error": "模型输出解析失败"} for retry in (0, 1, 2))
                raise ValueError("重试两次后仍无法解析")

        result = run_experiment(tasks=["s1_nearest_arm"],
                                out="out/test_experiment/fallback",
                                decomposer_factory=BrokenModel)
        self.assertFalse(result.success)
        for row in result.rows[1:]:
            self.assertEqual(row["status"], "solved")
            self.assertTrue(row["matches_none"])
            self.assertFalse(row["model_success"])
            self.assertFalse(row["json_valid"])
            self.assertEqual(row["parse_retries"], 2)
            self.assertEqual(row["model_calls"], 3)
            self.assertEqual(row["model_steps"], 0)
            self.assertIsNone(row["completion_goal_reached"])
            self.assertIn("无法解析", row["model_error"])
        summary = (result.directory / "summary.md").read_text(encoding="utf-8")
        self.assertIn("不能把回退后的 solved 当作模型成功生成方案", summary)
        self.assertIn("验收：未通过", summary)
        self.assertIn("规划一致性：通过", summary)
        self.assertIn("有效模型方案 0/3", summary)

    def test_an_empty_but_valid_model_proposal_counts_as_a_model_result(self):
        from skillnet.experiment import run_experiment

        class EmptyModel(FixedModel):
            def decompose(self, problem, feedback=None):
                proposal = super().decompose(problem, feedback)
                proposal.steps = []
                self.calls[-1]["raw_output"] = json.dumps({"goal": proposal.goal, "steps": []})
                return proposal

        result = run_experiment(tasks=["s1_nearest_arm"],
                                out="out/test_experiment/empty_proposal",
                                decomposer_factory=EmptyModel)
        self.assertTrue(result.success)
        for row in result.rows[1:]:
            self.assertTrue(row["model_success"])
            self.assertEqual(row["model_steps"], 0)
            self.assertEqual(row["added"], 2)
            self.assertTrue(row["matches_none"])

    def test_summary_marks_a_changed_cost_as_failure(self):
        from skillnet.experiment import run_experiment

        problem = load_problem("tasks/s1_nearest_arm.toml")
        baseline = run(problem)
        changed = run(problem)
        changed.schedule.makespan += 1
        with patch("skillnet.experiment.run", side_effect=[baseline, changed, baseline, baseline]):
            result = run_experiment(tasks=["s1_nearest_arm"],
                                    out="out/test_experiment/changed_cost",
                                    decomposer_factory=FixedModel)
        self.assertFalse(result.success)
        self.assertFalse(result.rows[1]["matches_none"])
        self.assertIn("未通过", (result.directory / "summary.md").read_text(encoding="utf-8"))

    def test_summary_distinguishes_parse_success_from_unfinished_repairs(self):
        from skillnet.experiment import run_experiment
        from skillnet.planning.completion import CompletionReport

        unfinished = CompletionReport(goal_reached=False, unresolved=["at(red, board_c)"])
        with patch("skillnet.pipeline.complete", return_value=unfinished):
            result = run_experiment(tasks=["s1_nearest_arm"],
                                    out="out/test_experiment/unfinished_repairs",
                                    decomposer_factory=FixedModel)
        self.assertTrue(result.success)
        self.assertTrue(all(row["model_success"] for row in result.rows[1:]))
        self.assertTrue(all(row["completion_goal_reached"] is False for row in result.rows[1:]))
        summary = (result.directory / "summary.md").read_text(encoding="utf-8")
        self.assertIn("补全达成 0/1", summary)
        self.assertIn("完整链路合计", summary)
        self.assertIn("独立搜索", summary)

    def test_reports_keep_call_records_and_goal_comparison(self):
        result = run(load_problem("tasks/s1_nearest_arm.toml"))
        result.model_calls = [{"model": "fake-7b", "prompt": "提示词", "raw_output": "原始输出",
                               "elapsed": 0.25, "json_valid": True}]
        result.model_goal_comparison = {"covers_explicit": False, "covered": [],
                                        "missing": ["at(red, board_c)"], "extra": ["free(A)"],
                                        "errors": ["未知对象 ghost"]}
        result.model_error = "模型连接失败，已退回不使用模型"
        data = plan_to_dict(result)
        self.assertEqual(data["model_calls"], result.model_calls)
        self.assertEqual(data["model_goal_comparison"], result.model_goal_comparison)
        self.assertEqual(data["model_error"], result.model_error)
        report = render_report(result)
        for text in ("模型目标对比", "at(red, board_c)", "未知对象 ghost", "模型连接失败"):
            self.assertIn(text, report)

    def test_run_model_cli_replaces_fixed_proposal_and_accepts_prompt_mode(self):
        stdout, stderr = StringIO(), StringIO()
        with patch("skillnet.decompose.ollama.OllamaDecomposer", FixedModel), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["run", "tasks/s1_nearest_arm.toml", "--model", "fake-7b",
                         "--mode", "minimal", "--host", "http://127.0.0.1:11434",
                         "--out", "out/test_experiment/cli_run"])
        self.assertEqual(code, 0, stderr.getvalue())
        data = json.loads(Path("out/test_experiment/cli_run/s1_nearest_arm/plan.json").read_text(encoding="utf-8"))
        self.assertEqual(data["proposal"]["source"], "模拟 Ollama")
        self.assertTrue(data["model_calls"])

    def test_model_and_no_proposal_are_rejected_together(self):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["run", "tasks/s1_nearest_arm.toml", "--model", "fake-7b", "--no-proposal"])
        self.assertEqual(code, 2)
        self.assertIn("不能同时", stderr.getvalue())

    def test_experiment_cli_prints_result_and_exits_nonzero_for_mismatch(self):
        from skillnet.experiment import ExperimentResult

        stdout, stderr = StringIO(), StringIO()
        expected = ExperimentResult([], False, Path("out/test_experiment/cli_experiment"))
        with patch("skillnet.experiment.run_experiment", return_value=expected) as execute, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["experiment", "--model", "fake-7b", "--tasks", "s1_nearest_arm",
                         "--out", str(expected.directory)])
        self.assertEqual(code, 1)
        self.assertEqual(execute.call_args.kwargs["tasks"], ["s1_nearest_arm"])
        self.assertIn("实验", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
