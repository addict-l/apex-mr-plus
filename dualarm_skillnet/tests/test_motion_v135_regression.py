"""Regression orchestration contracts; never run the default suite or search."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools import motion_v135_regression as regression


class RegressionDispatchTests(unittest.TestCase):
    def test_default_runs_once_and_seed_zero_is_explicit(self):
        for seed_zero, expected in ((False, [None]), (True, [None, "0"])):
            with self.subTest(seed_zero=seed_zero), tempfile.TemporaryDirectory(dir=regression.ROOT/"out") as folder:
                calls = []

                def run_job(name, command, seconds, *, live):
                    calls.append((name, command, seconds, live, os.environ.get("PYTHONHASHSEED")))
                    output = Path(command[command.index("--json")+1])
                    output.write_text(json.dumps({"status": "passed", "tests_run": 2,
                        "tests_discovered": 2, "failure_count": 0, "error_count": 0}), encoding="utf-8")
                    return {"status": "passed", "code_unchanged_during_job": True}

                with patch.object(regression.control, "run_job", side_effect=run_job), \
                        patch.object(regression.control, "record"):
                    answer = regression.dispatch("mock_default", "default", hashseed0=seed_zero, base=Path(folder))
                self.assertTrue(answer["passed"])
                self.assertEqual([row[4] for row in calls], expected)
                self.assertTrue(all(row[2] == 1200. and row[3] is False for row in calls))
                self.assertTrue(all("--worker-kind" in row[1] and "default" in row[1] for row in calls))
                self.assertEqual(len({row[0] for row in calls}), len(calls))

    def test_successful_exit_without_worker_evidence_cannot_pass(self):
        with tempfile.TemporaryDirectory(dir=regression.ROOT/"out") as folder, \
                patch.object(regression.control, "run_job", return_value={
                    "status": "passed", "code_unchanged_during_job": True}):
            result = regression.dispatch("mock_missing", "default", base=Path(folder))
        self.assertFalse(result["passed"])
        self.assertEqual(result["runs"][0]["worker_evidence_reason"], "missing_worker_json")

    def test_hashseed_environment_is_restored_after_failed_control(self):
        with tempfile.TemporaryDirectory(dir=regression.ROOT/"out") as folder, \
                patch.dict(os.environ, {"PYTHONHASHSEED": "73"}), \
                patch.object(regression.control, "run_job", side_effect=RuntimeError("machine occupied")):
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                regression.dispatch("mock_guard", "default", base=Path(folder))
            self.assertEqual(os.environ["PYTHONHASHSEED"], "73")

    def test_default_worker_delegates_to_existing_timed_worker_with_new_json(self):
        with tempfile.TemporaryDirectory(dir=regression.ROOT/"out") as folder:
            output = Path(folder)/"worker.json"
            with patch.object(regression.timed, "worker", return_value=0) as worker:
                self.assertEqual(regression.default_worker("mock_default", output), 0)
            args = worker.call_args.args[0]
            self.assertEqual(args.label, "mock_default")
            self.assertEqual(Path(args.json), output)
            self.assertIsNone(args.module)

    def test_existing_different_request_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=regression.ROOT/"out") as folder:
            output = Path(folder)/"regressions/mock_same/default"
            output.mkdir(parents=True)
            (output/"request.json").write_text(json.dumps({"name": "other"}), encoding="utf-8")
            with patch.object(regression.control, "run_job") as run_job:
                with self.assertRaisesRegex(ValueError, "request|请求|已有"):
                    regression.dispatch("mock_same", "default", base=Path(folder))
            run_job.assert_not_called()


class OriginalElevenTests(unittest.TestCase):
    def test_manifest_is_exact_original_eleven_with_matching_snapshot_sha(self):
        manifest = regression.original_manifest()
        self.assertEqual(len(manifest["tasks"]), 11)
        self.assertEqual({row["task"] for row in manifest["tasks"]}, set(regression.ORIGINAL11))
        self.assertTrue(all(row["task_sha256"] == row["snapshot_sha256"] == row["formal_task_sha256"]
                            for row in manifest["tasks"]))
        self.assertTrue(all(row["file"].startswith("tasks/") and row["file"].count("/") == 1
                            for row in manifest["tasks"]))

    def test_changed_original_sha_is_rejected_before_any_solver(self):
        original = regression.old.sha256
        changed = regression.ROOT/"tasks"/(regression.ORIGINAL11[0]+".toml")
        with patch.object(regression.old, "sha256", side_effect=lambda path:
                          "changed" if Path(path) == changed else original(path)), \
                patch.object(regression, "run_problem") as run:
            with self.assertRaisesRegex(ValueError, "SHA"):
                regression.original_manifest()
        run.assert_not_called()

    def test_original_worker_uses_pipeline_no_backend_no_proposal_and_rule_replay(self):
        manifest = regression.original_manifest()
        problems = []

        def load(path, library):
            problem = SimpleNamespace(name=path.stem, proposal="must be cleared")
            problems.append(problem)
            return problem

        references = {row["task"]: row["formal_reference"] for row in manifest["tasks"]}

        def solve(problem, **kwargs):
            self.assertIsNone(problem.proposal)
            self.assertEqual(set(kwargs), {"deadline"})
            ref = references[problem.name]
            schedule = SimpleNamespace(status=ref["status"], makespan=ref["makespan"],
                suboptimality_bound=ref["bound"], expanded=3, generated=7, elapsed=.01,
                lower_bound=ref["makespan"]/ref["bound"], weight=1., stages=(),
                first_solution_seconds=.005, first_solution_expanded=2, steps=[])
            return SimpleNamespace(status=schedule.status, schedule=schedule, feedback=[])

        with tempfile.TemporaryDirectory(dir=regression.ROOT/"out") as folder, \
                patch.object(regression, "load_problem", side_effect=load), \
                patch.object(regression, "Engine", side_effect=lambda problem: problem), \
                patch.object(regression, "run_problem", side_effect=solve) as run, \
                patch.object(regression, "rule_replay", return_value=[]) as replay, \
                patch.object(regression, "plan_to_dict", return_value={"status": "solved"}):
            output = Path(folder)/"originals.json"
            self.assertEqual(regression.original_worker("mock_originals", output, manifest=manifest), 0)
            proof = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(proof["passed"])
        self.assertEqual(len(proof["rows"]), 11)
        self.assertEqual(run.call_count, 11)
        self.assertEqual(replay.call_count, 11)
        self.assertTrue(all(row["rule_replay"]["passed"] and row["expanded"] == 3
                            and row["budget_seconds"] == 60. for row in proof["rows"]))

    def test_budget_row_is_preserved_and_does_not_abort_independent_remaining_tasks(self):
        manifest = regression.original_manifest()
        refs = {row["task"]: row["formal_reference"] for row in manifest["tasks"]}

        def solve(problem, **kwargs):
            ref = refs[problem.name]
            budget = problem.name == regression.ORIGINAL11[0]
            schedule = SimpleNamespace(status="budget" if budget else "solved",
                makespan=float("inf") if budget else ref["makespan"],
                suboptimality_bound=None if budget else ref["bound"], expanded=4, generated=9,
                elapsed=.01, lower_bound=None if budget else ref["makespan"]/ref["bound"], weight=1., stages=(), steps=[],
                first_solution_seconds=None, first_solution_expanded=None)
            return SimpleNamespace(status=schedule.status, schedule=schedule, feedback=[])

        with tempfile.TemporaryDirectory(dir=regression.ROOT/"out") as folder, \
                patch.object(regression, "load_problem", side_effect=lambda path, library: SimpleNamespace(name=path.stem)), \
                patch.object(regression, "Engine", side_effect=lambda problem: problem), \
                patch.object(regression, "run_problem", side_effect=solve), \
                patch.object(regression, "rule_replay", return_value=[]) as replay, \
                patch.object(regression, "plan_to_dict", return_value={}):
            output = Path(folder)/"originals.json"
            self.assertEqual(regression.original_worker("mock_budget", output, manifest=deepcopy(manifest)), 1)
            proof = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(proof["passed"])
        self.assertEqual(len(proof["rows"]), 11)
        self.assertEqual(proof["rows"][0]["status"], "budget")
        self.assertIsNone(proof["rows"][0]["makespan"])
        self.assertEqual(replay.call_count, 10)


if __name__ == "__main__":
    unittest.main()
