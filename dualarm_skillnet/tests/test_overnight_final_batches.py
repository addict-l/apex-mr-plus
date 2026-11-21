"""Post-M4 workflow orchestration with journal doubles, never real jobs."""
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import overnight_batch as batch
from tools import overnight_final_batches as final


class FinalBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="final_batch_test_", dir=batch.ROOT / "out")
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.paths = final.Locations(self.directory / "results")
        self.task = self.directory / "task.toml"
        self.task.write_text('name="mock"\n', encoding="utf-8")
        self.raw = {"schema_version": 1, "session_id": "overnight_20261115", "port": 23210,
            "state_dir": str(self.paths.overnight / "batch_state"), "report_dir": str(self.paths.base),
            "phase_budgets_seconds": {"plan": None, "sim": 18000, "gates": None},
            "jobs": [{"id": "p", "phase": "plan", "family": "fixture", "cohort": "precheck", "task": str(self.task),
                      "output": str(self.paths.overnight / "p")}], "preparation": {}}
        batch.atomic_json(self.paths.manifest, self.raw)
        batch.atomic_json(self.paths.m4, {"finished": True, "M4_passed": False})
        self.events = []

    def select(self, selection, seed, path):
        self.events.append("select" + str(selection))
        raw = batch.read_json(path)
        if not any(job.get("selection") == selection and job["phase"] == "sim" for job in raw["jobs"]):
            for index in range(20):
                job = {"id": f"s{selection}-{index}", "phase": "sim", "selection": selection,
                    "family": f"F{index//4+1}", "output": str(self.paths.overnight / f"s{selection}-{index}")}
                if index % 4 == 3:
                    job["skip_reason"] = "真实独立solved样本不足"
                else:
                    job["task"] = str(self.task)
                raw["jobs"].append(job)
            batch.atomic_json(path, raw)
        return raw

    def gates(self, path):
        self.events.append("prepare_gates")
        raw = batch.read_json(path)
        if not any(job["phase"] == "gates" for job in raw["jobs"]):
            for index, name in enumerate(batch.LIVE_TESTS):
                raw["jobs"].append({"id": f"g{index}", "phase": "gates", "test_name": name,
                    "output": str(self.paths.overnight / f"g{index}")})
            batch.atomic_json(path, raw)
        return raw

    def phase(self, manifest, phase, *, exhaust=False):
        self.events.append("run_" + phase)
        journal = batch.Journal(manifest["state_dir"])
        old = journal.load()
        state = batch._state(manifest)
        state["session_started_epoch"] = state.get("session_started_epoch") or time.time()
        state["phase_started_epoch"].setdefault("sim", time.time())
        pending = [job for job in manifest["jobs"] if job["phase"] == phase and job["id"] not in old]
        expired = batch.remaining_seconds(state, phase) <= 0
        for number, job in enumerate(pending):
            if expired or (exhaust and number > 0):
                break
            skipped = bool(job.get("skip_reason"))
            result = {"status": "skipped" if skipped else "passed", "passed": not skipped,
                "executed": not skipped, "wall_seconds": 0. if skipped else .1,
                "profile": "skillnet_v1_3", "simulator_closed": True,
                "actual_makespan_seconds": 1., "simulator_pid": 123}
            journal.append(batch._completion(job, result))
        if exhaust:
            state["phase_started_epoch"]["sim"] = time.time()-18001
        state["status"] = "phase_budget_exhausted" if expired or exhaust else "phase_complete"
        batch.save_state(manifest, state)
        return state

    def mocks(self, *, phase=None):
        stack = __import__("contextlib").ExitStack()
        stack.enter_context(patch.object(final, "verify_planning_complete"))
        stack.enter_context(patch.object(final, "guard", return_value={"ok": True}))
        stack.enter_context(patch.object(final.prepare, "prepare_selection", side_effect=self.select))
        stack.enter_context(patch.object(final.prepare, "prepare_gates", side_effect=self.gates))
        stack.enter_context(patch.object(final.batch, "run_phase", side_effect=phase or self.phase))
        def regression(paths, cursor):
            self.events.append("regression")
            return {"status": "failed", "tests_run": 500, "skipped": 6, "total_seconds": 1.}
        stack.enter_context(patch.object(final, "run_regression", side_effect=regression))
        def archive(paths):
            self.events.append("archives")
            return {"status": "passed", "returncode": 0}
        stack.enter_context(patch.object(final, "check_archives", side_effect=archive))
        stack.enter_context(patch.object(final.summary, "rebuild", return_value={}))
        return stack

    def test_unfinished_m4_blocks_all_experiments_and_does_not_require_pass_flag(self):
        batch.atomic_json(self.paths.m4, {"finished": False, "M4_passed": True})
        with self.mocks():
            cursor = final.orchestrate(self.paths.base)
        self.assertEqual(cursor["status"], "blocked")
        self.assertFalse(self.events)

    def test_order_partial_m4_and_failed_regression_still_continue_to_five_gates(self):
        with self.mocks():
            cursor = final.orchestrate(self.paths.base)
        self.assertEqual(cursor["status"], "finished")
        self.assertEqual(self.events, ["select1", "run_sim", "select2", "run_sim", "regression",
                                      "prepare_gates", "run_gates", "archives"])
        self.assertEqual(cursor["stages"]["regression"]["status"], "failed")
        report = self.paths.report.read_text(encoding="utf-8")
        self.assertIn("| 20 | 15 | 5 | 15 | 0 |", report)
        self.assertIn("不能等同于完成20项真实仿真", report)

    def test_first_round_budget_exhaustion_skips_second_but_continues_independent_steps(self):
        def phase(manifest, kind):
            return self.phase(manifest, kind, exhaust=kind == "sim")
        with self.mocks(phase=phase):
            cursor = final.orchestrate(self.paths.base)
        self.assertEqual(cursor["status"], "finished")
        self.assertNotIn("select2", self.events)
        self.assertIn("run_gates", self.events)
        self.assertEqual(cursor["stages"]["sim_round_2"]["status"], "not_prepared")

    def test_prepared_second_round_is_resumed_even_after_shared_budget_expired(self):
        self.select(1, 20261116, self.paths.manifest)
        manifest = batch.load_manifest(self.paths.manifest)
        self.phase(manifest, "sim")
        self.select(2, 20261117, self.paths.manifest)
        state_path = Path(manifest["state_dir"]) / "state.json"
        state = batch.read_json(state_path)
        state["phase_started_epoch"]["sim"] = time.time()-18001
        batch.atomic_json(state_path, state)
        self.events.clear()
        with self.mocks():
            cursor = final.orchestrate(self.paths.base)
        self.assertEqual(cursor["status"], "finished")
        self.assertNotIn("select2", self.events)
        self.assertEqual(self.events.count("run_sim"), 1)
        self.assertEqual(cursor["stages"]["sim_round_2"]["status"], "phase_budget_exhausted")

    def test_global_stop_in_any_session_or_own_cursor_is_persistent(self):
        other_state = self.paths.overnight / "m4_state/state.json"
        batch.atomic_json(other_state, {"global_stop": "三次启动失败"})
        self.assertIn("三次启动失败", final.global_stop_reason(self.paths))
        other_state.unlink()
        batch.atomic_json(self.paths.cursor, {"global_stop": "归档不符"})
        self.assertEqual(final.global_stop_reason(self.paths), "归档不符")

    def test_rebuild_is_idempotent_and_preserves_other_report_sections(self):
        self.select(1, 20261116, self.paths.manifest)
        self.phase(batch.load_manifest(self.paths.manifest), "sim")
        batch.atomic_text(self.paths.report, "原有M3与M4报告\n")
        with patch.object(final.summary, "rebuild", return_value={}), patch.object(final.batch, "run_phase") as run:
            first = final.rebuild(self.paths)
            final.rebuild(self.paths)
        run.assert_not_called()
        self.assertEqual(first["round1"]["executed"], 15)
        self.assertEqual(first["round1"]["skipped"], 5)
        report = self.paths.report.read_text(encoding="utf-8")
        self.assertTrue(report.startswith("原有M3与M4报告"))
        self.assertEqual(report.count("<!-- final-batches:step6-round1:start -->"), 1)
        self.assertEqual(report.count("<!-- final-batches:step7:start -->"), 1)

    def test_rebuild_recovers_round_and_regression_facts_without_cursor_stages(self):
        self.select(1, 20261116, self.paths.manifest)
        self.phase(batch.load_manifest(self.paths.manifest), "sim")
        batch.atomic_json(self.paths.regression, {"status": "failed", "tests_run": 503, "skipped": 6})
        with patch.object(final.summary, "rebuild", return_value={}):
            final.rebuild(self.paths, cursor={"stages": {}})
        report = self.paths.report.read_text(encoding="utf-8")
        self.assertIn("all_slots_recorded", report)
        self.assertIn("额外完整回归：failed；测试数503", report)

    def test_exact213_committed_plan_definitions_are_required(self):
        jobs, records = [], {}
        for index in range(213):
            job = {"id": str(index), "phase": "plan", "cohort": "precheck" if index < 13 else "planning_200",
                   "definition_sha256": "same"}
            jobs.append(job)
            records[str(index)] = {"definition_sha256": "same", "status": "budget"}
        final.verify_planning_complete({"jobs": jobs}, records)
        records.pop("212")
        with self.assertRaises(final.Blocked):
            final.verify_planning_complete({"jobs": jobs}, records)

    def test_completed_failed_regression_is_adopted_without_another_process(self):
        batch.atomic_json(self.paths.regression, {"status": "failed", "tests_run": 500, "finished": "past"})
        cursor = {"active_regression": None}
        with patch.object(final.batch, "OwnedProcess") as owner:
            record = final.run_regression(self.paths, cursor)
        owner.assert_not_called()
        self.assertEqual(record["status"], "failed")

    def test_low_disk_stops_owned_regression_and_commits_terminal_evidence_before_global_stop(self):
        cursor = final.load_cursor(self.paths)
        owner = Mock(pid=1234, birth="owned")
        exit_code = [None]
        owner.process.poll.side_effect = lambda: exit_code[0]
        owner.release.side_effect = lambda: batch.atomic_json(self.paths.regression, {"status": "running", "tests_run": 10})
        owner.terminate.side_effect = lambda: exit_code.__setitem__(0, 124)
        with patch.object(final.batch, "OwnedProcess", return_value=owner), \
             patch.object(final.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaises(final.GlobalStop):
                final.run_regression(self.paths, cursor)
        owner.terminate.assert_called_once()
        owner.close.assert_called_once()
        self.assertIsNone(cursor["active_regression"])
        self.assertEqual(cursor["stages"]["regression"]["status"], "stopped")
        self.assertEqual(batch.read_json(self.paths.regression)["status"], "stopped")

    def test_regression_worker_invokes_exact_harness_with_live_env_removed(self):
        start = self.directory / "start"
        batch.atomic_text(start, "assigned")
        specification = self.directory / "regression_spec.json"
        batch.atomic_json(specification, {"result": str(self.paths.regression), "start_signal": str(start)})
        with patch.dict(final.os.environ, {"SKILLNET_LIVE_SIM": "1", "SKILLNET_LIVE_MODEL": "1"}), \
             patch.object(final.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run:
            self.assertEqual(final.regression_worker(specification), 1)
        command = run.call_args.args[0]
        self.assertIn("tools/overnight_regression.py", command)
        self.assertIn("--worker", command)
        self.assertIn("final_after_sim", command)
        self.assertNotIn("SKILLNET_LIVE_SIM", run.call_args.kwargs["env"])
        self.assertNotIn("SKILLNET_LIVE_MODEL", run.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
