"""Durable unattended orchestration tests; no planner search or simulator."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import overnight_batch as batch


class BatchFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="batch_test_", dir=batch.ROOT / "out")
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def manifest(self, jobs=None, **settings):
        data = {"schema_version": 1, "session_id": self.directory.name,
            "state_dir": str(self.directory / "state"), "report_dir": str(self.directory / "reports"),
            "jobs": jobs or [{"id": "plan-1", "phase": "plan", "family": "F1",
                "task": "tasks/s1_parallel_place.toml", "output": str(self.directory / "plan-1")}]}
        data.update(settings)
        path = self.directory / "manifest.json"
        batch.atomic_json(path, data)
        return batch.load_manifest(path)

    def row(self, job_id="job-1", **values):
        result = {"id": job_id, "phase": "plan", "family": "F1", "cohort": "planning_200",
            "status": "solved", "passed": True, "output": str(self.directory / job_id)}
        result.update(values)
        return result

    def owner_factory(self, outcomes=None, *, hanging=False):
        created = []
        outcomes = outcomes or {}
        fixture = self

        class Owner:
            def __init__(self, command, log_path, start_signal):
                self.spec = batch.read_json(command[-1])
                self.job = self.spec["job"]
                self.pid, self.birth = 10000 + len(created), "test-birth"
                self.returncode, self.closed, self.released = None, False, False
                self.process = SimpleNamespace(poll=lambda: self.returncode)
                created.append(self)

            def refresh(self):
                return [{"pid": self.pid, "parent_pid": os.getpid(), "birth": self.birth}]

            def release(self):
                state = batch.read_json(Path(fixture.manifest_state) / "state.json")
                fixture.assertEqual(state["active_job"]["pid"], self.pid)
                self.released = True
                if not hanging:
                    result = {"status": "solved", "passed": True,
                        "definition_sha256": self.job["definition_sha256"]}
                    result.update(outcomes.get(self.job["id"], {}))
                    batch.atomic_json(self.spec["result_path"], result)
                    self.returncode = 0 if result["passed"] else 1

            def terminate(self):
                self.returncode = 124

            def close(self):
                self.closed = True

        return Owner, created


class ManifestAndJournalTests(BatchFixture):
    def test_atomic_text_retries_one_permission_failure_then_replaces_same_fsynced_temp(self):
        destination = self.directory / "state.json"
        destination.write_bytes(b"old-state\n")
        original_replace, sources = os.replace, []
        def transient(source, target):
            sources.append(Path(source))
            self.assertEqual(destination.read_bytes(), b"old-state\n")
            self.assertEqual(Path(source).read_bytes(), b"new-state\n")
            if len(sources) == 1:
                raise PermissionError(5, "temporary Windows sharing denial")
            return original_replace(source, target)
        with patch.object(batch.os, "replace", side_effect=transient) as replace, \
             patch.object(batch.os, "fsync", wraps=os.fsync) as fsync, \
             patch.object(batch.time, "sleep") as sleep:
            batch.atomic_text(destination, "new-state\n")
        self.assertEqual(replace.call_count, 2)
        self.assertEqual(fsync.call_count, 1)
        self.assertEqual(sources[0], sources[1])
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(destination.read_bytes(), b"new-state\n")
        self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_atomic_text_permanent_permission_failure_is_bounded_and_preserves_old_target(self):
        destination = self.directory / "state.json"
        destination.write_bytes(b"old-state\n")
        error = PermissionError(5, "persistent Windows sharing denial")
        with patch.object(batch.os, "replace", side_effect=error) as replace, \
             patch.object(batch.time, "sleep") as sleep:
            with self.assertRaises(PermissionError) as caught:
                batch.atomic_text(destination, "new-state\n")
        self.assertIs(caught.exception, error)
        self.assertEqual(replace.call_count, 8)
        self.assertEqual(sleep.call_count, 7)
        self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), 1.)
        self.assertTrue(all(0 < call.args[0] <= .2 for call in sleep.call_args_list))
        self.assertEqual(destination.read_bytes(), b"old-state\n")
        self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_atomic_text_does_not_retry_other_io_failures(self):
        destination = self.directory / "state.json"
        destination.write_bytes(b"old-state\n")
        with patch.object(batch.os, "replace", side_effect=FileNotFoundError("missing parent")) as replace, \
             patch.object(batch.time, "sleep") as sleep:
            with self.assertRaises(FileNotFoundError):
                batch.atomic_text(destination, "new-state\n")
        self.assertEqual(replace.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(destination.read_bytes(), b"old-state\n")

    def test_paths_port_and_default_timeout_are_confined(self):
        manifest = self.manifest()
        self.assertEqual(manifest["port"], 23210)
        self.assertEqual(manifest["jobs"][0]["timeout_seconds"], 1200)
        self.assertEqual(manifest["phase_budgets_seconds"]["sim"], 18000)
        self.assertTrue(Path(manifest["jobs"][0]["output"]).is_relative_to(batch.ROOT))
        with self.assertRaises(ValueError):
            self.manifest(port=23010)
        with self.assertRaises(ValueError):
            self.manifest(report_dir=str(batch.ROOT.parent / "foreign"))

    def test_limits_and_live_names_cannot_be_broadened(self):
        with self.assertRaises(ValueError):
            self.manifest(phase_budgets_seconds={"sim": 18001})
        with self.assertRaises(ValueError):
            self.manifest(jobs=[{"id": "bad", "phase": "gates", "test_name": "tests"}])
        with self.assertRaises(ValueError):
            self.manifest(jobs=[{"id": "bad", "phase": "plan", "task": "tasks/s1_parallel_place.toml",
                                "timeout_seconds": 1201}])
        with self.assertRaises(ValueError):
            self.manifest(jobs=[{"id": "bad", "phase": "plan", "task": "tasks/s1_parallel_place.toml",
                                "timeout_seconds": None}])

    def test_prepared_task_sha_is_enforced_even_without_journal(self):
        task = self.directory / "task.toml"
        task.write_text('name="first"\n', encoding="utf-8")
        manifest = self.manifest(jobs=[{"id": "immutable", "phase": "plan", "task": str(task),
            "prepared_task_sha256": batch.sha256(task), "output": str(self.directory / "immutable")}])
        task.write_text('name="modified"\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            batch.load_manifest(manifest["manifest_path"])

    def test_completion_is_fsynced_and_conflicting_overwrite_rejected(self):
        journal = batch.Journal(self.directory / "journal")
        with patch.object(batch.os, "fsync", wraps=os.fsync) as fsync:
            journal.append(self.row())
        fsync.assert_called()
        before = journal.paths()[0].read_bytes()
        journal.append(self.row())
        self.assertEqual(journal.paths()[0].read_bytes(), before)
        with self.assertRaises(ValueError):
            journal.append(self.row(status="failed"))

    def test_torn_tail_is_preserved_and_new_completion_rotates(self):
        journal = batch.Journal(self.directory / "journal")
        journal.append(self.row("first"))
        original = journal.paths()[0]
        with original.open("ab") as stream:
            stream.write(b'{"id":"unfinished"')
        before = original.read_bytes()
        journal.append(self.row("second"))
        self.assertEqual(set(journal.load()), {"first", "second"})
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(len(journal.paths()), 2)
        self.assertTrue(journal.warnings)

    def test_committed_corruption_is_not_silently_ignored(self):
        journal = batch.Journal(self.directory / "journal")
        (journal.directory / "completed.jsonl").write_bytes(b"broken\n")
        with self.assertRaises(ValueError):
            journal.load()

    def test_atomic_rebuild_uses_journal_and_inclusive_second_selection(self):
        jobs = [{"id": f"sim-{n}", "phase": "sim", "selection": n, "family": "F1",
                 "task": "tasks/s1_parallel_place.toml", "output": str(self.directory / f"sim-{n}")}
                for n in (1, 2)]
        manifest = self.manifest(jobs=jobs)
        journal = batch.Journal(manifest["state_dir"])
        journal.append(self.row("precheck", cohort="precheck"))
        journal.append(self.row("plan", reason="中文,逗号\n换行", passed=False, status="budget"))
        journal.append(self.row("sim-1", phase="sim", selection=1))
        journal.append(self.row("sim-2", phase="sim", selection=2))
        batch.rebuild_reports(manifest, journal)
        report = Path(manifest["report_dir"])
        def rows(name):
            return list(csv.DictReader(io.StringIO((report / name).read_text(encoding="utf-8"))))
        self.assertEqual(len(rows("planning_precheck.csv")), 1)
        self.assertEqual(rows("planning_200.csv")[0]["reason"], "中文,逗号\n换行")
        self.assertEqual(len(rows("sim_20.csv")), 1)
        self.assertEqual(len(rows("sim_40.csv")), 2)
        self.assertFalse(list(report.glob("*.tmp")))

    def test_simulation_csv_rebuild_preserves_json_phase_steps_feedback_and_qualification(self):
        jobs = [{"id": f"sim-{selection}", "phase": "sim", "selection": selection, "family": "F2",
                 "task": "tasks/s1_parallel_place.toml", "output": str(self.directory / f"sim-{selection}")}
                for selection in (1, 2)]
        manifest = self.manifest(jobs=jobs)
        journal = batch.Journal(manifest["state_dir"])
        metrics = {"phase_seconds": {"planning": .25, "real_planning": 1.5, "execute": 8.},
            "step_durations": [{"step": 1, "action": "pick(B, b1, stock_r1)",
                "estimated_seconds": 3.2, "actual_seconds": 4.5, "estimated_actual_ratio": 3.2/4.5}],
            "feedback": ['运动失败，原因：碰撞 "B手掌"；已加cut\n重新搜索。', "reach(B, goal) → VALID"],
            "qualification_calls": 2, "tpg_seconds": 1.25,
            "execution_wall_seconds": 8.5, "executor_wall_seconds": 10.75}
        empty_metrics = {"phase_seconds": {}, "step_durations": [], "feedback": [], "qualification_calls": 0}
        journal.append(self.row("sim-1", phase="sim", selection=1, makespan=10.25, **metrics))
        journal.append(self.row("sim-2", phase="sim", selection=2, makespan=12., **empty_metrics))
        before = {path: path.read_bytes() for path in journal.paths()}
        manifest_before = Path(manifest["manifest_path"]).read_bytes()
        batch.rebuild_reports(manifest, journal)
        report = Path(manifest["report_dir"])
        for name, expected_count in (("sim_20.csv", 1), ("sim_40.csv", 2)):
            with (report / name).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), expected_count)
            for row in rows:
                expected = metrics if row["id"] == "sim-1" else empty_metrics
                for field, value in expected.items():
                    self.assertEqual(json.loads(row[field]), value)
                if row["id"] == "sim-2":
                    for field in ("tpg_seconds", "execution_wall_seconds", "executor_wall_seconds"):
                        self.assertEqual(row[field], "")
            self.assertEqual(float(rows[0]["makespan"]), 10.25)
        self.assertEqual(before, {path: path.read_bytes() for path in journal.paths()})
        self.assertEqual(manifest_before, Path(manifest["manifest_path"]).read_bytes())

    def test_new_json_columns_are_null_when_missing_and_old_planning_columns_stay_compatible(self):
        old_columns = ("id", "family", "task", "status", "passed", "makespan", "expanded",
            "elapsed_seconds", "search_elapsed_seconds", "skills", "auxiliary", "auxiliary_skills",
            "failure_phase", "reason", "feedback_count", "compile_failed", "nominal_makespan_seconds",
            "actual_makespan_seconds", "actual_to_estimate_ratio", "wall_seconds", "selection", "executed", "output")
        self.assertEqual(batch.REPORT_COLUMNS[:len(old_columns)], old_columns)
        self.assertEqual(batch.REPORT_COLUMNS[-3:], ("tpg_seconds", "execution_wall_seconds", "executor_wall_seconds"))
        self.assertEqual(len(batch.REPORT_COLUMNS), len(set(batch.REPORT_COLUMNS)))
        raw = batch._csv_text([self.row("plan-1", status="budget", passed=False,
            makespan=0., skills=["pick", "place"])], batch.REPORT_COLUMNS)
        row = next(csv.DictReader(io.StringIO(raw)))
        self.assertEqual(row["id"], "plan-1")
        self.assertEqual(row["status"], "budget")
        self.assertEqual(row["makespan"], "0.0")
        self.assertEqual(json.loads(row["skills"]), ["pick", "place"])
        for field in ("phase_seconds", "step_durations", "feedback", "qualification_calls"):
            self.assertIsNone(json.loads(row[field]))
        for field in ("tpg_seconds", "execution_wall_seconds", "executor_wall_seconds"):
            self.assertEqual(row[field], "")

    def test_report_phase_comes_from_manifest_when_successful_runner_saved_complete(self):
        jobs = [{"id": f"sim-{selection}", "phase": "sim", "selection": selection, "family": "F2",
                 "task": "tasks/s1_parallel_place.toml", "output": str(self.directory / f"sim-{selection}")}
                for selection in (1, 2)]
        jobs.append({"id": "gate-1", "phase": "gates", "test_name": batch.LIVE_TESTS[0],
                     "output": str(self.directory / "gate-1")})
        jobs.append({"id": "plan-1", "phase": "plan", "cohort": "planning_200", "family": "F2",
                     "task": "tasks/s1_parallel_place.toml", "output": str(self.directory / "plan-1")})
        manifest = self.manifest(jobs=jobs)
        journal = batch.Journal(manifest["state_dir"])
        for job in manifest["jobs"]:
            result = {"phase": "complete", "status": "passed", "passed": True,
                "profile": "skillnet_v1_3", "simulator_closed": True,
                "actual_makespan_seconds": 3., "simulator_pid": 123}
            if job["phase"] == "plan":
                result.update(phase="plan", status="solved")
            journal.append(batch._completion(job, result))
        before = {path: path.read_bytes() for path in journal.paths()}
        manifest_before = Path(manifest["manifest_path"]).read_bytes()
        returned = batch.rebuild_reports(manifest, journal)
        self.assertEqual(returned["sim-1"]["phase"], "complete")
        self.assertEqual(returned["gate-1"]["phase"], "complete")
        report = Path(manifest["report_dir"])
        for name, expected in (("planning_200.csv", {"plan-1"}), ("sim_20.csv", {"sim-1"}),
                               ("sim_40.csv", {"sim-1", "sim-2"}), ("gates.csv", {"gate-1"})):
            with (report / name).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["id"] for row in rows}, expected, name)
        self.assertEqual(before, {path: path.read_bytes() for path in journal.paths()})
        self.assertEqual(manifest_before, Path(manifest["manifest_path"]).read_bytes())
        self.assertEqual(journal.load()["sim-1"]["phase"], "complete")


class ResumeAndBudgetTests(BatchFixture):
    def test_saved_wall_clock_is_not_reset_on_reload(self):
        manifest = self.manifest(session_budget_seconds=100,
                                 phase_budgets_seconds={"sim": 50})
        state = batch._state(manifest)
        state.update(session_started_epoch=1000, phase_started_epoch={"sim": 1010})
        batch.save_state(manifest, state)
        loaded = batch._state(manifest)
        self.assertEqual(batch.remaining_seconds(loaded, "sim", now=1040), 20)
        self.assertEqual(batch.remaining_seconds(loaded, "sim", now=1061), 0)
        self.assertEqual(loaded["session_started_epoch"], 1000)

    def test_resume_cannot_extend_phase_or_session_budget(self):
        manifest = self.manifest(session_budget_seconds=100,
                                 phase_budgets_seconds={"sim": 50})
        batch.save_state(manifest, batch._state(manifest))
        extended = dict(manifest, phase_budgets_seconds={**manifest["phase_budgets_seconds"], "sim": 51})
        with self.assertRaises(ValueError):
            batch._state(extended)
        with self.assertRaises(ValueError):
            batch._state(dict(manifest, session_budget_seconds=101))

    def test_completed_ids_are_skipped_and_results_survive_resume(self):
        manifest = self.manifest()
        self.manifest_state = manifest["state_dir"]
        factory, created = self.owner_factory()
        check = Mock(return_value={"ok": True})
        first = batch.run_phase(manifest, "plan", environment_check=check, process_factory=factory)
        self.assertEqual(first["status"], "phase_complete")
        self.assertEqual(len(created), 1)
        before = (Path(manifest["state_dir"]) / "completed.jsonl").read_bytes()
        second = batch.run_phase(manifest, "plan", environment_check=check, process_factory=factory)
        self.assertEqual(second["status"], "phase_complete")
        self.assertEqual(len(created), 1)
        self.assertEqual(check.call_count, 2)
        self.assertTrue(created[0].closed)
        self.assertEqual(before, (Path(manifest["state_dir"]) / "completed.jsonl").read_bytes())

    def test_changed_completed_definition_is_rejected_before_launch(self):
        manifest = self.manifest()
        journal = batch.Journal(manifest["state_dir"])
        journal.append(self.row("plan-1", definition_sha256="different"))
        factory = Mock()
        with self.assertRaises(ValueError):
            batch.run_phase(manifest, "plan", environment_check=Mock(), process_factory=factory)
        factory.assert_not_called()

    def test_crash_after_worker_commit_adopts_result_without_retry(self):
        manifest = self.manifest()
        job = manifest["jobs"][0]
        state = batch._state(manifest)
        result_path = self.directory / "recovered.json"
        batch.atomic_json(result_path, {"status": "solved", "passed": True,
                                      "definition_sha256": job["definition_sha256"]})
        state["active_job"] = {"id": job["id"], "job": job, "result_path": str(result_path),
                               "pid": 1234, "birth": "old", "started_epoch": 1000}
        journal = batch.Journal(manifest["state_dir"])
        batch._recover_active(manifest, state, journal)
        self.assertTrue(journal.load()[job["id"]]["recovered"])
        self.assertIsNone(state["active_job"])

    def test_incomplete_dead_worker_is_recorded_without_hidden_retry(self):
        manifest = self.manifest()
        job = manifest["jobs"][0]
        state = batch._state(manifest)
        state["active_job"] = {"id": job["id"], "job": job, "result_path": str(self.directory / "missing.json"),
                               "pid": 1234, "birth": "old", "started_epoch": time.time()}
        journal = batch.Journal(manifest["state_dir"])
        with patch.object(batch, "process_birth", return_value=None):
            batch._recover_active(manifest, state, journal)
        self.assertEqual(journal.load()[job["id"]]["status"], "interrupted")

    def test_live_previous_worker_is_not_terminated_or_duplicated(self):
        manifest = self.manifest()
        job = manifest["jobs"][0]
        state = batch._state(manifest)
        state["active_job"] = {"id": job["id"], "job": job, "result_path": str(self.directory / "missing.json"),
                               "pid": 1234, "birth": "old", "started_epoch": time.time()}
        with patch.object(batch, "process_birth", return_value="old"):
            with self.assertRaises(RuntimeError):
                batch._recover_active(manifest, state, batch.Journal(manifest["state_dir"]))

    def test_missing_creation_identity_does_not_make_dead_worker_look_live(self):
        manifest = self.manifest()
        job = manifest["jobs"][0]
        state = batch._state(manifest)
        state["active_job"] = {"id": job["id"], "job": job, "result_path": str(self.directory / "missing.json"),
                               "pid": 1234, "birth": None, "started_epoch": time.time()}
        journal = batch.Journal(manifest["state_dir"])
        with patch.object(batch, "process_birth", return_value=None):
            batch._recover_active(manifest, state, journal)
        self.assertEqual(journal.load()[job["id"]]["status"], "interrupted")

    def test_recovered_interruption_does_not_report_downtime_as_exact_runtime(self):
        manifest = self.manifest()
        job = manifest["jobs"][0]
        state = batch._state(manifest)
        state["active_job"] = {"id": job["id"], "job": job,
            "result_path": str(self.directory / "missing.json"), "pid": 1234, "birth": "old",
            "started_epoch": 1000., "elapsed_seconds": 22.469}
        journal = batch.Journal(manifest["state_dir"])
        with patch.object(batch, "process_birth", return_value=None), patch.object(batch.time, "time", return_value=5000.):
            batch._recover_active(manifest, state, journal)
        row = journal.load()[job["id"]]
        self.assertEqual(row["status"], "interrupted")
        self.assertIsNone(row["wall_seconds"])
        self.assertFalse(row["wall_seconds_is_exact"])
        self.assertEqual(row["last_observed_seconds"], 22.469)
        self.assertEqual(row["recovery_elapsed_upper_bound_seconds"], 4000.)

    def test_task_hard_timeout_uses_monotonic_clock_if_wall_clock_stands_still(self):
        manifest = self.manifest(jobs=[{"id": "plan-1", "phase": "plan",
            "task": "tasks/s1_parallel_place.toml", "output": str(self.directory / "plan-1"),
            "timeout_seconds": .1}])
        self.manifest_state = manifest["state_dir"]
        factory, created = self.owner_factory(hanging=True)
        clock = [0.]
        with patch.object(batch.time, "time", return_value=1000.), \
             patch.object(batch.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(batch.time, "sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0]+.2)), \
             patch.object(batch.shutil, "disk_usage", return_value=SimpleNamespace(free=10*1024**3)):
            batch.run_phase(manifest, "plan", poll_seconds=.1,
                environment_check=Mock(return_value={"ok": True}), process_factory=factory)
        row = batch.Journal(manifest["state_dir"]).load()["plan-1"]
        self.assertEqual(row["status"], "timeout")
        self.assertEqual(row["failure_phase"], "task_wall_timeout")
        self.assertEqual(created[0].returncode, 124)

    def test_phase_budget_terminates_owned_job_and_prevents_second_launch(self):
        jobs = [{"id": f"sim-{n}", "phase": "sim", "family": "F1", "task": "tasks/s1_parallel_place.toml",
                 "output": str(self.directory / f"sim-{n}")} for n in (1, 2)]
        manifest = self.manifest(jobs=jobs, phase_budgets_seconds={"sim": 1})
        self.manifest_state = manifest["state_dir"]
        factory, created = self.owner_factory(hanging=True)
        clock = [1000.]
        with patch.object(batch.time, "time", side_effect=lambda: clock[0]), \
             patch.object(batch.time, "sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0]+.6)), \
             patch.object(batch.shutil, "disk_usage", return_value=SimpleNamespace(free=10*1024**3)):
            state = batch.run_phase(manifest, "sim", poll_seconds=.5,
                environment_check=Mock(return_value={"ok": True}), process_factory=factory)
            again = batch.run_phase(manifest, "sim", environment_check=Mock(), process_factory=factory)
        self.assertEqual(state["status"], "phase_budget_exhausted")
        self.assertEqual(again["phase_started_epoch"]["sim"], 1000)
        self.assertEqual(len(created), 1)
        row = batch.Journal(manifest["state_dir"]).load()["sim-1"]
        self.assertEqual(row["status"], "timeout")
        self.assertEqual(row["failure_phase"], "phase_budget")
        self.assertTrue(created[0].closed)

    def test_three_consecutive_start_connect_failures_stop_and_reconcile(self):
        jobs = [{"id": f"sim-{n}", "phase": "sim", "task": "tasks/s1_parallel_place.toml",
                 "output": str(self.directory / f"sim-{n}")} for n in range(5)]
        manifest = self.manifest(jobs=jobs)
        self.manifest_state = manifest["state_dir"]
        outcomes = {job["id"]: {"status": "failed", "passed": False, "failure_phase": "simulator_connect"} for job in jobs}
        factory, created = self.owner_factory(outcomes)
        state = batch.run_phase(manifest, "sim", environment_check=Mock(return_value={"ok": True}), process_factory=factory)
        self.assertEqual(len(created), 3)
        self.assertEqual(state["status"], "global_stop")
        state["consecutive_start_connect_failures"] = 0
        state["global_stop"] = None
        batch._reconcile_failures(state, batch.Journal(manifest["state_dir"]).load())
        self.assertEqual(state["consecutive_start_connect_failures"], 3)
        self.assertTrue(state["global_stop"])

    def test_environment_stop_never_launches_or_marks_pending_job_complete(self):
        manifest = self.manifest()
        factory = Mock()
        state = batch.run_phase(manifest, "plan", environment_check=Mock(return_value={
            "ok": False, "global_stop": True, "reason": "C low"}), process_factory=factory)
        self.assertEqual(state["global_stop"], "C low")
        factory.assert_not_called()
        self.assertFalse(batch.Journal(manifest["state_dir"]).load())

    def test_final_job_archive_postcheck_cannot_report_phase_complete(self):
        manifest = self.manifest()
        self.manifest_state = manifest["state_dir"]
        factory, created = self.owner_factory()
        check = Mock(side_effect=[{"ok": True}, {"ok": False, "global_stop": True,
            "reason": "归档 SHA256 与基线不符"}])
        state = batch.run_phase(manifest, "plan", environment_check=check, process_factory=factory)
        self.assertEqual(state["status"], "global_stop")
        self.assertIn("归档", state["global_stop"])
        self.assertEqual(len(created), 1)
        self.assertTrue(batch.Journal(manifest["state_dir"]).load()["plan-1"]["passed"])

    def test_cap_expiring_during_preflight_does_not_launch_worker(self):
        manifest = self.manifest(phase_budgets_seconds={"plan": 1})
        clock = [1000.]
        def check(guard):
            clock[0] += 2
            return {"ok": True}
        factory = Mock()
        with patch.object(batch.time, "time", side_effect=lambda: clock[0]):
            state = batch.run_phase(manifest, "plan", environment_check=check, process_factory=factory)
        self.assertEqual(state["status"], "phase_budget_exhausted")
        factory.assert_not_called()

    def test_explicit_unavailable_sample_is_durable_skip_and_not_an_attempt(self):
        manifest = self.manifest(jobs=[{"id": "sim-missing", "phase": "sim", "family": "F5",
            "skip_reason": "本族独立规划 solved 样本不足 4 项", "output": str(self.directory / "missing")}])
        factory, check = Mock(), Mock()
        state = batch.run_phase(manifest, "sim", environment_check=check, process_factory=factory)
        row = batch.Journal(manifest["state_dir"]).load()["sim-missing"]
        self.assertEqual(row["status"], "skipped")
        self.assertFalse(row["executed"])
        factory.assert_not_called()
        check.assert_not_called()
        report = batch.read_json(Path(manifest["state_dir"]) / "report_status.json")
        self.assertEqual(report["requested"], 1)
        self.assertEqual(report["skipped"], 1)
        self.assertEqual(report["pending"], 0)

    def test_skip_does_not_reset_consecutive_start_failures(self):
        state = {}
        rows = {str(i): self.row(str(i), phase="sim", passed=False,
            failure_phase="simulator_start", status="failed") for i in (1, 2, 3)}
        rows = {"1": rows["1"], "skip": self.row("skip", phase="sim", status="skipped"),
                "2": rows["2"], "3": rows["3"]}
        batch._reconcile_failures(state, rows)
        self.assertEqual(state["consecutive_start_connect_failures"], 3)
        self.assertTrue(state["global_stop"])


class WorkerAndEnvironmentTests(BatchFixture):
    def test_plan_worker_calls_pipeline_without_proposal_or_witness(self):
        job = self.manifest()["jobs"][0]
        Path(job["output"]).mkdir()
        schedule = SimpleNamespace(steps=[], makespan=2., expanded=3, generated=4, elapsed=.1, rejected={})
        result = SimpleNamespace(status="budget", schedule=schedule, feedback=["search limit"])
        problem = SimpleNamespace(name="mock", hooks=SimpleNamespace(locations={}, slots={}))
        with patch("skillnet.domains.blocks.load_problem", return_value=problem), \
             patch("skillnet.pipeline.run", return_value=result) as run, \
             patch("skillnet.report.plan_to_dict", return_value={"unknown": []}):
            summary = batch.plan_job(job)
        self.assertEqual(run.call_args.args, (problem,))
        self.assertIsInstance(run.call_args.kwargs["decomposer"], batch.NoProposal)
        self.assertEqual(set(run.call_args.kwargs), {"decomposer"})
        self.assertIsNone(run.call_args.kwargs["decomposer"].decompose(problem))
        self.assertEqual(summary["status"], "budget")
        self.assertFalse(summary["proposal_enabled"])
        self.assertFalse(summary["witness_loaded"])
        self.assertTrue((Path(job["output"]) / "plan.json").exists())

    def test_sim_worker_preserves_runner_evidence_and_fixed_settings(self):
        job = self.manifest()["jobs"][0]
        with patch("simbridge.runner.run_task", return_value={"task": "mock", "passed": False,
                "status": "failed", "failure_phase": "compile", "motion_feedback_count": 2}) as run:
            result = batch.sim_job(job)
        self.assertEqual(run.call_args.kwargs["port"], 23210)
        self.assertFalse(run.call_args.kwargs["video"])
        self.assertTrue(run.call_args.kwargs["headless"])
        self.assertEqual(result["feedback_count"], 2)
        self.assertTrue(result["compile_failed"])

    def test_gate_runs_only_explicit_test_with_live_env_and_unchanged_acceptance(self):
        job = {"test_name": batch.LIVE_TESTS[0], "output": str(self.directory), "timeout_seconds": 1200}
        original = Mock(return_value={"passed": True, "status": "passed"})
        module = SimpleNamespace(run_task=original)
        result = SimpleNamespace(wasSuccessful=lambda: True, testsRun=1, failures=[], errors=[], skipped=[])
        def run_suite(suite):
            self.assertEqual(os.environ["SKILLNET_LIVE_SIM"], "1")
            module.run_task("mock.toml", out="old")
            return result
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(unittest.defaultTestLoader, "loadTestsFromName", return_value="suite") as loader, \
             patch.object(unittest, "TextTestRunner") as runner, \
             patch.object(batch.importlib, "import_module", return_value=module):
            runner.return_value.run.side_effect = run_suite
            actual = batch.gate_job(job)
        loader.assert_called_once_with(job["test_name"])
        self.assertTrue(actual["passed"])
        self.assertIs(module.run_task, original)
        self.assertEqual(original.call_args.kwargs["port"], 23210)
        self.assertFalse(original.call_args.kwargs["video"])

    def test_low_disk_stops_before_archive_or_process_queries(self):
        archive = Mock()
        with patch.object(batch.shutil, "disk_usage", return_value=SimpleNamespace(free=1024)):
            result = batch.preflight(archive)
        self.assertFalse(result["ok"])
        self.assertTrue(result["global_stop"])
        archive.check.assert_not_called()

    def test_foreign_simulator_is_reported_without_any_kill(self):
        archive = Mock()
        archive.check.return_value = {"passed": True}
        with patch.object(batch.shutil, "disk_usage", return_value=SimpleNamespace(free=10*1024**3)), \
             patch.object(batch, "process_snapshot", return_value=[{"pid": 123, "name": "coppeliaSim.exe"}]), \
             patch.object(batch.subprocess, "Popen") as launch:
            result = batch.preflight(archive)
        self.assertFalse(result["ok"])
        self.assertFalse(result["global_stop"])
        launch.assert_not_called()
        self.assertEqual(result["processes"][0]["pid"], 123)

    @unittest.skipUnless(os.name == "nt", "Windows default-suite process probe")
    def test_full_test_probe_includes_background_regression_supervisor(self):
        with patch.object(batch.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout='[{"ProcessId":123,"ParentProcessId":99}]')) as run:
            rows = batch.full_test_processes()
        self.assertEqual(rows[0]["ProcessId"], 123)
        self.assertIn("overnight_regression", run.call_args.args[0][-1])

    @unittest.skipUnless(os.name == "nt", "Windows native ownership")
    def test_native_inventory_records_our_pid_and_creation_identity(self):
        rows = batch.process_snapshot()
        self.assertIn(os.getpid(), {row["pid"] for row in rows})
        self.assertIsNotNone(batch.process_birth(os.getpid()))

    @unittest.skipUnless(os.name == "nt", "Windows Job Object ownership")
    def test_native_job_terminates_only_assigned_worker_and_descendant(self):
        start_signal = self.directory / "start"
        child_pid_path = self.directory / "child.pid"
        code = ("import pathlib,subprocess,sys,time; "
            "p=pathlib.Path(sys.argv[1]); "
            "exec('while not p.exists(): time.sleep(.02)'); "
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
            "pathlib.Path(sys.argv[2]).write_text(str(child.pid)); time.sleep(30)")
        foreign = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        owner = None
        try:
            owner = batch.OwnedProcess([sys.executable, "-c", code, str(start_signal), str(child_pid_path)],
                self.directory / "native.log", start_signal)
            self.assertFalse(start_signal.exists())
            owner.release()
            deadline = time.monotonic() + 5
            while not child_pid_path.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(child_pid_path.exists())
            child_pid = int(child_pid_path.read_text())
            owned = owner.refresh()
            self.assertIn(child_pid, {row["pid"] for row in owned})
            self.assertEqual(next(row for row in owned if row["pid"] == child_pid)["parent_pid"], owner.pid)
            owner.terminate()
            owner.close()
            owner = None
            self.assertIsNone(batch.process_birth(child_pid))
            self.assertIsNone(foreign.poll())
        finally:
            if owner is not None:
                owner.close()
            foreign.terminate()
            foreign.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
