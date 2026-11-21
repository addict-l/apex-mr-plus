"""Isolated workflow and one-job routing tests; never start a simulator/solver."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import motion_v135_workflow as flow
from tools import motion_v135_single as single
from tools import overnight_batch as old

PROJECT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        (PROJECT/"out").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT/"out")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.base = self.project/"results/sim_v1_3/motion_v135"
        self.source = self.project/"history_manifests"
        self.source.mkdir()
        self.inputs = self.project/"inputs"
        self.inputs.mkdir()
        self.code = self.project/"skillnet/pipeline.py"
        self.code.parent.mkdir(parents=True)
        self.code.write_text("# frozen fixture implementation\n", encoding="utf-8")
        self.registered_worker_files = flow.WORKER_FILES
        # Real registration covers the entire suite. The isolated state-machine
        # seam needs only these mocked files, rather than hundreds of copies
        # and repeated large proof maps in every temporary project.
        self.fixture_worker_files = (
            "tools/motion_v135_workflow.py", "tools/motion_v135_control.py",
            "tools/motion_v135_report.py", "tests/test_motion_v135_report.py",
            "tools/motion_v135_preflight.py", "tools/motion_v135_regression.py",
            "tools/motion_v135_single.py", "tools/motion_v135_detached.py",
            "tools/motion_v135_diagnostics.py", "tools/motion_v135_fast_idle.py")
        for name in self.fixture_worker_files:
            target = self.project/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# fixture worker\n", encoding="utf-8")
        self.counts = {"planning200": 2, "fixed20": 2, "walls3": 1, "gates": 1}
        self.started = []
        self.fake_results = lambda job: {}
        self.patches = [patch.object(flow, "ROOT", self.project),
                        patch.object(flow, "BASE", self.base),
                        patch.object(flow, "SOURCE", self.source),
                        patch.object(flow, "COUNTS", self.counts),
                        patch.object(flow, "WORKER_FILES", self.fixture_worker_files),
                        patch.object(old, "ROOT", self.project),
                        patch.object(flow.control, "version", side_effect=lambda: {
                            "skillnet/pipeline.py": old.sha256(self.code)}),
                        patch.object(flow.control, "audit", return_value={"passed": True}),
                        patch.object(flow.control, "record"),
                        patch.object(old, "preflight", return_value={"ok": True, "global_stop": False,
                            "archive": {"passed": True}})]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.baselines = {}
        for stage in flow.STAGES:
            jobs = []
            for index in range(self.counts[stage]):
                task = self.inputs/(stage+str(index)+".qualified.toml")
                task.write_text('name="fixture_'+stage+str(index)+'"\n', encoding="utf-8")
                job = {"id": "baseline-"+stage+"-"+str(index), "phase": flow.PHASE[stage],
                    "kind": "plan" if stage == "planning200" else ("gate" if stage == "gates" else "sim"),
                    "family": "F1" if stage != "walls3" else "F3", "index": index,
                    "source_job_id": "original-source-"+stage+"-"+str(index),
                    "output": str(self.project/"old_output"/stage/str(index)),
                    "timeout_seconds": 60. if stage == "planning200" else 1200.,
                    "planning_code_sha256": {"planner": "baseline-hash"},
                    "motion_code_sha256": {"compiler": "baseline-motion"},
                    "experiment_namespace": "c_wall_v134",
                    "experiment_selection_sha256": "old-registration"}
                if stage == "gates":
                    job["test_name"] = old.LIVE_TESTS[0]
                else:
                    job.update(task=str(task), prepared_task_sha256=old.sha256(task),
                               source_task_sha256=old.sha256(task))
                jobs.append(job)
            manifest = {"schema_version": 1, "session_id": "old_"+stage,
                "port": 23210, "state_dir": str(self.project/"old_states"/stage),
                "report_dir": str(self.project/"old_reports"/stage),
                "session_budget_seconds": None,
                "phase_budgets_seconds": {"plan": None, "sim": 18000., "gates": None}, "jobs": jobs}
            path = self.source/(flow.SOURCE_STAGE[stage]+".json")
            old.atomic_json(path, manifest)
            self.baselines[stage] = deepcopy(manifest)

    def prepare(self, attempt="formal_a1"):
        proof = self.formal_proof()
        answer = flow.prepare(attempt, preflight_proofs=proof)
        self.assertEqual(answer["status"], "prepared")
        return flow.attempt_root(attempt)

    def formal_proof(self):
        from tools.motion_v135_preflight import STEPS
        path = self.base/"m6/current_fixture/state/proofs.json"
        if path.exists():
            return path
        evidence = self.base/"m6/current_fixture/evidence.json"
        old.atomic_json(evidence, {"fixture": "complete detached evidence"})
        version = {"core": flow.control.version(), "tools": flow.worker_version()}
        old.atomic_json(path, {"formal_ready": True, "version": version,
            "required_steps": list(STEPS), "steps": {step: {
                "passed": True, "completed": True, "version": deepcopy(version),
                "evidence_files": {str(evidence): old.sha256(evidence)}} for step in STEPS}})
        return path

    def canonical(self, root, stage="planning200"):
        return old.load_manifest(root/"manifests"/(stage+".json"))

    def commit(self, manifest, job, **extra):
        result = {"id": job["id"], "phase": job["phase"], "definition_sha256": job["definition_sha256"],
                  "logical_job_id": job["logical_job_id"], "status": "solved", "passed": True,
                  "source_job_id": job["source_job_id"], "task_sha256": job.get("task_sha256"),
                  "completed_at": old.utc_now(), **extra}
        old.Journal(manifest["state_dir"]).append(result)
        return result

    def owner(self, command, log_path, start_signal):
        test = self
        specification = Path(command[-1])
        spec = old.read_json(specification)
        test.assertEqual(Path(command[3]), Path(flow.__file__).resolve())
        test.started.append(deepcopy(spec["job"]))
        class FakeOwner:
            pid = 999999
            birth = "fixture-process"
            process = SimpleNamespace(returncode=0, poll=lambda: 0)
            def refresh(self): return []
            def release(self):
                job = spec["job"]
                result = {"status": "solved" if job["phase"] == "plan" else "passed",
                    "passed": True, "definition_sha256": job["definition_sha256"],
                    "logical_job_id": job["logical_job_id"], "run_attempt": job["run_attempt"],
                    "worker_wall_seconds": .01}
                result.update(test.fake_results(job))
                old.atomic_json(spec["result_path"], result)
            def close(self): pass
        return FakeOwner()

    def test_prepare_copies_exact_identity_sha_budgets_and_preserves_baseline_fingerprints(self):
        before = {p.name: p.read_bytes() for p in self.source.glob("*.json")}
        with patch.object(old, "run_phase") as runs, patch.object(old, "OwnedProcess") as processes:
            root = self.prepare()
        runs.assert_not_called()
        processes.assert_not_called()
        for stage in flow.STAGES:
            actual = old.read_json(root/"manifests"/(stage+".json"))
            source = self.baselines[stage]
            self.assertEqual(actual["phase_budgets_seconds"], source["phase_budgets_seconds"])
            self.assertEqual(actual["session_budget_seconds"], source["session_budget_seconds"])
            for original, job in zip(source["jobs"], actual["jobs"]):
                self.assertEqual(job["source_job_id"], original["source_job_id"])
                self.assertEqual(job["baseline_job_id"], original["id"])
                self.assertEqual(job.get("prepared_task_sha256"), original.get("prepared_task_sha256"))
                self.assertEqual(job.get("task"), original.get("task"))
                self.assertEqual(job["timeout_seconds"], original["timeout_seconds"])
                self.assertEqual(job["baseline_planning_code_sha256"], original["planning_code_sha256"])
                self.assertEqual(job["baseline_motion_code_sha256"], original["motion_code_sha256"])
                self.assertNotIn("planning_code_sha256", job)
                self.assertTrue(Path(job["output"]).is_relative_to(root))
                self.assertNotEqual(job["id"], original["id"])
            self.assertTrue(Path(actual["state_dir"]).is_relative_to(root))
            self.assertTrue(Path(actual["report_dir"]).is_relative_to(root))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.glob("*.json")})
        self.assertEqual((root/"code_snapshot/skillnet/pipeline.py").read_bytes(), self.code.read_bytes())
        flow.control.audit.assert_called()

    def test_prepared_attempt_is_idempotent_and_cannot_accept_changed_code(self):
        root = self.prepare()
        selection = (root/"selection.json").read_bytes()
        self.assertEqual(flow.prepare("formal_a1")["status"], "already_prepared")
        self.code.write_text("# changed implementation\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "快照|混版本"):
            flow.verify(root)
        with self.assertRaises(ValueError):
            flow.prepare("formal_a1")
        self.assertEqual((root/"selection.json").read_bytes(), selection)

    def test_changed_input_sha_is_rejected_without_mutating_source_manifest(self):
        root = self.prepare()
        source = self.baselines["planning200"]["jobs"][0]
        Path(source["task"]).write_text("changed input\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256|SHA"):
            flow.verify(root)

    def test_completed_jobs_are_skipped_and_only_pending_job_is_spawned(self):
        root = self.prepare()
        manifest = self.canonical(root)
        self.commit(manifest, manifest["jobs"][0])
        with patch.object(old, "OwnedProcess", side_effect=self.owner):
            result = flow.run("formal_a1", ("planning200",))
            second = flow.run("formal_a1", ("planning200",))
        self.assertEqual(result["status"], "finished")
        self.assertEqual(second["status"], "finished")
        self.assertEqual([j["id"] for j in self.started], [manifest["jobs"][1]["id"]])
        self.assertEqual(flow.rebuild("formal_a1")["stages"]["planning200"]["committed"], 2)
        flow.control.audit.assert_called()

    def test_dead_uncommitted_attempt_gets_new_output_and_keeps_original_evidence(self):
        root = self.prepare()
        manifest = self.canonical(root)
        first = manifest["jobs"][0]
        original = Path(first["output"])/"partial.txt"
        original.parent.mkdir(parents=True)
        original.write_text("keep interrupted scene evidence", encoding="utf-8")
        state = old._state(manifest)
        state["active_job"] = {"id": first["id"], "job": first, "pid": 999999,
            "birth": "dead", "result_path": str(Path(first["output"])/"missing.json"),
            "started_epoch": time.time()-5, "elapsed_seconds": 5.}
        old.save_state(manifest, state)
        with patch.object(old, "process_birth", return_value=None), patch.object(old, "OwnedProcess", side_effect=self.owner):
            result = flow.run("formal_a1", ("planning200",))
        self.assertEqual(result["status"], "finished")
        journal = old.Journal(manifest["state_dir"]).load()
        self.assertEqual(journal[first["id"]]["status"], "interrupted")
        self.assertEqual(original.read_text(encoding="utf-8"), "keep interrupted scene evidence")
        replacement = next(j for j in self.started if j["logical_job_id"] == first["id"])
        self.assertEqual(replacement["id"], first["id"]+"-resume1")
        self.assertNotEqual(replacement["output"], first["output"])
        self.assertTrue(Path(replacement["output"]).is_relative_to(root))
        self.assertEqual(replacement["task_sha256"], first["task_sha256"])
        self.assertEqual(replacement["source_job_id"], first["source_job_id"])
        self.assertTrue((root/"recovery/planning200"/(replacement["id"]+".json")).exists())
        counts = flow.rebuild("formal_a1")["stages"]["planning200"]
        self.assertEqual((counts["committed"], counts["attempt_rows"], counts["interrupted_attempts"]), (2, 3, 1))

    def test_second_interruption_is_still_the_same_logical_task(self):
        root = self.prepare()
        manifest = self.canonical(root)
        first = manifest["jobs"][0]
        self.commit(manifest, first, status="interrupted", passed=False)
        raw = old.read_json(root/"manifests/planning200.json")
        retry = deepcopy(raw["jobs"][0])
        retry.update(id=first["id"]+"-resume1", run_attempt=1,
                     output=str(root/"recovery_outputs/planning200/previous"))
        raw["jobs"] = [retry]
        retry_manifest = root/"previous_execution.json"
        old.atomic_json(retry_manifest, raw)
        job = old.load_manifest(retry_manifest)["jobs"][0]
        state = old._state(manifest)
        state["active_job"] = {"id": job["id"], "job": job, "pid": 999999,
            "birth": "dead", "result_path": str(Path(job["output"])/"missing.json"),
            "started_epoch": time.time()-5, "elapsed_seconds": 5.}
        old.save_state(manifest, state)
        with patch.object(old, "process_birth", return_value=None), patch.object(old, "OwnedProcess", side_effect=self.owner):
            result = flow.run("formal_a1", ("planning200",))
        self.assertEqual(result["status"], "finished")
        resumed = next(j for j in self.started if j["logical_job_id"] == first["id"])
        self.assertEqual(resumed["id"], first["id"]+"-resume2")
        self.assertEqual(flow.rebuild("formal_a1")["stages"]["planning200"]["interrupted_attempts"], 2)

    def test_global_stop_prevents_spawn_and_is_persistent(self):
        self.prepare()
        old.atomic_json(self.base/"global_stop.json", {"reason": "C drive below threshold"})
        with patch.object(old, "OwnedProcess") as processes:
            result = flow.run("formal_a1")
        processes.assert_not_called()
        self.assertEqual(result["status"], "global_stop")
        self.assertTrue((self.base/"global_stop.json").exists())

    def test_three_consecutive_startup_failures_across_native_stages_stop_before_gates(self):
        self.prepare()
        self.fake_results = lambda job: ({"status": "failed", "passed": False,
            "failure_phase": "simulator_start", "reason": "mock startup failure"}
            if job["phase"] != "plan" else {})
        with patch.object(old, "OwnedProcess", side_effect=self.owner):
            result = flow.run("formal_a1")
        self.assertEqual(result["status"], "global_stop")
        self.assertEqual(len(self.started), 5)  # two plans, two fixed20, first wall
        self.assertFalse(any(job["phase"] == "gates" for job in self.started))
        self.assertIn("三", old.read_json(self.base/"global_stop.json")["reason"])
        self.assertEqual(result["counts"]["fixed20"]["statuses"], {"failed": 2})

    def test_phase_complete_without_committed_results_cannot_claim_finished(self):
        self.prepare()
        with patch.object(flow, "run_stage", return_value={"status": "phase_complete"}):
            result = flow.run("formal_a1", ("planning200",))
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["counts"]["planning200"]["pending"], 2)

    def test_history_audit_failure_stops_before_any_worker(self):
        self.prepare()
        flow.control.audit.return_value = {"passed": False, "changed": ["history/input.toml"]}
        with patch.object(old, "OwnedProcess") as processes:
            result = flow.run("formal_a1")
        processes.assert_not_called()
        self.assertEqual(result["status"], "global_stop")
        self.assertTrue((self.base/"global_stop.json").exists())

    def test_prepare_without_current_formal_ready_proof_refuses_before_creating_attempt(self):
        with patch.object(old, "OwnedProcess") as processes:
            with self.assertRaisesRegex(ValueError, "preflight|前置"):
                flow.prepare("missing_proof")
        processes.assert_not_called()
        self.assertFalse((flow.attempt_root("missing_proof")/"selection.json").exists())

    def test_formal_tool_manifest_covers_new_execution_and_report_helpers(self):
        required = {"tools/motion_v135_report.py", "tests/test_motion_v135_report.py",
                    "tools/motion_v135_preflight.py", "tools/motion_v135_regression.py",
                    "tools/motion_v135_single.py", "tools/motion_v135_detached.py",
                    "tools/motion_v135_diagnostics.py", "tools/motion_v135_fast_idle.py"}
        self.assertTrue(required <= set(flow.worker_version()))
        self.assertTrue(required <= set(self.registered_worker_files))

    def test_formal_ready_cannot_override_a_failed_required_step(self):
        proof = self.formal_proof()
        data = old.read_json(proof)
        data["steps"]["default"]["passed"] = False
        old.atomic_json(proof, data)
        with self.assertRaisesRegex(ValueError, "前置|preflight|default"):
            flow.prepare("failed_proof", preflight_proofs=proof)

    def test_formal_ready_cannot_replace_fixed_required_steps_with_an_empty_list(self):
        proof = self.formal_proof()
        data = old.read_json(proof)
        data.update(required_steps=[], steps={})
        old.atomic_json(proof, data)
        with self.assertRaisesRegex(ValueError, "required_steps|前置"):
            flow.prepare("empty_proof", preflight_proofs=proof)

    def test_changed_proof_registry_is_rejected_without_rewriting_original_selection(self):
        root = self.prepare()
        original = (root/"selection.json").read_bytes()
        proof = self.formal_proof()
        data = old.read_json(proof)
        data["formal_ready"] = False
        old.atomic_json(proof, data)
        with self.assertRaisesRegex(ValueError, "SHA|证明"):
            flow.verify(root)
        self.assertEqual((root/"selection.json").read_bytes(), original)

    def test_bound_preflight_evidence_is_rechecked_before_any_stage_worker(self):
        root = self.prepare()
        proof = self.formal_proof()
        before = proof.read_bytes()
        evidence = self.base/"m6/current_fixture/evidence.json"
        evidence.write_text("changed required evidence", encoding="utf-8")
        with patch.object(old, "OwnedProcess") as processes:
            with self.assertRaisesRegex(ValueError, "证明|前置|preflight|SHA"):
                flow.run_stage(root, "planning200")
        processes.assert_not_called()
        self.assertEqual(proof.read_bytes(), before, "validation cannot repair the saved proof")

    def test_worker_tool_change_during_job_invalidates_otherwise_solved_result(self):
        root = self.prepare()
        manifest = self.canonical(root)
        job = manifest["jobs"][0]
        start = root/"fixture.start"
        start.write_text("start", encoding="utf-8")
        result = root/"fixture.result.json"
        spec = root/"fixture.request.json"
        old.atomic_json(spec, {"job": job, "start_signal": str(start), "result_path": str(result),
            "execution_manifest": manifest["manifest_path"], "hard_deadline_epoch": time.time()+60.})
        def changed_tool(*args):
            (self.project/"tools/motion_v135_report.py").write_text("# changed in worker", encoding="utf-8")
            return {"status": "solved", "passed": True}
        with patch("tools.v134_batch.plan_job", side_effect=changed_tool):
            code = flow.worker(spec)
        saved = old.read_json(result)
        self.assertEqual(code, 1)
        self.assertEqual(saved["status"], "invalid_version")
        self.assertFalse(saved["passed"])


class SingleJobRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT/"out")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.base = self.project/"motion_v135"
        self.task = self.project/"fixture.toml"
        self.task.write_text('name="fixture"\n', encoding="utf-8")
        self.started = []
        test = self
        class FakeSimulator:
            def __init__(self, root, *args, **kwargs):
                test.started.append((Path(root), args, kwargs))
        self.owner = FakeSimulator
        from simbridge import runner
        self.runner = runner
        for item in (patch.object(single, "BASE", self.base), patch.object(old, "ROOT", self.project),
                     patch.object(runner, "Simulator", FakeSimulator)):
            item.start()
            self.addCleanup(item.stop)

    def fake_job(self, job):
        self.runner.Simulator("ignored historical log root", port=23210)
        return {"status": "passed", "passed": True}

    def test_task_and_gate_route_owned_logs_to_their_new_output_namespace(self):
        for kind in ("task", "gate"):
            with self.subTest(kind=kind):
                output = self.base/kind
                options = {"task": self.task} if kind == "task" else {"gate": old.LIVE_TESTS[0]}
                method = "sim_job" if kind == "task" else "gate_job"
                with patch.object(old, method, side_effect=self.fake_job) as job:
                    answer = single.run(output=output, **options)
                self.assertTrue(answer["passed"])
                self.assertEqual(self.started[-1][0], output/"instance")
                self.assertEqual(self.started[-1][2]["port"], 23210)
                self.assertIs(self.runner.Simulator, self.owner)
                self.assertTrue((output/"worker_result.json").exists())
                self.assertEqual(Path(job.call_args.args[0]["output"]), output)

    def test_foreign_output_is_rejected_before_dispatch(self):
        with patch.object(old, "sim_job") as job:
            with self.assertRaisesRegex(ValueError, "命名空间"):
                single.run(output=self.project/"outside", task=self.task)
        job.assert_not_called()
        self.assertEqual(self.started, [])

    def test_dispatch_error_restores_original_simulator_factory(self):
        with patch.object(old, "sim_job", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                single.run(output=self.base/"failed", task=self.task)
        self.assertIs(self.runner.Simulator, self.owner)


if __name__ == "__main__":
    unittest.main()
