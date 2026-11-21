"""Report reconstruction with temporary manifests and fake durable evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import os

from tools import motion_v135_report as report


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False)+"\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class ReportFixture:
    def __init__(self, project):
        self.project = project
        self.base = project/"results/sim_v1_3/motion_v135"
        self.root = self.base/"experiments/formal_a1"
        self.previous = project/"results/sim_v1_3/v134"
        self.manifests, self.old_manifests = {}, {}
        self.selection = {"code_sha256": {}, "worker_code_sha256": {}, "manifests": {}}
        for name in ("skillnet/pipeline.py", "tools/motion_v135_report.py", "tests/test_motion_v135_report.py"):
            path = project/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen test fixture\n", encoding="utf-8")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            group = "code_sha256" if name.startswith("skillnet/") else "worker_code_sha256"
            self.selection[group][name] = sha
            snapshot = self.root/"code_snapshot"/name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
        for stage, (old_name, count) in report.STAGES.items():
            old_jobs, jobs = [], []
            # Native identity tests need the fixed20 cohort; the planning
            # certificate seam needs one case. Other missing source slots
            # deliberately exercise preservation of the formal denominators.
            fixture_count = 20 if stage == "fixed20" else 1 if stage == "planning200" else 0
            for index in range(fixture_count):
                phase = "plan" if stage == "planning200" else "gates" if stage == "gates" else "sim"
                family = "F"+str(index//40+1) if stage == "planning200" else "F"+str(index//4+1) if stage == "fixed20" else "F3" if stage == "walls3" else "fixture"
                task = project/"tasks"/stage/"cohort.toml"
                test_name = None
                if stage == "gates":
                    test_name = "tests.test_live_sim.LiveSimTests."+list(report.GATE_TASKS)[index]
                    task = project/"tasks"/(list(report.GATE_TASKS.values())[index]+".toml")
                task.parent.mkdir(parents=True, exist_ok=True)
                if not task.is_file():
                    task.write_text("name = '"+stage+"'\n", encoding="utf-8")
                task_sha = hashlib.sha256(task.read_bytes()).hexdigest()
                old_job = {"id": f"old-{stage}-{index}", "source_job_id": f"source-{stage}-{index}",
                    "phase": phase, "family": family, "cohort": "planning_200" if phase == "plan" else phase,
                    "selection": 1, "timeout_seconds": 1200.,
                    "output": str(self.previous/"experiments/c_wall_v134/runtime_fix2"/old_name/str(index))}
                if test_name:
                    old_job["test_name"] = test_name
                else:
                    old_job.update(task=str(task), task_sha256=task_sha, prepared_task_sha256=task_sha)
                job = {**deepcopy(old_job), "id": f"new-{stage}-{index}", "logical_job_id": f"new-{stage}-{index}",
                       "baseline_job_id": old_job["id"], "output": str(self.root/stage/str(index))}
                old_jobs.append(old_job)
                jobs.append(job)
            common = {"schema_version": 1, "phase_budgets_seconds": {"plan": None, "sim": 18000., "gates": None}}
            old_manifest = {**common, "jobs": old_jobs,
                            "state_dir": str(self.previous/"states/runtime_fix2"/old_name)}
            manifest = {**common, "jobs": jobs, "state_dir": str(self.root/"states"/stage)}
            old_path = self.previous/"manifests"/(old_name+".json")
            path = self.root/"manifests"/(stage+".json")
            write_json(old_path, old_manifest)
            write_json(path, manifest)
            self.old_manifests[stage], self.manifests[stage] = old_manifest, manifest
            self.selection["manifests"][stage] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "baseline_manifest": str(old_path), "baseline_manifest_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest()}
        self.save_selection()

    def save_selection(self):
        write_json(self.root/"selection.json", self.selection)

    def commit(self, stage, index, *, current=True, status="passed", native=True, job=None, **extra):
        manifest = self.manifests[stage] if current else self.old_manifests[stage]
        job = deepcopy(job or manifest["jobs"][index])
        task = Path(job["task"]) if job.get("task") else self.project/"tasks"/(report.GATE_TASKS[job["test_name"].rsplit(".", 1)[-1]]+".toml")
        task_sha = hashlib.sha256(task.read_bytes()).hexdigest()
        row = {"id": job["id"], "phase": job["phase"], "family": job["family"], "output": job["output"],
               "definition_sha256": digest(job), "task_sha256": task_sha,
               "status": status, "passed": status in {"passed", "solved"}, **extra}
        if current:
            row.update(logical_job_id=job.get("logical_job_id", job["id"]),
                       code_version_sha256=digest(self.selection["code_sha256"]))
        output = Path(job["output"])
        if stage == "planning200":
            row.update(makespan=90., bound=1.1, expanded=12, elapsed_seconds=1., first_solution_seconds=.1,
                       stages=[{"phase": "weighted", "non_admissible_ordering": False, "suboptimality_bound": 1.1}])
            write_json(output/"plan.json", row)
        elif native:
            directory = output/"simulation"/task.stem
            summary = {"task_file": str(task), "task_sha256": task_sha, "simulator_pid": 1234,
                "simulator_closed": True, "actual_makespan_seconds": 100., "status": status,
                "passed": status == "passed", "success": status == "passed",
                "wall_seconds": 12., "real_planning_status": "solved",
                "nominal_makespan_seconds": 95., "phase_seconds": {"real_planning": 2., "execute": 5.},
                "stage_seconds": {"tpg_structure": .1, "tpg_geometry": 1.},
                "tpg_seconds": 1.1, "tpg_nodes": 10, "tpg_edges": 20,
                "step_durations": [{"step": 1, "action": "pick", "estimated_seconds": 9.,
                                    "actual_seconds": 10., "estimated_actual_ratio": .9}]}
            row.update(summary, evidence_directory=str(directory))
            write_json(directory/"summary.json", summary)
            write_json(directory/"plan.json", {"status": "solved", "makespan": 90.})
            write_json(directory/"execution.json", {"success": True, "done": True,
                "independent_validation_passed": True, "total": 3, "completed": 3,
                "tracking_failures": [], "collision_failures": [], "task_schedule": [{"id": "n1"}]})
            write_json(directory/"validation.json", {"success": True, "violations": [],
                "expected_nodes": 3, "actual_node_intervals": 3})
            write_json(directory/"geometry/rollout_verification.json", {"valid": True,
                "coverage_complete": True, "no_sampled_collision": True,
                "stationary_waiting_poses_included": True, "planned_samples": 4, "tested_samples": 4})
            write_json(directory/"geometry/geometry_statistics.json", {"oracle_calls": 4, "cache_hits": 3})
            write_json(directory/"motion_backend.json", {"validations": [], "feedback_rounds": []})
            write_json(directory/"execution_validation.json", {"success": True})
            write_json(directory/"reference_receipt.json", {"tasks_sha256": "fixture-reference"})
            write_json(directory/"tasks.json", [{"id": "task1"}])
            write_json(directory/"tpg.json", {"nodes": [{"id": "n1"}]})
        journal = Path(manifest["state_dir"])/"completed.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row)+"\n")
        return row


class MotionReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)

    def fixture(self):
        return ReportFixture(self.project)

    def test_unprepared_attempt_preserves_all_denominators_without_creating_state(self):
        result = report.collect(project=self.project)
        self.assertFalse(result["complete"])
        self.assertEqual({name: row["pending"] for name, row in result["stages"].items()},
                         {"planning200": 200, "fixed20": 20, "walls3": 3, "gates": 5})
        self.assertFalse(any(self.project.iterdir()))
        self.assertTrue(all(item["status"] == "pending" for item in result["acceptance"].values()))

    def test_native_pass_requires_execution_and_independent_validation(self):
        fixture = self.fixture()
        row = fixture.commit("fixed20", 0)
        result = report.collect(project=self.project)
        self.assertEqual(result["stages"]["fixed20"]["passed"], 1)
        write_json(Path(row["evidence_directory"])/"validation.json", {"success": False})
        result = report.collect(project=self.project)
        self.assertEqual(result["stages"]["fixed20"]["passed"], 0)
        self.assertEqual(result["stages"]["fixed20"]["statuses"], {"passed": 1, "pending": 19})
        self.assertEqual(result["stages"]["fixed20"]["cases"][0]["current"]["failure_category"],
                         "evidence:uncredited_native_pass")

    def test_current_native_pass_requires_complete_counts_coverage_and_all_reference_files(self):
        cases = [("execution.json", "completed", 2),
                 ("execution.json", "tracking_failures", ["lost tracking"]),
                 ("validation.json", "actual_node_intervals", 2),
                 ("validation.json", "violations", ["missing interval"]),
                 ("geometry/rollout_verification.json", "coverage_complete", False),
                 ("geometry/rollout_verification.json", "tested_samples", 3),
                 ("reference_receipt.json", None, None),
                 ("tasks.json", None, None), ("tpg.json", None, None),
                 ("execution_validation.json", None, None)]
        for name, field, value in cases:
            with self.subTest(name=name, field=field), tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                fixture = ReportFixture(project)
                row = fixture.commit("fixed20", 0)
                path = Path(row["evidence_directory"])/name
                if field is None:
                    path.unlink()
                else:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    data[field] = value
                    write_json(path, data)
                result = report.collect(project=project)
                self.assertEqual(result["stages"]["fixed20"]["passed"], 0)
                self.assertEqual(result["stages"]["fixed20"]["cases"][0]["current"]["failure_category"],
                                 "evidence:uncredited_native_pass")

    def test_legacy_baseline_contract_does_not_require_new_reference_receipt(self):
        fixture = self.fixture()
        row = fixture.commit("fixed20", 0, current=False)
        (Path(row["evidence_directory"])/"reference_receipt.json").unlink()
        result = report.collect(project=self.project)
        before = result["stages"]["fixed20"]["cases"][0]["baseline"]
        self.assertTrue(before["real_passed"])
        self.assertEqual(before["native_evidence_contract"], "v134_archived_contract")

    def test_motion_cut_missing_global_bound_does_not_remove_a_native_pass(self):
        fixture = self.fixture()
        row = fixture.commit("fixed20", 0)
        directory = Path(row["evidence_directory"])
        summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        summary.update(symbolic_lower_bound=None, symbolic_suboptimality_bound=None)
        write_json(directory/"summary.json", summary)
        scope = {"scope": "original symbolic cost only", "cuts": 1, "temporary_cuts": 1,
                 "unrestricted_lower_bound": None, "suboptimality_bound": None,
                 "restricted_search": {"lower_bound": 90., "suboptimality_bound": 1.}}
        write_json(directory/"motion_backend.json", {"validations": [],
            "feedback_rounds": [{"motion": [], "certificate_scope": scope}]})
        result = report.collect(project=self.project)
        current = result["stages"]["fixed20"]["cases"][0]["current"]
        self.assertTrue(current["real_passed"])
        self.assertIsNone(current["symbolic_certificate"]["suboptimality_bound"])
        self.assertEqual(current["symbolic_certificate"]["feedback_scopes"], [scope])

    def test_timeout_uses_latest_motion_boundary_and_preserves_uncommitted_journal_tail(self):
        fixture = self.fixture()
        job = fixture.manifests["fixed20"]["jobs"][0]
        fixture.commit("fixed20", 0, status="timeout", native=False, failure_phase="task_wall_timeout")
        directory = Path(job["output"])/"simulation"/Path(job["task"]).stem
        old = directory/"run_progress.json"
        write_json(old, {"phase": "real_planning"})
        event = {"query": "validate", "phase": "before", "action": "insert(A, beam, target)"}
        pointer = directory/"motion_progress.json"
        write_json(pointer, {"phase": "real_planning/validate:before", "event": event})
        os.utime(old, (1., 1.)); os.utime(pointer, (2., 2.))
        journal = directory/"motion_live.jsonl"
        original = json.dumps({"sequence": 1, "event": event}).encode()+b'\n{"event":'
        journal.write_bytes(original)
        current = report.collect(project=self.project)["stages"]["fixed20"]["cases"][0]["current"]
        self.assertEqual(current["failure_category"], "timeout:real_planning/validate:before")
        self.assertEqual(current["last_motion_event"]["event"], event)
        self.assertEqual(journal.read_bytes(), original)
        self.assertFalse(current["real_passed"])

    def test_fsynced_journal_boundary_can_be_newer_than_the_atomic_progress_pointer(self):
        fixture = self.fixture()
        job = fixture.manifests["fixed20"]["jobs"][0]
        fixture.commit("fixed20", 0, status="timeout", native=False, failure_phase="task_wall_timeout")
        directory = Path(job["output"])/"simulation"/Path(job["task"]).stem
        write_json(directory/"motion_progress.json", {"phase": "real_planning/symbolic_search:after",
                   "elapsed_seconds": 10., "event": {"query": "symbolic_search", "phase": "after"}})
        event = {"query": "validate", "phase": "before", "action": "insert(A, beam, target)"}
        journal = directory/"motion_live.jsonl"
        journal.write_bytes(json.dumps({"sequence": 2, "elapsed_seconds": 12., "event": event}).encode()+b"\n")
        before = journal.read_bytes()
        current = report.collect(project=self.project)["stages"]["fixed20"]["cases"][0]["current"]
        self.assertEqual(current["failure_category"], "timeout:real_planning/validate:before")
        self.assertEqual(current["last_motion_event"]["event"], event)
        self.assertEqual(journal.read_bytes(), before)

    def test_old_motion_journal_seconds_do_not_override_a_newer_execution_checkpoint(self):
        fixture = self.fixture()
        job = fixture.manifests["fixed20"]["jobs"][0]
        fixture.commit("fixed20", 0, status="timeout", native=False, failure_phase="task_wall_timeout")
        directory = Path(job["output"])/"simulation"/Path(job["task"]).stem
        write_json(directory/"executor_progress.json", {"phase": "execute", "elapsed_seconds": 5.})
        (directory/"motion_live.jsonl").write_bytes(json.dumps({"sequence": 2, "elapsed_seconds": 100.,
            "event": {"query": "validate", "phase": "after"}}).encode()+b"\n")
        current = report.collect(project=self.project)["stages"]["fixed20"]["cases"][0]["current"]
        self.assertEqual(current["failure_category"], "timeout:execute")

    def test_original_eight_passes_are_kept_for_individual_regression_checks(self):
        fixture = self.fixture()
        for index in range(8):
            fixture.commit("fixed20", index, current=False)
        fixture.commit("fixed20", 0, status="failed")
        result = report.collect(project=self.project)
        stage = result["stages"]["fixed20"]
        self.assertEqual(stage["baseline_passed"], 8)
        self.assertEqual(stage["baseline_pass_regressions"], ["source-fixed20-0"])
        self.assertEqual(len(stage["baseline_pass_pending"]), 7)
        self.assertEqual(result["acceptance"]["original_eight_regressions"]["status"], "failed")

    def test_timeout_without_summary_keeps_sample_and_geometry_phase(self):
        fixture = self.fixture()
        job = fixture.manifests["fixed20"]["jobs"][0]
        fixture.commit("fixed20", 0, status="timeout", native=False, failure_phase="task_wall_timeout")
        write_json(Path(job["output"])/"simulation"/Path(job["task"]).stem/"executor_progress.json",
                   {"phase": "tpg_geometry", "stage_seconds": {"tpg_structure": 1., "tpg_geometry": 1190.}})
        result = report.collect(project=self.project)
        case = result["stages"]["fixed20"]["cases"][0]["current"]
        self.assertEqual(case["status"], "timeout")
        self.assertEqual(case["failure_category"], "timeout:tpg_geometry")
        self.assertEqual(result["stages"]["fixed20"]["pending"], 19)

    def test_interrupted_recovery_matches_original_identity_and_does_not_repeat_denominator(self):
        fixture = self.fixture()
        original = fixture.manifests["fixed20"]["jobs"][0]
        fixture.commit("fixed20", 0, status="interrupted", native=False)
        replacement = {**deepcopy(original), "id": original["id"]+"-resume1",
                       "output": str(fixture.root/"recovery_outputs/fixed20/000/001")}
        write_json(fixture.root/"recovery/fixed20"/(replacement["id"]+".json"),
                   {"logical_job_id": original["id"], "job": replacement})
        fixture.commit("fixed20", 0, job=replacement)
        result = report.collect(project=self.project)
        stage = result["stages"]["fixed20"]
        self.assertEqual(stage["passed"], 1)
        self.assertEqual(stage["interrupted_attempts"], 1)
        self.assertEqual(stage["denominator"], 20)
        self.assertEqual(stage["cases"][0]["source_job_id"], "source-fixed20-0")

    def test_duplicate_formal_commits_are_not_silently_selected(self):
        fixture = self.fixture()
        original = fixture.manifests["fixed20"]["jobs"][0]
        fixture.commit("fixed20", 0)
        replacement = {**deepcopy(original), "id": original["id"]+"-resume1"}
        fixture.commit("fixed20", 0, job=replacement)
        result = report.collect(project=self.project)
        self.assertEqual(result["stages"]["fixed20"]["passed"], 0)
        self.assertTrue(any(row["reason"] == "multiple_formal_commits" for row in result["issues"]))

    def test_uncommitted_tail_is_retained_byte_for_byte(self):
        fixture = self.fixture()
        path = Path(fixture.manifests["fixed20"]["state_dir"])/"completed.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        original = b'{"id":"new-fixed20-0","status":"passed"}'
        path.write_bytes(original)
        result = report.collect(project=self.project)
        self.assertEqual(result["stages"]["fixed20"]["pending"], 20)
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(any(row["reason"] == "uncommitted_tail_preserved" for row in result["issues"]))

    def test_wrong_original_task_sha_is_unmatched_and_cannot_receive_a_pass(self):
        fixture = self.fixture()
        fixture.manifests["fixed20"]["jobs"][0]["prepared_task_sha256"] = "0"*64
        write_json(fixture.root/"manifests/fixed20.json", fixture.manifests["fixed20"])
        fixture.commit("fixed20", 0)
        result = report.collect(project=self.project)
        self.assertEqual(result["stages"]["fixed20"]["passed"], 0)
        self.assertEqual(result["stages"]["fixed20"]["unmatched_current_jobs"], ["new-fixed20-0"])

    def test_versions_check_current_files_snapshots_and_registered_report_tools(self):
        fixture = self.fixture()
        self.assertTrue(report.collect(project=self.project)["versions"]["matched"])
        (self.project/"skillnet/pipeline.py").write_text("# changed\n", encoding="utf-8")
        result = report.collect(project=self.project)
        self.assertFalse(result["versions"]["matched"])
        self.assertIn("skillnet/pipeline.py", result["versions"]["current_changed_files"])

    def test_formal_worker_tool_sha_mismatch_cannot_receive_native_credit(self):
        fixture = self.fixture()
        manifest = fixture.manifests["fixed20"]
        manifest["jobs"][0]["worker_code_version_sha256"] = digest(fixture.selection["worker_code_sha256"])
        path = fixture.root/"manifests/fixed20.json"
        write_json(path, manifest)
        fixture.selection["manifests"]["fixed20"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        fixture.save_selection()
        fixture.commit("fixed20", 0, worker_code_version_sha256="0"*64)
        result = report.collect(project=self.project)
        self.assertEqual(result["stages"]["fixed20"]["passed"], 0)
        self.assertIn("formal_worker_version_mismatch_or_missing",
                      result["stages"]["fixed20"]["cases"][0]["current"]["evidence_issues"])

    def test_planning_cost_bound_and_admissibility_scope_remain_explicit(self):
        fixture = self.fixture()
        fixture.commit("planning200", 0, current=False, status="solved")
        fixture.commit("planning200", 0, status="solved")
        result = report.collect(project=self.project)
        case = result["stages"]["planning200"]["cases"][0]
        self.assertTrue(case["current"]["certificate_bound_present"])
        self.assertEqual(case["cost_delta_seconds"], 0.)
        self.assertIn("excludes motion", case["current"]["certificate_scope"])
        self.assertEqual(result["stages"]["planning200"]["families"]["F1"]["solved"], 1)
        self.assertEqual(result["acceptance"]["planning_regression_v134_ruling"]["status"], "pending")

    def test_wrongly_versioned_full_suite_proof_remains_pending(self):
        fixture = self.fixture()
        proof = fixture.root/"validation/full.json"
        write_json(proof, {"passed": True, "code_sha256": {"skillnet/pipeline.py": "0"*64}})
        write_json(fixture.root/"acceptance_evidence.json", {"full_suite": str(proof)})
        result = report.collect(project=self.project)
        self.assertEqual(result["proofs"]["full_suite"]["status"], "pending")
        self.assertFalse(result["proofs"]["full_suite"]["code_match"])

    def test_default_rebuild_writes_only_reports_and_publish_is_explicit(self):
        fixture = self.fixture()
        overview = fixture.base/"summary.md"
        overview.write_text("current overview", encoding="utf-8")
        previous_summary = fixture.previous/"summary.md"
        previous_summary.write_text("historic summary", encoding="utf-8")
        before = {path: path.read_bytes() for path in self.project.rglob("*") if path.is_file()}
        report.rebuild(project=self.project)
        self.assertEqual(overview.read_text(encoding="utf-8"), "current overview")
        self.assertEqual(previous_summary.read_text(encoding="utf-8"), "historic summary")
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        added = {path for path in self.project.rglob("*") if path.is_file()}-before.keys()
        self.assertTrue(added)
        self.assertTrue(all(path.is_relative_to(fixture.root/"reports") for path in added))
        report.rebuild(project=self.project, publish=True)
        self.assertIn("交付报告", overview.read_text(encoding="utf-8"))
        self.assertEqual(previous_summary.read_text(encoding="utf-8"), "historic summary")

    def test_attempt_path_traversal_is_rejected(self):
        for attempt in ("../escape", "a/b", "..", "a\\b"):
            with self.subTest(attempt=attempt), self.assertRaises(ValueError):
                report.collect(attempt, project=self.project)


if __name__ == "__main__":
    unittest.main()
