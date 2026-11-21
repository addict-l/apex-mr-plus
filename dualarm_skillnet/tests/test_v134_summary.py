"""Pure files/JSON report tests: never start search or a simulator."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tools import v134_summary as report
from tools import overnight_batch as old


class V134SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=report.ROOT / "out", prefix="v134_report_mock_")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.base = self.project / "results/sim_v1_3/v134"
        self.previous = self.project / "results/sim_v1_3/v133"
        self.old_jobs = []
        self.new_jobs = []
        for i in range(20):
            task = self.project / "tasks" / f"task_{i}.toml"
            task.parent.mkdir(parents=True, exist_ok=True)
            task.write_text(f'name="task_{i}"\n', encoding="utf-8")
            self.old_jobs.append({"id": f"old-{i}", "phase": "sim", "kind": "sim",
                "task": str(task), "family": f"F{i//4+1}", "output": str(self.previous / "sim20" / str(i))})
        self.old_manifest = self.manifest(self.previous, "sim20", self.old_jobs)
        for i, job in enumerate(self.old_jobs):
            self.new_jobs.append({**job, "id": f"new-{i}", "output": str(self.base / "sim20" / str(i)),
                "source_job_id": job["id"], "source_task_sha256": old.sha256(job["task"]),
                "source_manifest_sha256": old.sha256(self.old_manifest), "fixed_sample_index": i})
        self.manifest(self.base, "sim20", self.new_jobs)

    def write(self, path, data):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def manifest(self, base, stage, jobs):
        return self.write(base / "manifests" / (stage+".json"), {"session_id": "mock_"+stage,
            "state_dir": str(base / "states" / stage), "report_dir": str(base / "reports"), "jobs": jobs})

    def commit(self, base, stage, index, *, passed=False, backend=None, reason="unknown", feedback=None):
        manifest = old.load_manifest(base / "manifests" / (stage+".json"))
        job = manifest["jobs"][index]
        native = Path(job["output"]) / "simulation" / ("task_"+str(index))
        summary = {"task_file": job["task"], "task_sha256": old.sha256(job["task"]),
            "status": "passed" if passed else "failed", "passed": passed,
            "simulator_pid": 101, "simulator_closed": True,
            "actual_makespan_seconds": 10. if passed else None,
            "real_planning_status": "solved" if passed else "unknown",
            "failure_phase": None if passed else "real_planning", "reason": None if passed else reason,
            "motion_feedback_count": 999, "qualification_calls": 888, "feedback": feedback or []}
        self.write(native / "summary.json", summary)
        self.write(native / "plan.json", {"status": "solved" if passed else "unknown", "makespan": 11.,
            "feedback": feedback or [], "motion": [] if passed else [{"status": "UNKNOWN"}],
            "search": {"rejected": {"cut": 0}}})
        self.write(native / "motion_backend.json", backend or {})
        row = {"id": job["id"], "definition_sha256": job["definition_sha256"],
            "evidence_directory": str(native), **summary}
        journal = Path(manifest["state_dir"]) / "completed.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        return native

    def test_same_twenty_match_by_source_id_and_sha_and_keep_pending_denominator(self):
        self.commit(self.previous, "sim20", 0, backend={"validations": [{"status": "UNKNOWN",
            "failure_evidence": [{"ok": False, "problem": {"reason": "table_collision:A", "pair": ["link", "table"]}}]}]},
            feedback=["运动验证未知：pick；原因：8个种子耗尽。"])
        self.commit(self.base, "sim20", 0, passed=True)
        before = {p: p.read_bytes() for p in self.previous.rglob("*") if p.is_file()}
        data = report.rebuild(base=self.base, project=self.project)
        comparison = data["sim20_comparison"]
        self.assertEqual(comparison["denominator"], 20)
        self.assertEqual(len(comparison["cases"]), 20)
        self.assertEqual(comparison["v133"]["real_passed"], 0)
        self.assertEqual(comparison["v134"]["real_passed"], 1)
        self.assertEqual(comparison["v134"]["pending"], 19)
        self.assertTrue(comparison["cases"][0]["source_chain_valid"])
        self.assertEqual(comparison["cases"][0]["v133"]["failure_category"], "real_planning:ik_exhausted")
        self.assertTrue(comparison["cases"][0]["v133"]["feedback_metrics"]["unknown_direct_terminal"])
        self.assertEqual(data["sim20_final_plan_to_actual"]["n"], 1)
        self.assertAlmostEqual(data["sim20_final_plan_to_actual"]["median"], 1.1)
        self.assertEqual(before, {p: p.read_bytes() for p in self.previous.rglob("*") if p.is_file()})
        self.assertTrue((self.base / "sim20_comparison.csv").is_file())

    def test_structured_cut_and_actual_escalation_replace_direct_unknown_without_invented_counts(self):
        motion = {"action": "pick(A,b,l)", "state": ["free(A)"], "initial_status": "UNKNOWN",
            "initial_context": {"ik_seed_limit": 8, "classification": "sampling_unknown"},
            "escalated": True, "escalation_context": {"ik_seed_limit": 24, "classification": "sampling_unknown"},
            "status": "UNKNOWN", "cut": True, "action_cut_count": 3, "downgraded_capabilities": ["reach(A,l)"]}
        backend = {"feedback_rounds": [{"round": 1, "search_status": "solved", "motion": [motion]},
            {"round": 2, "search_status": "no_plan", "motion": [], "terminal_status": "unknown"}],
            "validations": [{"status": "UNKNOWN", "failure_evidence": [{"ok": False}]}]}
        self.commit(self.base, "sim20", 0, backend=backend, reason={"status": "unknown", "detail": "finite attempts"},
                    feedback=["运动验证未知；已规避该状态下的此动作并重新搜索。"])
        data = report.rebuild(base=self.base, project=self.project)
        stats = data["sim20_comparison"]["cases"][0]["v134"]["feedback_metrics"]
        self.assertEqual(stats["cut_events"], 1)
        self.assertEqual(stats["actual_8_to_24_escalations"], 1)
        self.assertEqual(stats["structured_rounds"], 2)
        self.assertEqual(stats["downgraded_capabilities"], ["reach(A,l)"])
        self.assertTrue(stats["unknown_after_cut"])
        self.assertFalse(stats["unknown_direct_terminal"])
        self.assertEqual(stats["cut_reasons"], {"sampling_unknown": 1})

    def test_source_sha_mismatch_cannot_credit_success_or_ratio(self):
        self.new_jobs[0]["source_task_sha256"] = "incorrect"
        self.manifest(self.base, "sim20", self.new_jobs)
        self.commit(self.base, "sim20", 0, passed=True)
        data = report.rebuild(base=self.base, project=self.project)
        comparison = data["sim20_comparison"]
        self.assertFalse(comparison["cases"][0]["source_chain_valid"])
        self.assertEqual(comparison["v134"]["real_passed"], 0)
        self.assertEqual(data["sim20_final_plan_to_actual"]["n"], 0)
        self.assertEqual(data["native_metrics"]["sim20"]["real_passed"], 0)

    def test_native_failure_cannot_be_hidden_by_passed_journal(self):
        native = self.commit(self.base, "sim20", 0, passed=True)
        summary = json.loads((native / "summary.json").read_text(encoding="utf-8"))
        summary.update(passed=False, status="failed", failure_phase="acceptance", reason="Servo tracking limit on B")
        self.write(native / "summary.json", summary)
        data = report.rebuild(base=self.base, project=self.project)
        self.assertEqual(data["native_metrics"]["sim20"]["real_passed"], 0)
        self.assertEqual(data["sim20_final_plan_to_actual"]["n"], 0)
        self.assertEqual(data["sim20_comparison"]["cases"][0]["v134"]["failure_category"], "execution:tracking_failure")

    def test_prequalification_and_already_escalated_request_are_not_new_feedback(self):
        backend = {"feedback_rounds": [{"round": 1, "search_status": "solved", "motion": [{
            "action": "pick", "initial_context": {"ik_seed_limit": 24}, "escalated": True,
            "escalation_context": {"ik_seed_limit": 24, "escalation_skipped": "already_at_least_24"}}],
            "qualifications": [{"fact": "reach(A,l)", "status": "VALID"}], "terminal_status": "solved"}]}
        self.commit(self.base, "sim20", 0, passed=True, backend=backend)
        data = report.rebuild(base=self.base, project=self.project)
        stats = data["sim20_comparison"]["cases"][0]["v134"]["feedback_metrics"]
        self.assertEqual(stats["cut_events"], 0)
        self.assertEqual(stats["actual_8_to_24_escalations"], 0)
        self.assertEqual(stats["escalation_requests"], 1)

    def test_legacy_rrt_and_tracking_are_separate_from_ik_candidate_collision(self):
        self.commit(self.previous, "sim20", 0, backend={"validations": [{"status": "UNKNOWN", "failure_evidence": {
            "attempts": [{"problem": {"pair": ["link", "table"]}}]}}]})
        native = self.commit(self.previous, "sim20", 1, reason="Servo tracking limit on B")
        summary = json.loads((native / "summary.json").read_text(encoding="utf-8"))
        summary["failure_phase"] = "acceptance";self.write(native / "summary.json", summary)
        data = report.rebuild(base=self.base, project=self.project)
        categories = data["sim20_comparison"]["v133"]["failure_categories"]
        self.assertEqual(categories, {"real_planning:rrt_unknown": 1, "execution:tracking_failure": 1})

    def test_changed_actual_task_cannot_borrow_old_source_hash(self):
        unrelated = self.project / "tasks/unrelated.toml"
        unrelated.write_text('name="unrelated"\n', encoding="utf-8")
        self.new_jobs[0]["task"] = str(unrelated)
        self.manifest(self.base, "sim20", self.new_jobs)
        self.commit(self.base, "sim20", 0, passed=True)
        data = report.rebuild(base=self.base, project=self.project)
        case = data["sim20_comparison"]["cases"][0]
        self.assertFalse(case["source_chain_valid"])
        self.assertEqual(data["sim20_comparison"]["v134"]["real_passed"], 0)

    def test_walls_are_separate_and_missing_native_is_never_a_pass(self):
        jobs = [{**self.new_jobs[i], "id": f"wall-{i}", "output": str(self.base / "walls3" / str(i))} for i in range(3)]
        self.manifest(self.base, "walls3", jobs)
        self.commit(self.base, "walls3", 0, passed=True)
        native = self.commit(self.base, "sim20", 0, passed=True)
        (native / "summary.json").unlink()
        data = report.rebuild(base=self.base, project=self.project)
        self.assertEqual(data["native_metrics"]["walls3"]["real_passed"], 1)
        self.assertEqual(data["sim20_comparison"]["v134"]["real_passed"], 0)
        self.assertEqual(data["sim20_comparison"]["denominator"], 20)
        self.assertEqual(data["sim20_final_plan_to_actual"]["n"], 0)

    def test_duplicate_source_ids_do_not_replace_missing_sample(self):
        self.new_jobs[1]["source_job_id"] = self.new_jobs[0]["source_job_id"]
        self.manifest(self.base, "sim20", self.new_jobs)
        self.commit(self.base, "sim20", 0, passed=True)
        data = report.rebuild(base=self.base, project=self.project)
        cases = data["sim20_comparison"]["cases"]
        self.assertIn("duplicate_source_job_id", cases[0]["source_chain_issues"])
        self.assertIn("v134_job_missing", cases[1]["source_chain_issues"])
        self.assertEqual(data["sim20_comparison"]["v134"]["real_passed"], 0)

    def test_legacy_twenty_keep_thirteen_ik_one_rrt_one_servo_and_fourteen_direct_unknowns(self):
        for i in range(20):
            evidence = ({"attempts": [{"problem": {"pair": ["link", "table"]}}]}
                        if i == 18 else [{"ok": False, "problem": {"pair": ["link", "table"]}}])
            native = self.commit(self.previous, "sim20", i, passed=i < 5,
                backend={"validations": [{"status": "UNKNOWN", "failure_evidence": evidence}]} if i >= 5 else {},
                feedback=["运动验证未知：有限预算失败。"] if 5 <= i <= 18 else [])
            if i == 19:
                summary = json.loads((native / "summary.json").read_text(encoding="utf-8"))
                summary.update(real_planning_status="solved", failure_phase="acceptance", reason="Servo tracking limit on B")
                self.write(native / "summary.json", summary)
        data = report.rebuild(base=self.base, project=self.project)
        previous = data["sim20_comparison"]["v133"]
        self.assertEqual(previous["real_passed"], 5)
        self.assertEqual(previous["failure_categories"], {"real_planning:ik_exhausted": 13,
            "real_planning:rrt_unknown": 1, "execution:tracking_failure": 1})
        self.assertEqual(previous["unknown_direct_terminal"], 14)
        self.assertEqual(previous["cut_events"], 0)

    def test_planning_uses_selected_old_trial_and_preserves_fields_and_family_comparison(self):
        self.write(self.previous / "experiment_selection.json", {"stages": {"planning200": "planning200_retry1"}})
        for base, stage, status, bound in [(self.previous, "planning200_retry1", "budget", None),
                                          (self.base, "planning200", "solved", 1.2)]:
            job = {**self.old_jobs[0], "id": "p-"+stage, "phase": "plan", "kind": "plan", "table": "qualified",
                   "index": 0, "output": str(base / stage / "0")}
            path = self.manifest(base, stage, [job])
            loaded = old.load_manifest(path)["jobs"][0]
            journal = base / "states" / stage / "completed.jsonl"
            journal.parent.mkdir(parents=True, exist_ok=True)
            journal.write_text(json.dumps({"id": loaded["id"], "definition_sha256": loaded["definition_sha256"],
                "status": status, "bound": bound, "expanded": 12, "elapsed_seconds": 1.,
                "first_solution_seconds": .2 if status == "solved" else None})+"\n", encoding="utf-8")
        data = report.rebuild(base=self.base, project=self.project)
        family = data["planning_families"]["F1"]
        self.assertEqual((family["v133_committed"], family["v133_solved"], family["solved"]), (1, 0, 1))
        rows = json.loads((self.base / "planning_200.json").read_text(encoding="utf-8"))
        self.assertEqual(rows[0]["v133_status"], "budget")
        self.assertEqual(rows[0]["bound"], 1.2)
        self.assertEqual(rows[0]["first_solution_seconds"], .2)


if __name__ == "__main__":
    unittest.main()
