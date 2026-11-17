"""Synthetic evidence only: no planner, process, simulator, or experiment."""
import json
from pathlib import Path
import tempfile
import unittest

from tools import v133_summary as report


class V133SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v133_summary_mock_", dir=report.ROOT / "out")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.base = self.project / "results/sim_v1_3/v133"

    def save(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def stage(self, name, jobs, records):
        self.save(self.base / "manifests" / (name + ".json"), {"jobs": jobs})
        path = self.base / "states" / name / "completed.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return path

    def collect(self):
        return report.collect(self.base, project=self.project)

    def test_rebuild_includes_separate_section_four_scale_evidence(self):
        metrics = report.rebuild(self.base, project=self.project)
        self.assertEqual(metrics['scale_acceptance']['status'], 'pending')
        self.assertFalse(metrics['scale_acceptance']['passed'])
        self.assertTrue((self.base/'scale_acceptance.json').is_file())
        self.assertTrue((self.base/'scale_acceptance.md').is_file())
        self.assertIn('第4节规模', (self.base/'summary.md').read_text(encoding='utf-8'))

    def real(self, **extra):
        return {"passed": True, "status": "passed", "profile": "skillnet_v1_3",
                "simulator_pid": 123, "simulator_closed": True,
                "actual_makespan_seconds": 10., **extra}

    def native(self, name, *, family=None, **extra):
        output = self.base / "sim20" / name
        task = self.project / (name + ".toml")
        task.write_text('name="' + name + '"\narms=["A","B"]\n', encoding="utf-8")
        job = {"id": name, "phase": "sim", "kind": "sim", "task": str(task), "output": str(output)}
        if family:
            job["family"] = family
        summary = self.real(task=name, task_file=str(task), simulator_log=str(output / "native.log"), **extra)
        directory = output / "simulation" / name
        self.save(directory / "summary.json", summary)
        self.save(directory / "initial_plan.json", {"status": "solved", "makespan": 2.})
        self.save(directory / "plan.json", {"status": "solved", "makespan": 9.})
        return job, {"id": name, "phase": "complete", **summary}

    def fixture_attempt(self, stage, name, *, passed=True, task_name=None):
        task = self.project / "tasks/sim" / (name + ".toml")
        task.parent.mkdir(parents=True, exist_ok=True)
        task.write_text(f'name="{name}"\narms=["A","B"]\n[[location]]\nid="stock"\n', encoding="utf-8")
        output = self.base / stage / name
        directory = output / "simulation" / name
        job = {"id": stage + "-" + name, "phase": "sim", "kind": "sim", "task": str(task), "output": str(output)}
        summary = self.real(task=task_name or name, task_file=str(task), passed=passed,
            status="passed" if passed else "failed", simulator_log=str(output / "native.log"))
        self.save(directory / "summary.json", summary)
        self.save(directory / "prequalification.json", {"total": 2, "checked": 2,
            "records": [{"arm": arm, "place": "stock", "status": "VALID"} for arm in ("A", "B")]})
        return job, {"id": job["id"], **summary, "evidence_directory": str(directory)}, directory / "summary.json"

    def test_original_d_success_does_not_cover_failed_required_platform_specialty(self):
        pairs = [self.fixture_attempt("d_e2e_attempt3", name) for name in report.FIXTURES[1:]]
        self.stage("d_e2e_attempt3", [p[0] for p in pairs], [p[1] for p in pairs])
        name, stage = "s3a_platform_required_sim", "d_platform_required_attempt2"
        job, row, path = self.fixture_attempt(stage, name, passed=False)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary.pop("actual_makespan_seconds")
        summary.update(failure_phase="real_planning", real_planning_status="unknown",
            reason="有限IK种子耗尽", feedback=["detach(A,temp_block2,tp2) r2 table_collision:A"])
        self.save(path, summary)
        row.update(summary);row.pop("actual_makespan_seconds")
        self.stage(stage, [job], [row])
        self.save(self.base/"physical_stage_outcome.json", {
            "finished": True, "required_platform": {"stage": stage, "passed": True}})
        metrics = self.collect()
        self.assertTrue(metrics["physical"]["d_e2e_passed"])
        extra = metrics["physical"]["required_platform"]
        self.assertFalse(metrics["physical"]["temporary_platform_coverage_passed"])
        self.assertEqual(extra["status"], "not_met")
        self.assertTrue(extra["attempted"])
        self.assertEqual(extra["phase"], "real_planning")
        self.assertEqual(extra["real_planning_status"], "unknown")
        self.assertFalse(extra["tpg_entered"])
        self.assertFalse(extra["dynamics_entered"])
        rendered = report.markdown(metrics)
        self.assertIn("临时台搭拆专项", rendered)
        self.assertIn("table_collision:A", rendered)
        self.assertIn("不能据此证明数学不可达", rendered)

    def test_required_platform_only_uses_the_explicit_committed_stage_and_native_source(self):
        name = "s3a_platform_required_sim"
        job, row, _ = self.fixture_attempt("old_required_success", name)
        self.stage("old_required_success", [job], [row])
        self.save(self.base/"physical_stage_outcome.json", {
            "required_platform": {"stage": "missing_required_stage"}})
        extra = self.collect()["physical"]["required_platform"]
        self.assertEqual(extra["status"], "pending")
        self.assertFalse(extra["passed"])
        job, row, path = self.fixture_attempt("missing_required_stage", name)
        self.stage("missing_required_stage", [job], [])
        self.assertEqual(self.collect()["physical"]["required_platform"]["status"], "pending")
        self.stage("missing_required_stage", [job], [row]);path.unlink()
        self.assertEqual(self.collect()["physical"]["required_platform"]["status"], "pending")

    def test_required_platform_coverage_is_separate_in_milestones_and_preserves_scale(self):
        stage, name = "chosen_required_attempt", "s3a_platform_required_sim"
        job, row, path = self.fixture_attempt(stage, name)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary.update(phase="complete", tpg_seconds=2., tpg_nodes=12, execution_wall_seconds=4.)
        self.save(path, summary);row.update(summary)
        self.stage(stage, [job], [row])
        self.save(self.base/"physical_stage_outcome.json", {"required_platform": {"stage": stage}})
        self.save(self.base.parent/"milestones.json", {"M1": {"preserved": True}})
        metrics = report.rebuild(self.base, project=self.project)
        extra = metrics["physical"]["required_platform"]
        self.assertTrue(extra["passed"])
        self.assertTrue(extra["tpg_entered"])
        self.assertTrue(extra["dynamics_entered"])
        milestones = json.loads((self.base.parent/"milestones.json").read_text(encoding="utf-8"))
        self.assertEqual(milestones["M1"], {"preserved": True})
        self.assertTrue(milestones["v1.3.3"]["temporary_platform_coverage"]["passed"])
        self.assertIn("scale_status", milestones["v1.3.3"])
        self.assertIn("scale_acceptance", metrics)
        self.assertEqual(len(metrics["acceptance"]), 6)

    def test_diagnostics_keep_structured_unknowns_out_of_cuts_and_keep_twenty_denominator(self):
        job, row = self.native("unknown_native", family="F1", passed=False, status="failed",
            failure_phase="real_planning", qualification_calls=100,
            reason=[{"code": "IK_seed_exhaustion", "message": "x|y\nz，未知"}],
            feedback=["运动验证未知：pick(A,red,stock)，原因：8个IK种子耗尽。", {"unparsed": "保留结构"}])
        self.stage("sim20", [job], [row])
        metrics = self.collect()
        diag = metrics["diagnostics"]
        self.assertEqual(diag["simulation"]["denominator"], 20)
        self.assertEqual(diag["simulation"]["attempted_failures"], 1)
        self.assertAlmostEqual(diag["simulation"]["failure_rate"], 1/20)
        feedback = diag["feedback"]["sim20"]
        self.assertEqual(feedback["invalid_cut_messages"], 0)
        self.assertEqual(feedback["qualification_update_messages"], 0)
        self.assertEqual(feedback["motion_unknown_messages"], 1)
        self.assertEqual(feedback["qualification_calls_observed"], 100)
        self.assertEqual(feedback["other_messages"], 1)
        self.assertFalse(any("unreachable" in k for k in diag["simulation"]["category_counts"]))
        rendered = report.markdown(metrics)
        self.assertIn("失败分类", rendered)
        self.assertIn("x\\|y", rendered)

    def test_explicit_feedback_messages_phase_totals_and_full_ratio_distribution(self):
        job, row = self.native("with_feedback", family="F1", qualification_calls=41,
            phase_seconds={"compile": 2., "real_planning": 1.}, feedback=[
                "运动失败：place(A,red,goal)，原因：碰撞。已禁止该状态下的此动作并重新搜索。",
                "运动后端按需验证：reach(A, stock) → VALID。"])
        job2, row2 = self.native("without_feedback", family="F2", actual_makespan_seconds=20.,
            phase_seconds={"compile": 3.}, feedback=[])
        self.stage("sim20", [job, job2], [row, row2])
        metrics = self.collect()
        feedback = metrics["diagnostics"]["feedback"]["sim20"]
        self.assertEqual(feedback["invalid_cut_messages"], 1)
        self.assertEqual(feedback["qualification_update_messages"], 1)
        self.assertEqual(feedback["explicit_loop_messages"], 2)
        self.assertEqual(feedback["qualification_calls_observed"], 41)
        self.assertEqual(metrics["diagnostics"]["phase_seconds"]["sim20"]["compile"], 5.)
        ratio = metrics["ratios"]["sim20_final_estimate_to_actual"]
        self.assertEqual(ratio["n"], 2)
        self.assertAlmostEqual(ratio["min"], .45)
        self.assertAlmostEqual(ratio["max"], .9)
        rendered = report.markdown(metrics)
        self.assertIn("p25", rendered)
        self.assertIn("p75", rendered)
        self.assertIn("全量预资格", rendered)

    def test_paired_planning_unsolved_rows_keep_family_status_and_evidence_links(self):
        jobs, rows = [], []
        for table, status in (("raw", "unknown"), ("qualified", "budget")):
            output = self.base/"planning/F1/000"/table
            job = {"id": table+"-f1", "phase": "plan", "kind": "plan", "family": "F1", "index": 0,
                   "table": table, "output": str(output)}
            jobs.append(job);rows.append({"id": job["id"], "status": status, "passed": False,
                "elapsed_seconds": 1., "reason": [{"status": status}], "unknown": ["reach(A,goal)"]})
            self.save(output/"plan.json", {"status": status})
        self.stage("planning200", jobs, rows)
        metrics = self.collect()
        diag = metrics["diagnostics"]["planning"]
        self.assertEqual(diag["distance_prior"]["families"]["F1"]["status_counts"], {"unknown": 1})
        self.assertEqual(diag["ik"]["families"]["F1"]["status_counts"], {"budget": 1})
        self.assertTrue(diag["ik"]["families"]["F1"]["unsolved_cases"][0]["plan_present"])
        rendered = report.markdown(metrics)
        self.assertIn("逐族未解", rendered)
        self.assertIn("[plan.json](", rendered)

    def test_optional_final_full_failure_cannot_be_hidden_by_abc_and_e_successes(self):
        full = {"status": "passed", "tests_run": 3, "failure_count": 0, "error_count": 0,
            "per_test": [{"id": module+".Mock.test_case", "status": "passed"} for module in report.V11_MODULES]}
        for label in ("abc", "e"):
            self.save(self.project/"out"/f"v133_{label}_full.json", full)
            self.save(self.base/f"original11_{label}.json", {"passed": True, "rows": [{"passed": True}]*11})
        self.assertTrue(self.collect()["regressions"]["passed"])
        self.save(self.project/"out/v133_final_full.json", {**full, "status": "failed", "failure_count": 1})
        reg = self.collect()["regressions"]
        self.assertFalse(reg["passed"])
        self.assertEqual([row["label"] for row in reg["full"]], ["abc", "e", "final"])
        self.assertEqual([row["label"] for row in reg["original11"]], ["abc", "e"])

    def test_selected_planning_missing_or_unknown_never_adopts_old_trial_success(self):
        job = {"id": "same-id", "phase": "plan", "kind": "plan", "family": "F1", "index": 0,
               "table": "qualified", "output": str(self.base/"old_planning")}
        self.stage("planning200", [job], [{"id": "same-id", "status": "solved", "passed": True,
                                          "bound": 1., "elapsed_seconds": 1.}])
        self.save(self.base/"experiment_selection.json", {"selection_id": "c_scale_retry1",
            "stages": {"planning200": "planning200_retry1"}, "diagnostic_trials": ["planning200"]})
        metrics = self.collect()
        self.assertEqual(metrics["planning"]["ik"]["families"]["F1"]["solved"], 0)
        self.assertEqual(metrics["stages"]["planning200"]["physical_stage"], "planning200_retry1")
        self.assertEqual(metrics["diagnostic_trials"][0]["status_counts"], {"solved": 1})
        self.assertFalse(metrics["diagnostic_trials"][0]["formal"])
        self.assertEqual(metrics["scale_acceptance"]["status"], "pending")
        self.assertEqual(metrics["scale_acceptance"]["selected_stage"], "planning200_retry1")
        self.stage("planning200_retry1", [job], [{"id": "same-id", "status": "unknown", "passed": False,
                                                "elapsed_seconds": .01}])
        metrics = self.collect()
        case = metrics["planning"]["ik"]["cases"][0]
        self.assertEqual(case["stage"], "planning200_retry1")
        self.assertIn("states/planning200_retry1/", case["source"])
        self.assertEqual(case["status"], "unknown")
        self.assertEqual(metrics["planning"]["ik"]["families"]["F1"]["solved"], 0)
        self.assertIn("旧诊断批次", report.markdown(metrics))

    def test_selected_core_regressions_require_both_new_files_and_matching_fingerprint(self):
        full = {"status": "passed", "tests_run": 3, "failure_count": 0, "error_count": 0,
            "per_test": [{"id": module+".Mock.test_case", "status": "passed"} for module in report.V11_MODULES]}
        original = {"passed": True, "rows": [{"passed": True}]*11}
        for label in ("abc", "e"):
            self.save(self.project/"out"/f"v133_{label}_full.json", full)
            self.save(self.base/f"original11_{label}.json", original)
        fingerprint = {"skillnet/planning/search.py": "a"*64}
        self.save(self.base/"experiment_selection.json", {"selection_id": "c_scale_retry1",
            "stages": {"planning200": "planning200_retry1"}, "diagnostic_trials": ["planning200"],
            "regressions": {"full": "out/v133_c_scale_retry1_full.json",
                "original11": "results/sim_v1_3/v133/original11_c_scale_retry1.json"},
            "planning_code_sha256": fingerprint})
        self.assertFalse(self.collect()["regressions"]["passed"])
        self.save(self.project/"out/v133_c_scale_retry1_full.json", {**full, "planning_code_sha256": fingerprint})
        self.save(self.base/"original11_c_scale_retry1.json", {**original, "planning_code_sha256": {"wrong": "old"}})
        self.assertFalse(self.collect()["regressions"]["passed"])
        self.save(self.base/"original11_c_scale_retry1.json", {**original, "planning_code_sha256": fingerprint})
        reg = self.collect()["regressions"]
        self.assertTrue(reg["passed"])
        self.assertTrue(reg["full"][-1]["fingerprint_passed"])
        self.assertTrue(reg["original11"][-1]["fingerprint_passed"])

    def test_search_stage_traces_are_copied_without_promoting_probe_certificates(self):
        traces = [{"stage": "reference", "weight": 3., "action_scope": "goal_domain",
            "allowed_actions": 8, "incumbent_only": True, "time_budget": 8., "expansion_budget": 20000,
            "elapsed": 1., "expanded": 20, "generated": 30, "lower_bound": None, "suboptimality_bound": None},
            {"stage": "weighted", "weight": 3., "action_scope": "all_actions", "allowed_actions": None,
             "incumbent_only": False, "lower_bound": 10., "suboptimality_bound": 1.8}]
        job = {"id": "retry-q", "phase": "plan", "kind": "plan", "family": "F1", "index": 0,
               "table": "qualified"}
        row = {"id": "retry-q", "status": "solved", "passed": True, "bound": 1.8,
               "elapsed_seconds": 1., "stages": traces, "search_stages": traces}
        self.stage("planning200_retry1", [job], [row])
        self.save(self.base/"experiment_selection.json", {"selection_id": "c_scale_retry1",
            "stages": {"planning200": "planning200_retry1"}, "diagnostic_trials": ["planning200"]})
        case = self.collect()["planning"]["ik"]["cases"][0]
        self.assertEqual(case["stage"], "planning200_retry1")
        self.assertEqual(case["stages"], traces)
        self.assertEqual(case["search_stages"], traces)
        self.assertIsNone(case["search_stages"][0]["lower_bound"])
        self.assertIsNone(case["search_stages"][0]["suboptimality_bound"])
        self.assertEqual(case["bound"], 1.8)

    def test_d_retry_success_replaces_failed_attempt_without_rewriting_history(self):
        for stage, passed in (("d_e2e", False), ("d_e2e_attempt2", True)):
            pairs = [self.fixture_attempt(stage, name, passed=passed) for name in report.FIXTURES[1:]]
            self.stage(stage, [p[0] for p in pairs], [p[1] for p in pairs])
        m = self.collect()
        self.assertTrue(m["physical"]["d_e2e_passed"])
        self.assertEqual(len(m["physical"]["d_e2e_attempts"]), 4)
        self.assertTrue(all("attempt2" in r["source"] for r in m["physical"]["d_e2e"]))

    def test_d_outcome_requires_native_source_inside_v133_and_matching_task(self):
        proofs = {}
        for name in report.FIXTURES[1:]:
            _, _, path = self.fixture_attempt("d_manual_attempt2", name)
            proofs[name] = {"summary": str(path), "passed": False}
        self.save(self.base / "physical_stage_outcome.json", {"finished": True, "passed": False, "fixtures": proofs})
        self.assertTrue(self.collect()["physical"]["d_e2e_passed"])
        mismatch = Path(proofs[report.FIXTURES[1]]["summary"])
        data = json.loads(mismatch.read_text(encoding="utf-8"))
        self.save(mismatch, {**data, "task": "wrong_task"})
        self.assertFalse(self.collect()["physical"]["d_e2e_passed"])
        self.save(mismatch, data)
        outside = self.project / "old_v132/summary.json"
        self.save(outside, self.real(task=report.FIXTURES[1]))
        proofs[report.FIXTURES[1]] = {"summary": str(outside), "passed": True}
        self.save(self.base / "physical_stage_outcome.json", {"finished": True, "passed": True, "fixtures": proofs})
        self.assertFalse(self.collect()["physical"]["d_e2e_passed"])

    def test_d_new_failed_attempt_cannot_be_covered_by_old_pass_or_outcome(self):
        proofs = {}
        for stage, passed in (("d_e2e", True), ("d_e2e_attempt2", False)):
            pairs = [self.fixture_attempt(stage, name, passed=passed) for name in report.FIXTURES[1:]]
            self.stage(stage, [p[0] for p in pairs], [p[1] for p in pairs])
            if stage == "d_e2e":
                proofs = {name: {"summary": str(p[2])} for name, p in zip(report.FIXTURES[1:], pairs)}
        self.save(self.base / "physical_stage_outcome.json", {"finished": True, "fixtures": proofs})
        self.assertFalse(self.collect()["physical"]["d_e2e_passed"])

    def test_fixture_ik_comparison_uses_original_table_and_real_complete_cache(self):
        name = "s3a_cantilever_sim"
        job, row, source = self.fixture_attempt("d_e2e_attempt2", name, passed=False)
        self.stage("d_e2e_attempt2", [job], [row])
        baseline = self.base / "baseline_code/tasks/sim" / (name + ".toml")
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text('name="' + name + '"\narms=["A","B"]\n[[location]]\nid="stock"\n'
            '[capabilities.reach]\ndefault="UNKNOWN"\n[capabilities.reach.valid]\nA=["stock"]\n'
            '[capabilities.reach.invalid]\nB=["stock"]\n', encoding="utf-8")
        qualified = source.parent / (name + ".qualified.toml")
        qualified.write_text('name="' + name + '"\ncapability_source="ik_qualified"\narms=["A","B"]\n[[location]]\nid="stock"\n'
            '[capabilities.reach]\ndefault="UNKNOWN"\n[capabilities.reach.valid]\nB=["stock"]\n', encoding="utf-8")
        self.save(source.parent / "prequalification.json", {"total": 2, "checked": 2,
            "records": [{"arm": "A", "place": "stock", "status": "UNKNOWN"}, {"arm": "B", "place": "stock", "status": "VALID"}]})
        m = self.collect()["fixture_ik_differences"][name]
        self.assertEqual(m["status"], "compared")
        self.assertEqual(m["transitions"], {"VALID→UNKNOWN": 1, "INVALID→VALID": 1})
        self.assertEqual(len(m["changes"]), 2)
        qualified.write_text(qualified.read_text(encoding="utf-8").replace("ik_qualified", "distance_prior"), encoding="utf-8")
        self.assertEqual(self.collect()["fixture_ik_differences"][name]["status"], "pending")

    def test_empty_evidence_cannot_pass(self):
        m = self.collect()
        self.assertTrue(all(c["status"] == "not_met" for c in m["acceptance"].values()))
        self.assertEqual(m["simulation"]["denominator"], 20)
        self.assertFalse(m["regressions"]["passed"])

    def test_planning_uses_manifest_table_and_unknown_missing_time_fails(self):
        jobs, rows = [], []
        for family in report.FAMILIES:
            for index in range(40):
                ident = family + str(index)
                jobs.append({"id": ident, "phase": "plan", "kind": "plan", "table": "qualified", "family": family, "index": index})
                rows.append({"id": ident, "phase": "complete", "status": "solved", "passed": True, "bound": 1., "elapsed_seconds": 1.})
        rows[0].update(status="unknown", passed=False)
        rows[0].pop("elapsed_seconds")
        self.stage("planning200", jobs, rows)
        m = self.collect()
        self.assertTrue(m["planning"]["ik"]["thresholds_passed"])
        self.assertFalse(m["planning"]["unknown_within_5s"])
        self.assertEqual(m["planning"]["ik"]["families"]["F1"]["solved"], 39)
        self.assertEqual(m["acceptance"]["planning_200"]["status"], "not_met")

    def test_missing_samples_keep_denominator_and_final_not_nominal_ratio(self):
        job, row = self.native("one", family="F1", nominal_makespan_seconds=100.,
            phase_seconds={"compile": 2.}, step_durations=[{"estimated_seconds": 9., "actual_seconds": 10.}])
        self.stage("sim20", [job], [row])
        m = self.collect()
        self.assertEqual(m["simulation"]["success_rate"], .05)
        self.assertFalse(m["simulation"]["each_family_four_attempts"])
        self.assertEqual(m["ratios"]["sim20_final_estimate_to_actual"]["median"], .9)
        self.assertEqual(m["native_runs"][0]["initial_estimate_to_actual"], .2)
        self.assertEqual(m["simulation"]["cases"][0]["phase_seconds"], {"compile": 2.})

    def test_passed_journal_without_native_file_is_not_real_pass(self):
        job = {"id": "fake", "phase": "sim", "kind": "sim", "family": "F1", "output": str(self.base / "absent")}
        self.stage("sim20", [job], [{"id": "fake", **self.real()}])
        self.assertEqual(self.collect()["simulation"]["passed"], 0)

    def test_structured_acceptance_reasons_remain_parseable_and_countable(self):
        reason = [{"code": "support_ready_missing_measured_force", "node": "support"}]
        job, row = self.native("structured_failure", family="F4", passed=False, status="failed", reason=reason)
        self.stage("sim20", [job], [row])
        m = self.collect()
        self.assertEqual(m["simulation"]["cases"][0]["reason"], reason)
        self.assertIn("support_ready_missing_measured_force", next(iter(m["simulation"]["families"]["F4"]["failure_reasons"])))

    def test_skipped_journal_cannot_adopt_an_older_successful_summary(self):
        job, row = self.native("stale", family="F1")
        row.update(status="skipped", passed=False, executed=False)
        self.stage("sim20", [job], [row])
        m = self.collect()
        self.assertEqual(m["simulation"]["passed"], 0)
        self.assertEqual(m["ratios"]["sim20_final_estimate_to_actual"]["n"], 0)

    def test_failed_gates_keep_sources_and_one_test_zero_skip_required(self):
        jobs, rows = [], []
        for index, test in enumerate(report.LIVE_TESTS):
            job, row = self.native("gate" + str(index))
            job.update(phase="gates", kind="gate", test_name=test)
            row.update(tests_run=1, test_skipped=1 if index == 2 else 0, test_failures=0, test_errors=0)
            jobs.append(job); rows.append(row)
        self.stage("gates", jobs, rows)
        gates = self.collect()["physical"]["gates"]
        self.assertEqual(sum(g["passed"] for g in gates), 4)
        self.assertIn("summary.json", gates[2]["source"])

    def test_d_strict_geometry_is_independent_of_real_gated_passes(self):
        jobs, rows = [], []
        for index, test in enumerate(report.LIVE_TESTS):
            job, row = self.native("geometry_gate" + str(index))
            job.update(phase="gates", kind="gate", test_name=test)
            row.update(tests_run=1, test_skipped=0, test_failures=0, test_errors=0)
            jobs.append(job); rows.append(row)
        self.stage("gates", jobs, rows)
        self.save(self.base / "support_geometry.json", {"strict_tip_centre_80mm_passed": False,
            "whole_tip_faces_in_beam_width": False, "exact_geometry_contract_passed": False,
            "nearest_tip_face_80mm_passed": True, "fixed_shift_clearance_and_contact_passed": True})
        m = self.collect()
        self.assertEqual(m["physical"]["gates_passed"], 5)
        self.assertEqual(m["d_geometry_contract"]["status"], "not_met")
        self.assertFalse(m["d_geometry_contract"]["checks"]["strict_tip_centre_80mm_passed"]["passed"])
        self.assertFalse(m["revision_passed"])
        self.assertIn("D 严格几何约束未通过", report.markdown(m))

    def test_section_eight_has_six_independent_checks_for_no_reach_and_physical(self):
        pairs = [self.fixture_attempt("d_e2e", name) for name in report.FIXTURES[1:]]
        self.stage("d_e2e", [p[0] for p in pairs], [p[1] for p in pairs])
        jobs, rows = [], []
        for index, test in enumerate(report.LIVE_TESTS):
            job, row = self.native("split_gate" + str(index))
            job.update(phase="gates", kind="gate", test_name=test)
            row.update(tests_run=1, test_skipped=0, test_failures=0, test_errors=0)
            jobs.append(job); rows.append(row)
        self.stage("gates", jobs, rows)
        m = self.collect()
        self.assertEqual(len(m["acceptance"]), 6)
        self.assertEqual(m["acceptance"]["physical_fixtures_and_gates"]["status"], "passed")
        self.assertEqual(m["acceptance"]["no_reach_pipeline"]["status"], "not_met")

    def test_native_duplicate_logs_count_once(self):
        job, row = self.native("same")
        self.stage("sim20", [dict(job, family="F1")], [row])
        self.save(self.base / "copy/summary.json", self.real(task="same", simulator_log=row["simulator_log"]))
        m = self.collect()
        self.assertEqual(len(m["native_runs"]), 1)
        self.assertEqual(len(m["native_runs"][0]["duplicate_sources"]), 1)

    def test_sim_twenty_requires_four_attempts_per_family_and_fourteen_real_passes(self):
        jobs, rows = [], []
        for family_index, family in enumerate(report.FAMILIES):
            for index in range(4):
                passed = family_index * 4 + index < 14
                job, row = self.native(family + str(index), family=family,
                    passed=passed, status="passed" if passed else "failed")
                jobs.append(job); rows.append(row)
        self.stage("sim20", jobs, rows)
        m = self.collect()
        self.assertEqual(m["simulation"]["passed"], 14)
        self.assertEqual(m["acceptance"]["sim_20"]["status"], "passed")
        self.assertEqual(m["acceptance"]["calibrated_estimate_ratio"]["status"], "not_met")

    def test_calibration_requires_thirty_distinct_real_source_steps(self):
        source = self.save(self.project / "old_native/summary.json", self.real())
        sample = {"source_summary": str(source), "step": 0, "distance_m": .3, "actual_seconds": 3.}
        self.save(self.base / "calibration.json", {"skills": {"pick": {}}, "samples": [sample] * 30})
        self.assertEqual(self.collect()["calibration"]["verified_distinct_successful_steps"], 1)
        self.save(self.base / "calibration.json", {"skills": {"pick": {}},
            "samples": [{**sample, "step": index} for index in range(30)]})
        self.assertTrue(self.collect()["calibration"]["passed"])

    def test_qualification_reports_all_combinations_and_transitions(self):
        task = self.project / "q.toml"
        task.write_text('name="q"\narms=["A","B"]\n[[location]]\nid="stock"\n[[slot]]\nid="goal"\n', encoding="utf-8")
        output = self.base / "qualification/F1/000"
        job = {"id": "q", "phase": "sim", "kind": "qualify", "task": str(task), "family": "F1", "output": str(output)}
        self.stage("qualify200", [job], [{"id": "q", "status": "completed", "log_file": str(output / "q/qualify_log.json")}])
        updates = [{"arm": a, "place": p, "status": "INVALID" if a == "B" else "VALID"} for a in ["A", "B"] for p in ["stock", "goal"]]
        self.save(output / "q/qualify_log.json", {"total": 4, "checked": 4, "updates": updates, "records": [{**u, "seconds": .1} for u in updates], "source_before": [{"fact": f'reach({u["arm"]}, {u["place"]})', "status": "UNKNOWN"} for u in updates]})
        q = self.collect()["qualification"]
        self.assertEqual(q["expected_combinations"], 4)
        self.assertEqual(q["unknown_to_valid"], 2)
        self.assertEqual(q["unknown_to_invalid"], 2)
        self.assertTrue(q["cases"][0]["all_combinations_recorded"])

    def test_regression_evidence_cannot_be_invented_from_status(self):
        for name in ("abc", "e"):
            self.save(self.project / "out" / f"v133_{name}_full.json", {"status": "passed", "tests_run": 100, "skipped": 6,
                "per_test": [{"id": "tests.test_v11_search.X.test_optimal", "status": "passed"}]})
        m = self.collect()
        self.assertFalse(m["regressions"]["passed"])
        self.assertFalse(m["regressions"]["full"][0]["v11_modules"]["tests.test_v11_admissibility"]["passed"])

    def test_rebuild_preserves_old_milestones_and_journal_bytes(self):
        path = self.stage("planning200", [], [])
        old = {"M1": {"status": "passed"}, "M2": {"evidence": "old"}, "M3": {"accepted": True}, "M4": {"old": True}, "history": {"x": [1]}}
        self.save(self.base.parent / "milestones.json", old)
        before = path.read_bytes()
        report.rebuild(self.base, project=self.project)
        result = json.loads((self.base.parent / "milestones.json").read_text(encoding="utf-8"))
        for key in old:
            self.assertEqual(result[key], old[key])
        self.assertIn("v1.3.3", result)
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue((self.base.parent / "summary.md").exists())
        self.assertTrue((self.base / "fixture_ik_differences.json").exists())


if __name__ == "__main__":
    unittest.main()
