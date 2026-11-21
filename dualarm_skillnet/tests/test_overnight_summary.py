"""Evidence accounting with synthetic files only; no planner or simulator."""
from pathlib import Path
import tempfile
import unittest

from tools import overnight_batch as batch
from tools import overnight_summary as report


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="summary_mock_", dir=report.ROOT / "out")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.state = self.base / "overnight/batch_state"
        self.jobs = []
        self.rows = batch.Journal(self.state)

    def save(self, relative, data):
        path = self.base / relative
        batch.atomic_json(path, data)
        return path

    def manifest(self):
        return self.save("overnight/manifest.json", {"state_dir": str(self.state), "jobs": self.jobs})

    def real(self, **extra):
        return {"passed": True, "status": "passed", "profile": "skillnet_v1_3",
                "simulator_closed": True, "simulator_pid": 1234, "actual_makespan_seconds": 10., **extra}

    def planning(self, failures=None):
        failures = failures or {}
        for family in report.FAMILIES:
            for index in range(40):
                job = {"id": f"{family}-{index}", "phase": "plan", "family": family,
                       "cohort": "planning_200", "output": str(self.base / f"plans/{family}-{index}")}
                self.jobs.append(job)
                failed = index < failures.get(family, 0)
                self.rows.append({**job, "status": "budget" if failed else "solved", "passed": not failed,
                                  "reason": "unchanged search budget" if failed else None,
                                  "auxiliary": index % 2 == 0})

    def simulations(self, *, skip=0, fail=0):
        for index in range(20):
            job = {"id": f"sim-{index}", "phase": "sim", "family": report.FAMILIES[index//4],
                   "selection": 1, "output": str(self.base / f"sim/{index}")}
            self.jobs.append(job)
            result = self.real(makespan=5.)
            if index < skip:
                result.update(status="skipped", passed=False, executed=False, reason="no solved candidates")
            elif index < skip+fail:
                result.update(status="failed", passed=False, failure_phase="compile", compile_failed=True)
                result.pop("actual_makespan_seconds")
            self.rows.append({**job, **result})

    def test_empty_inputs_preserve_old_m1_m2_and_archive_old_m3_once(self):
        old = {"M1": {"status": "passed", "proof": 1}, "M2": {"status": "passed", "proof": 2},
               "M3": {"status": "blocked", "handover_attempts": [1, 2, 3]}}
        self.save("milestones.json", old)
        metrics = report.rebuild(self.base)
        milestones = batch.read_json(self.base / "milestones.json")
        self.assertEqual(milestones["M1"], old["M1"])
        self.assertEqual(milestones["M2"], old["M2"])
        self.assertEqual(milestones["history"]["M3"], [old["M3"]])
        report.rebuild(self.base)
        self.assertEqual(batch.read_json(self.base / "milestones.json")["history"]["M3"], [old["M3"]])
        self.assertFalse(metrics["M4"]["passed"])
        self.assertFalse(metrics["M5"]["passed"])
        self.assertEqual(metrics["simulations"]["sim20"]["success_rate_executed"], None)

    def test_family_thresholds_use_all_forty_and_list_every_budget(self):
        self.planning({"F4": 1, "F5": 4})
        self.manifest()
        metrics = report.collect(self.base)
        self.assertTrue(metrics["planning"]["thresholds_passed"])
        self.assertEqual(len(metrics["planning"]["unsolved"]), 5)
        self.assertTrue(all(row["reason"] == "unchanged search budget" for row in metrics["planning"]["unsolved"]))
        self.assertEqual(metrics["planning"]["families"]["F5"]["success_rate_requested"], .9)
        self.assertEqual(metrics["planning"]["families"]["F1"]["auxiliary_rate_executed"], .5)
        self.assertEqual(metrics["planning"]["families"]["F1"]["auxiliary_rate_passed"], .5)

    def test_simulation_slots_do_not_inflate_executed_denominator(self):
        self.planning()
        self.simulations(skip=10)
        self.save("overnight/generator_diversity.json", {"exact_contract_passed": True})
        self.manifest()
        metrics = report.collect(self.base)
        sim = metrics["simulations"]["sim20"]
        self.assertEqual((sim["requested"], sim["executed"], sim["skipped"], sim["passed"]), (20, 10, 10, 10))
        self.assertEqual(sim["success_rate_executed"], 1.)
        self.assertEqual(sim["success_rate_requested"], .5)
        self.assertFalse(metrics["M5"]["passed"])

    def test_m5_needs_generator_exact_contract_even_when_success_rates_pass(self):
        self.planning()
        self.simulations(fail=4)
        self.save("overnight/generator_diversity.json", {"exact_contract_passed": True})
        self.manifest()
        metrics = report.collect(self.base)
        self.assertTrue(metrics["M5"]["passed"])
        self.assertEqual(metrics["simulations"]["sim20"]["compilation_failure_rate_executed"], .2)
        self.assertEqual(metrics["simulations"]["sim20"]["ratios"]["actual_to_estimated"]["n"], 16)
        self.assertEqual(metrics["simulations"]["sim20"]["ratios"]["estimated_to_actual"]["median"], .5)
        self.assertEqual(metrics["ratios"]["actual_to_estimated"]["n"], 0)
        self.save("overnight/generator_diversity.json", {"exact_contract_passed": False,
                  "exact_contract_failure_reasons": ["F4 x standard deviation is below contract"]})
        self.assertFalse(report.collect(self.base)["M5"]["passed"])

    def test_real_runs_are_deduplicated_and_both_ratio_directions_reported(self):
        data = self.real(task="mock", task_file="mock.toml", wall_seconds=40.,
                         simulator_log="out/mock-unique-launch.log", phase_seconds={"execute": 30.},
                         step_durations=[{"action": "mock", "estimated_seconds": 2., "actual_seconds": 8.}])
        for directory in ("overnight/a", "m3_attempt_5/copied"):
            self.save(directory + "/summary.json", data)
            self.save(directory + "/initial_plan.json", {"makespan": 5.})
        metrics = report.collect(self.base)
        self.assertEqual(metrics["real_run_count"], 1)
        self.assertEqual(metrics["real_attempt_wall_seconds"], 40.)
        self.assertEqual(metrics["phase_seconds"], {"execute": 30.})
        self.assertEqual(metrics["ratios"]["actual_to_estimated"]["median"], 2.)
        self.assertEqual(metrics["ratios"]["estimated_to_actual"]["median"], .5)
        self.assertEqual(metrics["ratios"]["step_actual_to_estimated"]["median"], 4.)
        self.assertEqual(len(metrics["real_runs"][0]["duplicate_sources"]), 1)

    def test_m4_aggregate_boolean_never_substitutes_for_real_cut_and_qualification_evidence(self):
        self.save("overnight/m4_outcome.json", {"M4_passed": True})
        self.assertFalse(report.collect(self.base)["M4"]["passed"])
        fixture_rows = []
        for task in report.FIXTURES:
            path = self.save(f"overnight/{task}/summary.json", self.real(task=task))
            fixture_rows.append({"task": task, "summary": str(path)})
        invalid = self.save("overnight/cut/summary.json", self.real(motion_feedback_count=1,
            feedback=["运动失败：place；已禁止该状态下的动作并重新搜索。 "]))
        unknown = self.save("overnight/qualify/summary.json", self.real(qualification_calls=1,
            feedback=["运动后端按需验证：reach(B, stock) → VALID。"]))
        self.save("overnight/m4_outcome.json", {"fixtures": fixture_rows,
            "invalid_feedback": {"summary": str(invalid)}, "unknown_qualification": {"summary": str(unknown)}})
        self.assertFalse(report.collect(self.base)["M4"]["passed"])
        self.save("overnight/cut/motion_backend.json", {"validations": [{"status": "INVALID"}]})
        self.save("overnight/qualify/motion_backend.json", {"qualifications": [{"fact": "reach(B, stock)", "status": "VALID"}]})
        self.assertTrue(report.collect(self.base)["M4"]["passed"])

    def test_gate_proof_requires_all_five_actual_single_test_runs_without_skip(self):
        for index, name in enumerate(batch.LIVE_TESTS):
            job = {"id": f"gate-{index}", "phase": "gates", "test_name": name}
            self.jobs.append(job)
            self.rows.append({**job,
                **self.real(), "tests_run": 1, "test_skipped": 1 if index == 4 else 0,
                "test_errors": 0, "test_failures": 0, "output": "mock-gate-evidence"})
        self.manifest()
        metrics = report.collect(self.base)
        self.assertEqual(sum(row["passed"] for row in metrics["gates"]["tests"]), 4)
        self.assertFalse(metrics["gates"]["all_five_passed"])
        self.assertTrue(all(row["source"] == "mock-gate-evidence"
                            for row in metrics["gates"]["tests"]))

    def test_failure_categories_keep_phase_timeout_and_each_id_reason(self):
        failures = [
            {"id": "compile", "phase": "sim", "selection": 1, "family": "F1", "status": "failed",
             "passed": False, "failure_phase": "compile", "motion_status": "UNKNOWN", "reason": "finite IK seeds", "wall_seconds": 6.},
            {"id": "deadline", "phase": "sim", "selection": 2, "family": "F1", "status": "timeout",
             "passed": False, "failure_phase": "task_wall_timeout", "reason": "hard cap", "wall_seconds": 9.}]
        for row in failures:
            job = {key: row[key] for key in ("id", "phase", "selection", "family")}
            self.jobs.append(job)
            self.rows.append(row)
            summary = self.real(**{key: value for key, value in row.items() if key not in {"id", "phase", "selection", "family"}},
                                task="mock", task_file="mock.toml", simulator_log="out/launch-"+row["id"]+".log")
            self.save(f"overnight/{row['id']}/summary.json", summary)
            if row["id"] == "compile":
                self.save("m3_attempt_5/copied_failure/summary.json", summary)
        self.manifest()
        metrics = report.collect(self.base)
        self.assertEqual(metrics["real_failure_categories"], [
            {"failure_phase": "compile", "category": "motion_unknown", "count": 1},
            {"failure_phase": "task_wall_timeout", "category": "timeout", "count": 1}])
        self.assertEqual(len(metrics["simulations"]["sim20"]["failure_categories"]), 1)
        self.assertEqual(len(metrics["simulations"]["sim40"]["failure_categories"]), 2)
        text = report.markdown(metrics)
        self.assertIn("finite IK seeds", text)
        self.assertIn("hard cap", text)
        self.assertIn("| 整夜去重运行 | compile | motion_unknown | 1 |", text)

    def test_scope_wall_totals_use_exact_gate_jobs_and_skip_summary_references(self):
        for index, name in enumerate(batch.LIVE_TESTS):
            job = {"id": f"gate-{index}", "phase": "gates", "test_name": name}
            self.jobs.append(job)
            row = {**job, **self.real(wall_seconds=float(index+1)), "tests_run": 1,
                   "test_skipped": 0, "test_errors": 0, "test_failures": 0}
            self.rows.append(row)
        self.rows.append({"id": "unrequested-gate", "phase": "gates", "test_name": batch.LIVE_TESTS[0],
                          **self.real(wall_seconds=999.)})
        for index in range(2):
            job = {"id": f"sim-{index}", "phase": "sim", "selection": 1, "family": "F1"}
            self.jobs.append(job)
            self.rows.append({**job, **self.real(wall_seconds=5., simulator_log="out/shared-reference.log")})
        self.save("overnight/copy_of_gate/summary.json", self.real(task="gate", task_file="mock.toml", wall_seconds=1.))
        self.manifest()
        metrics = report.collect(self.base)
        self.assertEqual(metrics["gates"]["execution_wall"]["seconds"], 15.)
        self.assertEqual(metrics["gates"]["execution_wall"]["measured_runs"], 5)
        self.assertTrue(metrics["gates"]["all_five_passed"])
        self.assertEqual(metrics["simulations"]["sim20"]["execution_wall"]["seconds"], 5.)
        self.assertEqual(metrics["simulations"]["sim20"]["execution_wall"]["duplicate_ids_excluded"], ["sim-1"])
        self.assertEqual(metrics["real_attempt_wall_seconds"], 1.)
        self.assertIn("| 精确五项真实门控 | 5 | 5 | 15.000000 |", report.markdown(metrics))

    def test_post_m3_worker_regression_is_extra_and_cannot_replace_initial_three(self):
        self.save("out/overnight_post_m3_generator_regression.json", {"label": "post_m3_generator",
            "status": "passed", "tests_run": 449, "skipped": 6, "total_seconds": 77.})
        metrics = report.collect(self.base)
        self.assertEqual(len(metrics["regressions"]), 1)
        self.assertFalse(metrics["regressions_passed"])
        self.save("overnight_regression_mock.json", {"runs": [
            {"label": label, "status": "passed", "tests_run": 391, "skipped": 6, "total_seconds": 60.}
            for label in ("default_1", "default_2", "seed0")]})
        metrics = report.collect(self.base)
        self.assertTrue(metrics["regressions_passed"])
        self.assertEqual(len(metrics["regressions"]), 4)

    def test_final_regression_is_optional_and_torn_artifact_cannot_replace_initial_runs(self):
        self.save("out/overnight_final_regression.json", {"label": "default_1", "status": "passed",
            "tests_run": 450, "skipped": 6, "total_seconds": 80.})
        metrics = report.collect(self.base)
        self.assertEqual(metrics["regressions"][0]["label"], "final_after_sim")
        self.assertFalse(metrics["regressions_passed"])
        self.save("overnight_regression_mock.json", {"runs": [
            {"label": label, "status": "passed", "tests_run": 391, "skipped": 6, "total_seconds": 60.}
            for label in ("default_1", "default_2", "seed0")]})
        (self.base / "out/overnight_final_regression.json").write_text('{"status":', encoding="utf-8")
        metrics = report.collect(self.base)
        self.assertTrue(metrics["regressions_passed"])
        self.assertEqual(metrics["regressions"][-1]["status"], "incomplete_artifact")
        self.assertIn("不计为通过", report.markdown(metrics))

    def test_final_route_ratios_keep_initial_zero_and_changed_route_separate(self):
        for name, initial, final in (("unknown", 0., 5.), ("cut", 2., 4.)):
            directory = self.base / "overnight" / name
            self.save(f"overnight/{name}/summary.json", self.real(task=name, task_file="mock.toml", wall_seconds=1.,
                simulator_log=f"out/mock-ratio-{name}.log"))
            self.save(f"overnight/{name}/initial_plan.json", {"makespan": initial})
            self.save(f"overnight/{name}/plan.json", {"status": "solved", "makespan": final})
            job = {"id": name, "phase": "sim", "family": "F1", "selection": 1}
            self.jobs.append(job)
            self.rows.append({**job, **self.real(), "makespan": initial, "evidence_directory": str(directory)})
        self.manifest()
        metrics = report.collect(self.base)
        self.assertEqual(metrics["ratios"]["actual_to_estimated"]["n"], 1)
        self.assertEqual(metrics["ratios"]["actual_to_estimated"]["median"], 5.)
        self.assertEqual(metrics["ratios"]["actual_to_final_plan_estimated"]["n"], 2)
        self.assertEqual(metrics["ratios"]["actual_to_final_plan_estimated"]["median"], 2.25)
        self.assertEqual(metrics["ratios"]["final_plan_estimated_to_actual"]["median"], .45)
        self.assertEqual(metrics["simulations"]["sim20"]["ratios"]["actual_to_final_plan_estimated"]["median"], 2.25)

    def test_findings_and_prechecks_use_factual_counts_without_changing_conclusions(self):
        samples = [
            {"id": "budget", "family": "F1", "status": "budget", "passed": False, "expanded": 200000},
            {"id": "unknown", "family": "F4", "status": "unknown", "passed": False, "unknown": ["reach(A, beam)", "reach(B, beam)"]}]
        for row in samples:
            job = {"id": row["id"], "family": row["family"], "phase": "plan", "cohort": "planning_200"}
            self.jobs.append(job)
            self.rows.append({**job, **row})
        precheck = {"id": "precheck-tower", "phase": "plan", "family": "fixture", "cohort": "precheck",
                    "witness_estimate_seconds": 30.}
        self.jobs.append(precheck)
        self.rows.append({**precheck, "status": "solved", "passed": True, "makespan": 26.})
        self.save("overnight/m4_outcome.json", {"notes": ["首个目录顶抓被水平SupportShoe穿入24mm阻塞。"]})
        self.save("overnight/generator_diversity.json", {"exact_contract_passed": False,
            "exact_contract_failure_reasons": ["F4 shared reach lens conflicts with per-axis xstd"]})
        self.manifest()
        metrics = report.collect(self.base)
        self.assertEqual(metrics["planning"]["families"]["F1"]["diagnostics"]["reason_counts"], {"budget_expanded_ge_200000": 1})
        self.assertEqual(metrics["planning"]["families"]["F4"]["diagnostics"]["unknown_facts"]["reach(A, beam)"], 1)
        self.assertIsNone(metrics["planning"]["families"]["F4"]["auxiliary_rate_passed"])
        self.assertEqual(metrics["precheck"]["comparisons"][0]["independent_to_witness_ratio"], 26./30.)
        text = report.markdown(metrics)
        self.assertIn("## 发现的问题与建议", text)
        self.assertIn("24mm", text)
        self.assertIn("1 项 budget", text)
        self.assertIn("F4 shared reach lens", text)
        self.assertIn("预检见证与独立规划事后对照", text)
        self.assertFalse(metrics["M4"]["passed"])
        self.assertFalse(metrics["M5"]["passed"])

    def test_quartiles_use_endpoint_interpolation_and_ignore_nonfinite_values(self):
        result = report.distribution([1., 3., None, float("inf"), float("nan")])
        self.assertEqual(result, {"n": 2, "min": 1., "p25": 1.5, "median": 2., "p75": 2.5, "max": 3.})


if __name__ == "__main__":
    unittest.main()
