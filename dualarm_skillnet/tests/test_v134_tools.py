"""Namespace/immutable provenance contracts; no planning, tests, or native runs."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from tools import v134_batch as batch
from tools import v134_guard as guard


class V134ToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=batch.ROOT / "out", prefix="v134_tools_mock_")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.base = self.project / "results/sim_v1_3/v134"
        self.source = self.project / "results/sim_v1_3/v133"
        for name in set(batch.FINGERPRINT_FILES) | {"simbridge/" + name for name in batch.MOTION_FILES}:
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# immutable mock fingerprint\n", encoding="utf-8")
        self.save(self.source / "calibration.json", {"skills": {"pick": {}}})
        self.inputs = []
        for family in batch.FAMILIES:
            for index in range(40):
                name = f"gen_{family.lower()}_s20261115_{index:03d}"
                path = self.source / "qualification" / family / f"{index:03d}" / name / (name + ".qualified.toml")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'name="{name}"\narms=["A","B"]\n[capabilities.reach]\ndefault="UNKNOWN"\n', encoding="utf-8")
                self.inputs.append({"id": "old-" + name, "family": family, "index": index,
                    "table": "qualified", "task": str(path), "output": str(self.source / "old_plans" / name)})
        self.save(self.source / "experiment_selection.json", {"stages": {"planning200": "planning200_retry1"}})
        self.save(self.source / "manifests/planning200_retry1.json", {"jobs": self.inputs})
        self.fixed = [{"id": f"sim20-{f.lower()}-{i}", "family": f, "task": next(j["task"] for j in self.inputs if j["family"] == f and j["index"] == i),
                       "source_plan": "old-solved-" + f + str(i)} for f in batch.FAMILIES for i in range(4)]
        self.save(self.source / "manifests/sim20.json", {"jobs": self.fixed})
        self.save(self.base / "baseline.json", {"project_files": {}, "v133_files": {}})
        self.fingerprint = batch.planning_fingerprint(project=self.project)
        self.full = self.save(self.project / "out/v134_four_full.json", {"status": "passed", "tests_run": 20,
            "experiment_namespace": batch.NAMESPACE, "fixes_completed": ["B", "A", "C", "D"],
            "planning_code_sha256": self.fingerprint, "per_test": [{"id": m + ".X.test_case", "status": "passed"} for m in batch.V11_MODULES]})
        self.original = self.save(self.base / "original11.json", {"passed": True, "experiment_namespace": batch.NAMESPACE,
            "fixes_completed": ["B", "A", "C", "D"], "planning_code_sha256": self.fingerprint,
            "rows": [{"task": str(i), "passed": True} for i in range(11)]})

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def register(self):
        return batch.register(self.full, self.original, base=self.base, project=self.project)

    def d_results(self):
        mapping = {}
        path = batch.prepare_qualify5(base=self.base, project=self.project)
        manifest = batch.old.load_manifest(path)
        records = []
        for job in manifest["jobs"]:
            name = batch._task_name(job["task"])
            target = Path(job["output"]) / name / (name + ".qualified.toml")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(job["task"]).read_bytes())
            log = self.save(target.parent / "qualify_log.json", {"source_task": job["task"], "source_sha256": batch.sha256(job["task"]),
                "qualified_task": str(target), "status": "completed", "simulator_closed": True, "simulator_pid": 123, "escalation": []})
            records.append({"id": job["id"], "definition_sha256": job["definition_sha256"], "status": "completed",
                "simulator_closed": True, "qualified_task": str(target), "log_file": str(log)})
            mapping[name] = {"qualified_task": str(target)}
        journal = Path(manifest["state_dir"]) / "completed.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        batch.collect_d_results(base=self.base, project=self.project)
        return mapping

    def test_registration_rejects_old_or_absent_namespace_without_writing_selection(self):
        for namespace in (None, "c_scale_retry1"):
            data = json.loads(self.full.read_text(encoding="utf-8"))
            if namespace is None:
                data.pop("experiment_namespace", None)
            else:
                data["experiment_namespace"] = namespace
            self.save(self.full, data)
            with self.assertRaisesRegex(ValueError, "namespace|命名空间"):
                self.register()
            self.assertFalse((self.base / "experiment_selection.json").exists())

    def test_registration_rejects_missing_fix_proof_or_stale_fingerprint(self):
        for field, value in (("fixes_completed", ["B", "A", "C"]), ("planning_code_sha256", {})):
            original = json.loads(self.full.read_text(encoding="utf-8"))
            self.save(self.full, {**original, field: value})
            with self.assertRaises(ValueError):
                self.register()
            self.save(self.full, original)

    def test_registration_seals_all_required_fingerprints_and_never_writes_v133(self):
        before = {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        path = self.register()
        proof = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(proof["selection_id"], "c_wall_v134")
        self.assertEqual(len(proof["qualification_inputs"]), 200)
        for name in ("skillnet/planning/bounds.py", "skillnet/planning/allocation.py", "skillnet/pipeline.py", "simbridge/backend.py", "simbridge/qualifier.py", "simbridge/__main__.py", "simbridge/taskio.py"):
            self.assertIn(name, proof["planning_code_sha256"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()})

    def test_planning_preparation_requires_d_five_and_uses_ik_only(self):
        self.register()
        with self.assertRaisesRegex(ValueError, "D|鉴定"):
            batch.prepare_planning(base=self.base, project=self.project)
        replacements = self.d_results()
        path = batch.prepare_planning(base=self.base, project=self.project)
        jobs = json.loads(path.read_text(encoding="utf-8"))["jobs"]
        self.assertEqual(len(jobs), 200)
        self.assertTrue(all(j["table"] == "qualified" for j in jobs))
        self.assertEqual(sum(Path(j["task"]).is_relative_to(self.base) for j in jobs), 5)
        self.assertEqual({j["task"] for j in jobs if Path(j["task"]).is_relative_to(self.base)}, {r["qualified_task"] for r in replacements.values()})

    def test_sim_samples_are_exact_old_twenty_order_with_no_sampling(self):
        self.register(); self.d_results()
        with self.assertRaisesRegex(ValueError, "planning200"):
            batch.prepare_fixed_sim20(base=self.base, project=self.project)
        with patch.object(batch, "planning_committed", return_value=True):
            path = batch.prepare_fixed_sim20(base=self.base, project=self.project)
        jobs = json.loads(path.read_text(encoding="utf-8"))["jobs"]
        self.assertEqual([j["source_job_id"] for j in jobs], [j["id"] for j in self.fixed])
        self.assertEqual([j["task"] for j in jobs], [j["task"] for j in self.fixed])
        self.assertTrue(all(j["experiment_namespace"] == batch.NAMESPACE for j in jobs))
        walls = json.loads(batch.prepare_walls(base=self.base, project=self.project).read_text(encoding="utf-8"))["jobs"]
        self.assertEqual([j["index"] for j in walls], [1, 9, 37])

    def test_outside_v134_outputs_are_rejected_before_any_write(self):
        self.register()
        with self.assertRaises(ValueError):
            batch.prepare_manifest("bad", [{"id": "bad", "phase": "plan", "kind": "plan", "task": self.inputs[0]["task"], "output": str(self.source / "oops")}], base=self.base, project=self.project)
        self.assertFalse((self.source / "oops").exists())

    def test_guard_denies_protected_code_old_results_and_new_old_result_files(self):
        protected = self.project / "simbridge/compiler.py"
        protected.write_text("original", encoding="utf-8")
        guarded_source = self.project / "guard_old"
        old = guarded_source / "result.json"
        self.save(old, {"status": "passed"})
        baseline = {"project_files": {**{name: batch.sha256(self.project / name) for name in batch.FINGERPRINT_FILES},
                    **{"simbridge/" + name: batch.sha256(self.project / "simbridge" / name) for name in batch.MOTION_FILES}},
                    "v133_files": {str(old.relative_to(self.project)).replace("\\", "/"): batch.sha256(old)}}
        fake_archive = Mock(check=Mock(return_value={"passed": True}))
        checker = guard.Guard(baseline, project=self.project, source_base=guarded_source, archive_guard=fake_archive)
        self.assertTrue(checker.check()["passed"])
        for name in ("simbridge/compiler.py", "simbridge/frames.py", "simbridge/runner.py", "skills/blocks_v1.toml", "skillnet/cost/model.py"):
            path = self.project / name
            before = path.read_bytes()
            path.write_text("changed", encoding="utf-8")
            self.assertFalse(checker.check()["passed"], name)
            path.write_bytes(before)
        extra = self.save(guarded_source / "unexpected.json", {})
        self.assertFalse(checker.check()["passed"])
        extra.unlink()
        old.write_text("changed", encoding="utf-8")
        self.assertFalse(checker.check()["passed"])

    def test_workflow_never_starts_native_when_planning_incomplete(self):
        from tools import v134_workflow as workflow
        stages = []
        def fake(stage):
            stages.append(stage)
            return {"status": "phase_budget_exhausted" if stage == "planning200" else "phase_complete"}
        result = workflow.run(base=self.base, project=self.project, ready=lambda: None,
            preparer=lambda stage: None, runner=fake, checker=lambda: {"passed": True}, reporter=lambda: None)
        self.assertEqual(result["status"], "interrupted")
        self.assertNotIn("sim20", stages)
        self.assertNotIn("walls3", stages)

    def test_native_phase_requires_all_two_hundred_committed_planning_jobs(self):
        from tools import v134_workflow as workflow
        runner = Mock()
        result = workflow.run(base=self.base, project=self.project, phase="native", ready=lambda: None,
            preparer=lambda stage: None, runner=runner, checker=lambda: {"passed": True}, reporter=lambda: None)
        self.assertEqual(result["status"], "interrupted")
        self.assertIn("planning200", result["reason"])
        runner.assert_not_called()

    def test_planning_supervisor_qualifies_five_then_stops_before_simulation(self):
        from tools import v134_workflow as workflow
        stages = []
        result = workflow.run(base=self.base, project=self.project, phase="planning", ready=lambda: None,
            preparer=lambda stage: None, runner=lambda stage: (stages.append(stage) or {"status": "phase_complete"}),
            checker=lambda: {"passed": True}, reporter=lambda: None)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(stages, ["motion_gate", "qualify5", "planning200"])

    def test_comparison_csv_preserves_missing_first_solution_and_explicit_d_denominator(self):
        from tools import v134_summary as summary
        self.register()
        sources = [self.inputs[0], next(j for j in self.inputs if j["family"] == "F3" and j["index"] == 5)]
        path = batch.prepare_manifest("planning200", [{"id": "mock-" + str(i), "kind": "plan", "phase": "plan",
            "family": src["family"], "index": src["index"], "task": src["task"], "d_requalified": i == 1,
            "output": str(self.base / "mock_plans" / str(i))} for i, src in enumerate(sources)], base=self.base, project=self.project)
        manifest = batch.old.load_manifest(path)
        rows = [{"id": j["id"], "definition_sha256": j["definition_sha256"], "status": "solved" if i == 0 else "no_plan",
            "passed": i == 0, "elapsed_seconds": .4, "first_solution_seconds": None, "first_solution_expanded": None,
            "physical_infeasible": i == 1, "physical_infeasible_goal_proof": ["filled(s5_1)"] if i == 1 else []}
            for i, j in enumerate(manifest["jobs"])]
        journal = Path(manifest["state_dir"]) / "completed.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        before = {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        metrics = summary.rebuild(base=self.base, project=self.project)
        self.assertEqual(metrics["planning_families"]["F3"]["original_denominator"], 40)
        self.assertEqual(metrics["planning_families"]["F3"]["feasible_denominator"], 39)
        self.assertEqual(len(metrics["solved_first_solution_over_fifteen_or_missing"]), 1)
        self.assertIn("first_solution_expanded", (self.base / "planning_200.csv").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(before, {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()})

    def test_hard_timeout_keeps_startup_phase_and_workcell_is_connected(self):
        root = self.base / "mock_native"
        progress = self.save(root / "simulation/task/run_progress.json", {"phase": "simulator_connect", "pid": 44})
        job = {"kind": "sim", "output": str(root)}
        first = batch.adapt_result(job, {"status": "timeout", "failure_phase": "task_wall_timeout"})
        self.assertEqual(first["startup_failure_phase"], "simulator_connect")
        self.save(progress, {"phase": "workcell", "pid": 44})
        second = batch.adapt_result(job, {"status": "timeout", "failure_phase": "task_wall_timeout"})
        self.assertTrue(second["startup_connect_completed"])
        self.assertNotIn("startup_failure_phase", second)

    def test_d_index_without_committed_escalation_proof_cannot_feed_planning(self):
        self.register(); self.d_results()
        index = json.loads((self.base / "qualification_index.json").read_text(encoding="utf-8"))
        entry = next(iter(index["tasks"].values()))
        log = json.loads(Path(entry["log_file"]).read_text(encoding="utf-8"))
        log.pop("escalation")
        self.save(Path(entry["log_file"]), log)
        with self.assertRaisesRegex(ValueError, "D"):
            batch.prepare_planning(base=self.base, project=self.project)

    def test_original11_comparison_preserves_optima_and_never_worsens_nonoptima(self):
        from tools.v134_originals import comparison
        exact = {"status": "solved", "makespan": 100., "bound": 1.}
        approximate = {"status": "solved", "makespan": 100., "bound": 2.}
        self.assertTrue(comparison(exact, "solved", 100., 1.))
        self.assertFalse(comparison(exact, "solved", 99., 1.))
        self.assertTrue(comparison(approximate, "solved", 90., 1.5))
        self.assertFalse(comparison(approximate, "solved", 101., 1.5))
        self.assertFalse(comparison(approximate, "solved", 90., 2.1))

    def test_three_startup_failures_across_stages_stop_and_connection_resets_counter(self):
        rows = [{"id": "startup-" + str(i), "phase": "sim", "status": "failed", "executed": True,
                 "startup_failure_phase": "simulator_connect", "completed_at": str(i)} for i in range(3)]
        for i, row in enumerate(rows):
            path = self.base / "states" / str(i) / "completed.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.assertIn("三次", batch.stop_reason(self.base))
        path = self.base / "states/last/completed.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": "connected", "phase": "sim", "status": "timeout", "executed": True,
            "startup_connect_completed": True, "completed_at": "9"}) + "\n", encoding="utf-8")
        self.assertIsNone(batch.stop_reason(self.base))

    def test_workflow_persists_stage_and_failure_events_preserving_root_steps(self):
        from tools import v134_workflow as workflow
        progress = self.save(self.base / "progress_state.json", {"version": "v134", "phase": "root_owned",
            "steps": [{"step": "root_B", "status": "passed"}], "failures": [{"issue": "old_failure"}],
            "remaining": ["root_item"], "custom": {"preserved": True}})
        report = self.base / "report.md"
        report.write_text("root authored report\n", encoding="utf-8")
        before = {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        result = workflow.run(base=self.base, project=self.project, phase="planning", ready=lambda: None,
            preparer=lambda stage: None,
            runner=lambda stage: {"status": "phase_budget_exhausted" if stage == "planning200" else "phase_complete", "reason": "保留失败原因" if stage == "planning200" else None},
            checker=lambda: {"passed": True}, reporter=lambda: None)
        saved = json.loads(progress.read_text(encoding="utf-8"))
        self.assertEqual(saved["steps"][0], {"step": "root_B", "status": "passed"})
        self.assertEqual(saved["phase"], "root_owned")
        self.assertEqual(saved["remaining"], ["root_item"])
        self.assertEqual(saved["custom"], {"preserved": True})
        self.assertEqual(saved["failures"][0], {"issue": "old_failure"})
        self.assertTrue(any(e.get("stage") == "planning200" and e.get("status") == "phase_budget_exhausted" for e in saved["steps"]))
        self.assertEqual(saved["steps"][-1]["status"], "interrupted")
        self.assertIn("保留失败原因", report.read_text(encoding="utf-8"))
        self.assertTrue(report.read_text(encoding="utf-8").startswith("root authored report\n"))
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(before, {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()})
        count = len(saved["steps"])
        workflow.run(base=self.base, project=self.project, phase="planning", ready=lambda: None,
            preparer=lambda stage: None,
            runner=lambda stage: {"status": "phase_budget_exhausted" if stage == "planning200" else "phase_complete", "reason": "保留失败原因" if stage == "planning200" else None},
            checker=lambda: {"passed": True}, reporter=lambda: None)
        self.assertEqual(len(json.loads(progress.read_text(encoding="utf-8"))["steps"]), count)

    def test_unknown_only_worker_passes_explicit_prior_cache_and_other_modes_do_not(self):
        for unknown_only in (True, False):
            task = self.inputs[0]["task"]
            root = self.base / "worker_mock" / str(unknown_only)
            root.mkdir(parents=True, exist_ok=True)
            signal = root / "ready.start"
            signal.write_text("ready", encoding="utf-8")
            job = {"id": "mock-qualification", "kind": "qualify", "phase": "sim", "unknown_only": unknown_only,
                "prior_qualification": task, "task": task, "task_sha256": batch.sha256(task), "output": str(root),
                "definition_sha256": "mock", "experiment_selection_sha256": "mock"}
            spec = self.save(root / "spec.json", {"job": job, "start_signal": str(signal), "hard_deadline_epoch": time.time()+60., "result_path": str(root / "result.json")})
            with (patch.object(batch, "selection", return_value={}), patch.object(batch, "require_motion_gate"),
                  patch.object(batch, "motion_fingerprint", return_value={}),
                  patch("simbridge.qualifier.qualify_task", return_value={"status": "completed", "simulator_closed": True}) as qualify):
                self.assertEqual(batch.worker(spec), 0)
            if unknown_only:
                self.assertEqual(qualify.call_args.kwargs["prior_qualification"], task)
            else:
                self.assertNotIn("prior_qualification", qualify.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
