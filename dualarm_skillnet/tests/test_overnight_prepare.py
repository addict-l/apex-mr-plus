"""Manifest preparation with tiny generated doubles; never run search or sim."""
import csv
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from tools import overnight_batch as batch
from tools import overnight_prepare as prepare


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prepare_test_", dir=batch.ROOT / "out")
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.overnight = self.directory / "overnight"
        self.patch = patch.object(prepare, "OVERNIGHT", self.overnight)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = prepare._default_manifest()
        self.manifest["report_dir"] = prepare.relative(self.directory / "reports")
        batch.atomic_json(self.manifest_path, self.manifest)

    def task(self, name, directory=None):
        directory = directory or self.directory / "tasks"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + ".toml")
        path.write_text('name = "' + name + '"\n', encoding="utf-8")
        return path

    def planning_results(self, per_family=6, *, csv_only=False, omit_last=False):
        jobs = []
        for family in prepare.FAMILIES:
            count = per_family.get(family, 0) if isinstance(per_family, dict) else per_family
            for index in range(count):
                path = self.task(f"{family.lower()}_{index}")
                output = self.overnight / "planning" / path.stem
                job = prepare._planning_job(f"plan-{family.lower()}-{index:03d}", family,
                    path, output, "planning_200")
                output.mkdir(parents=True, exist_ok=True)
                batch.atomic_json(output / "plan.json", {"status": "solved", "used_proposal": False,
                    "proposal": None, "batch_independent_planning": {"proposal_enabled": False, "witness_loaded": False}})
                jobs.append(job)
        self.manifest["jobs"] = jobs
        prepare._save_manifest(self.manifest_path, self.manifest)
        normalized = batch.load_manifest(self.manifest_path)
        journal = batch.Journal(normalized["state_dir"])
        rows = []
        for job in normalized["jobs"][:-1] if omit_last else normalized["jobs"]:
            row = batch._completion(job, {"status": "solved", "passed": True, "makespan": 5})
            rows.append(row)
            if not csv_only:
                journal.append(row)
        if csv_only:
            destination = batch.confined(self.manifest["report_dir"]) / "planning_200.csv"
            batch.atomic_text(destination, batch._csv_text(rows, batch.REPORT_COLUMNS))
        return normalized

    def complete_first_selection(self, fail_first=False, expired=False):
        normalized = batch.load_manifest(self.manifest_path)
        journal = batch.Journal(normalized["state_dir"])
        failed_task = None
        for job in normalized["jobs"]:
            if job["phase"] != "sim" or job["selection"] != 1:
                continue
            if job.get("skip_reason"):
                result = {"status": "skipped", "passed": False, "executed": False}
            elif fail_first and failed_task is None:
                result = {"status": "failed", "passed": False, "failure_phase": "execution"}
                failed_task = job["task"]
            else:
                result = {"status": "passed", "passed": True}
            journal.append(batch._completion(job, result))
        state = batch._state(normalized)
        state["session_started_epoch"] = time.time()
        state["phase_started_epoch"]["sim"] = time.time() - (18001 if expired else 10)
        batch.save_state(normalized, state)
        return failed_task

    def test_plan_prepares_13_then_200_and_resumes_without_overwriting_old_samples(self):
        old = self.directory / "old"
        for family in prepare.FAMILIES:
            for index in range(2):
                task = self.task(f"gen_{family.lower()}_s20261115_{index:03d}", old)
                batch.atomic_json(task.with_suffix(".witness.json"), {"serial_witness_estimate_seconds": 8})
        before = {path.name: path.read_bytes() for path in old.iterdir()}
        fixtures = [self.task(name) for name in ("s2_tower_sim", "s3a_cantilever_sim", "s3a_platform_sim")]
        witness_file = self.directory / "fixture_witnesses.json"
        batch.atomic_json(witness_file, {"reports": [{"task": task.stem, "serial_witness_estimate_seconds": 10}
                                                     for task in fixtures]})
        new = self.directory / "new"
        def fake_generate(family, seed, index, out):
            task = self.task(f"gen_{family.lower()}_s{seed}_{index:03d}", out)
            return {"path": str(task), "layout": {"valid": True}, "witness": {
                "rules_replayed": True, "goal_satisfied": True, "serial_witness_estimate_seconds": 9}}
        generator = Mock(side_effect=fake_generate)
        first = prepare.prepare_plan(self.manifest_path, generator=generator, generated=new,
            old_generated=old, fixtures=fixtures, fixture_witnesses=witness_file)
        self.assertEqual(len(first["jobs"]), 213)
        self.assertTrue(all(job["cohort"] == "precheck" for job in first["jobs"][:13]))
        self.assertTrue(all(job["cohort"] == "planning_200" for job in first["jobs"][13:]))
        self.assertEqual(generator.call_count, 200)
        self.assertTrue(all(call.args[1] == 20261115 for call in generator.call_args_list))
        second = prepare.prepare_plan(self.manifest_path, generator=generator, generated=new,
            old_generated=old, fixtures=fixtures, fixture_witnesses=witness_file)
        self.assertEqual(second["jobs"], first["jobs"])
        self.assertEqual(generator.call_count, 200)
        self.assertEqual(before, {path.name: path.read_bytes() for path in old.iterdir()})

    def test_new_generation_cannot_target_old_samples(self):
        old = self.directory / "old"
        with self.assertRaises(ValueError):
            prepare.prepare_plan(self.manifest_path, generator=Mock(), generated=old, old_generated=old)

    def test_selection_is_deterministic_and_same_ids_never_change(self):
        self.planning_results()
        first = prepare.prepare_selection(1, 713, self.manifest_path)
        before = self.manifest_path.read_bytes()
        second = prepare.prepare_selection(1, 713, self.manifest_path)
        jobs = [job for job in first["jobs"] if job["phase"] == "sim"]
        self.assertEqual(len(jobs), 20)
        self.assertTrue(all(job["source_plan_status"] == "solved" for job in jobs))
        self.assertEqual(before, self.manifest_path.read_bytes())
        self.assertEqual(jobs, [job for job in second["jobs"] if job["phase"] == "sim"])
        with self.assertRaises(ValueError):
            prepare.prepare_selection(1, 714, self.manifest_path)

    def test_unfinished_independent_planning_cannot_freeze_early_selection(self):
        self.planning_results(omit_last=True)
        before = self.manifest_path.read_bytes()
        with self.assertRaises(ValueError):
            prepare.prepare_selection(1, 713, self.manifest_path)
        self.assertEqual(before, self.manifest_path.read_bytes())

    def test_missing_solved_samples_become_explicit_unexecuted_slots(self):
        self.planning_results(per_family=2)
        manifest = prepare.prepare_selection(1, 713, self.manifest_path)
        sim = [job for job in manifest["jobs"] if job["phase"] == "sim"]
        self.assertEqual(len(sim), 20)
        self.assertEqual(sum(bool(job.get("task")) for job in sim), 10)
        self.assertEqual(sum(bool(job.get("skip_reason")) for job in sim), 10)
        self.assertTrue(all("不以见证" in job["skip_reason"] for job in sim if job.get("skip_reason")))

    def test_budget_or_witness_only_artifact_never_qualifies_as_solved_sample(self):
        normalized = self.planning_results(per_family=1)
        first = normalized["jobs"][0]
        batch.atomic_json(Path(first["output"]) / "plan.json", {"status": "solved", "proposal": None,
            "batch_independent_planning": {"proposal_enabled": False, "witness_loaded": True}})
        manifest = prepare.prepare_selection(1, 713, self.manifest_path)
        self.assertTrue(all(job.get("skip_reason") for job in manifest["jobs"]
                            if job["phase"] == "sim" and job["family"] == "F1"))
        rejected = manifest["preparation"]["selections"]["1"]["rejected_candidates"]
        self.assertEqual(rejected[0]["id"], first["id"])

    def test_csv_fallback_requires_and_uses_actual_independent_plan_artifacts(self):
        self.planning_results(per_family=1, csv_only=True)
        manifest = prepare.prepare_selection(1, 713, self.manifest_path)
        actual = [job for job in manifest["jobs"] if job["phase"] == "sim" and job.get("task")]
        self.assertEqual(len(actual), 5)
        self.assertTrue(all(job["source_record"] == "csv" for job in actual))

    def test_second_selection_uses_new_then_successful_overlap_and_never_retries_failure(self):
        self.planning_results(per_family=6)
        first = prepare.prepare_selection(1, 713, self.manifest_path)
        previous = [dict(job) for job in first["jobs"] if job["phase"] == "sim"]
        failed_task = self.complete_first_selection(fail_first=True)
        second = prepare.prepare_selection(2, 714, self.manifest_path)
        self.assertEqual(previous, [job for job in second["jobs"] if job["phase"] == "sim" and job["selection"] == 1])
        selected = [job for job in second["jobs"] if job["phase"] == "sim" and job["selection"] == 2]
        self.assertEqual(len(selected), 20)
        self.assertNotIn(prepare.relative(failed_task), {job.get("task") for job in selected})
        for family in prepare.FAMILIES:
            entries = [job for job in selected if job["family"] == family]
            self.assertEqual(sum(job["overlap_with_selection_1"] for job in entries), 2)

    def test_second_selection_needs_first_results_different_seed_and_unexpired_clock(self):
        self.planning_results()
        prepare.prepare_selection(1, 713, self.manifest_path)
        with self.assertRaises(ValueError):
            prepare.prepare_selection(2, 714, self.manifest_path)
        self.complete_first_selection(expired=True)
        with self.assertRaises(ValueError):
            prepare.prepare_selection(2, 713, self.manifest_path)
        with self.assertRaises(ValueError):
            prepare.prepare_selection(2, 714, self.manifest_path)

    def test_gates_append_exact_five_names_and_are_idempotent(self):
        manifest = prepare.prepare_gates(self.manifest_path)
        self.assertEqual([job["test_name"] for job in manifest["jobs"]], list(batch.LIVE_TESTS))
        before = self.manifest_path.read_bytes()
        prepare.prepare_gates(self.manifest_path)
        self.assertEqual(before, self.manifest_path.read_bytes())

    def test_precheck_witness_comparison_is_posthoc_and_preserves_budget_status(self):
        task = self.task("precheck")
        job = prepare._planning_job("precheck", "fixture", task, self.overnight / "check", "precheck",
            witness=10, witness_source=task.with_suffix(".witness.json"))
        self.manifest["jobs"] = [job]
        prepare._save_manifest(self.manifest_path, self.manifest)
        normalized = batch.load_manifest(self.manifest_path)
        batch.Journal(normalized["state_dir"]).append(batch._completion(normalized["jobs"][0],
            {"status": "budget", "passed": False, "makespan": 0, "reason": "default search budget"}))
        rows = prepare.rebuild_precheck_comparison(self.manifest_path)
        self.assertEqual(rows[0]["status"], "budget")
        self.assertIsNone(rows[0]["independent_makespan_seconds"])
        self.assertIsNone(rows[0]["independent_to_witness_ratio"])
        report = batch.confined(self.manifest["report_dir"]) / "precheck_comparison.csv"
        with report.open(encoding="utf-8", newline="") as stream:
            actual = list(csv.DictReader(stream))
        self.assertEqual(actual[0]["witness_estimate_seconds"], "10")
        self.assertEqual(actual[0]["reason"], "default search budget")


if __name__ == "__main__":
    unittest.main()
