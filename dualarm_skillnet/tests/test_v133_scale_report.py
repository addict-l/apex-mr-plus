"""Committed synthetic planning evidence only; never invoke a planner."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools import v133_scale_report as report


class ScaleReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="v133_scale_mock_", dir=report.ROOT/"out")
        self.addCleanup(temp.cleanup)
        self.project = Path(temp.name)
        self.base = self.project/"results/sim_v1_3/v133"
        self.jobs, self.records = [], []

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def case(self, family, blocks, index=0, *, status="solved", elapsed=1., table="qualified", committed=True):
        name = f"task_{family.lower()}_{index:03d}"
        task = self.base/"qualification"/family/f"{index:03d}"/(name+".qualified.toml")
        task.parent.mkdir(parents=True, exist_ok=True)
        task.write_text(f'name="{name}"\ncapability_source="ik_qualified"\n'
            +'\n'.join(f'[[block]]\nid="b{i}"' for i in range(blocks))
            +'\n[capabilities.reach]\ndefault="UNKNOWN"\n', encoding="utf-8")
        digest = hashlib.sha256(task.read_bytes()).hexdigest()
        output = self.base/"planning"/family/f"{index:03d}"/table
        job = {"id": f"p-{table}-{family.lower()}-{index:03d}", "kind": "plan", "phase": "plan",
               "table": table, "family": family, "index": index, "task": str(task),
               "output": str(output), "prepared_task_sha256": digest}
        proof = {"task": name, "status": status, "passed": status == "solved", "elapsed_seconds": elapsed,
                 "bound": 1.5 if status == "solved" else None,
                 "proposal_enabled": False, "witness_loaded": False}
        self.save(output/"plan.json", proof)
        row = {"id": job["id"], "task_sha256": digest, **proof}
        self.jobs.append(job)
        if committed:
            self.records.append(row)
        return job, row, output/"plan.json"

    def collect(self):
        self.save(self.base/"manifests/planning200.json", {"jobs": self.jobs})
        journal = self.base/"states/planning200/completed.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text("".join(json.dumps(r)+"\n" for r in self.records), encoding="utf-8")
        return report.collect(self.base, project=self.project)

    def test_size_groups_use_exact_inclusive_boundaries(self):
        for family, blocks, group in (("F1",4,None),("F1",5,"F1_5_6"),("F1",6,"F1_5_6"),("F1",7,None),
            ("F3",10,"F3_up_to_10"),("F3",11,None),("F3",12,"F3_12_15"),("F3",15,"F3_12_15"),
            ("F3",16,None),("F5",6,None),("F5",7,"F5_7"),("F5",8,None)):
            self.assertEqual(report.scale_group(family, blocks), group, (family, blocks))

    def test_no_batch_results_are_pending_and_rebuild_is_report_only(self):
        metrics = report.rebuild(self.base, project=self.project)
        self.assertEqual(metrics["status"], "pending")
        self.assertFalse(metrics["passed"])
        self.assertTrue((self.base/"scale_acceptance.json").is_file())
        self.assertTrue((self.base/"scale_acceptance.md").is_file())

    def test_all_required_cases_pass_while_large_f3_budget_is_reported_honestly(self):
        for family in ("F1", "F3", "F5"):
            for index in range(40):
                blocks = (5+index%2 if family == "F1" else 7 if family == "F5" else 15 if index == 39 else 9)
                self.case(family, blocks, index, status="budget" if family == "F3" and index == 39 else "solved")
        metrics = self.collect()
        self.assertEqual(metrics["status"], "passed")
        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["groups"]["F1_5_6"]["represented_block_counts"], [5, 6])
        self.assertEqual(metrics["groups"]["F3_up_to_10"]["requested"], 39)
        self.assertEqual(metrics["large_f3"]["status_counts"], {"budget": 1})
        self.assertEqual(metrics["large_f3"]["cases"][0]["blocks"], 15)
        self.assertIsNone(metrics["large_f3"]["cases"][0]["bound"])
        self.assertTrue(metrics["large_f3"]["evidence_complete"])
        rendered = report.markdown(metrics)
        self.assertIn("[plan.json](planning/F3/039/qualified/plan.json)", rendered)
        self.records = [row for row in self.records if row["id"] != "p-qualified-f3-039"]
        missing = self.collect()
        self.assertEqual(missing["status"], "pending")
        self.assertFalse(missing["passed"])
        self.assertFalse(missing["large_f3"]["evidence_complete"])

    def test_elapsed_limit_and_actual_input_block_count_cannot_be_overridden(self):
        job, row, _ = self.case("F1", 5, elapsed=60.001)
        job["blocks"] = row["blocks"] = 2
        metrics = self.collect()
        case = metrics["groups"]["F1_5_6"]["cases"][0]
        self.assertEqual(case["blocks"], 5)
        self.assertEqual(case["acceptance_status"], "not_met")
        self.assertIn("elapsed_exceeds_60s", case["reasons"])

    def test_raw_success_or_orphan_old_sample_cannot_fill_qualified_pending(self):
        self.case("F1", 5, committed=False)
        self.case("F1", 5, table="raw")
        self.records.append({"id": "old-f1-000", "status": "solved", "passed": True, "elapsed_seconds": 1.})
        metrics = self.collect()
        group = metrics["groups"]["F1_5_6"]
        self.assertEqual(group["solved_within_60s"], 0)
        self.assertEqual(group["cases"][0]["acceptance_status"], "pending")
        self.assertFalse(metrics["passed"])
        self.assertIn("old-f1-000", metrics["orphan_record_ids"])

    def test_plan_file_must_explicitly_disable_proposal_and_witness(self):
        _, _, path = self.case("F5", 7)
        for flags in ({"proposal_enabled": True, "witness_loaded": False},
                      {"proposal_enabled": False, "witness_loaded": True}, {}):
            proof = json.loads(path.read_text(encoding="utf-8"))
            proof.pop("proposal_enabled", None);proof.pop("witness_loaded", None)
            self.save(path, {**proof, **flags})
            case = self.collect()["groups"]["F5_7"]["cases"][0]
            self.assertEqual(case["acceptance_status"], "not_met")
            self.assertIn("independent_search_flags_missing_or_enabled", case["reasons"])

    def test_missing_plan_bound_or_input_hash_is_missing_evidence_not_a_pass(self):
        job, row, path = self.case("F3", 9)
        proof = json.loads(path.read_text(encoding="utf-8"));proof.pop("bound")
        self.save(path, proof)
        case = self.collect()["groups"]["F3_up_to_10"]["cases"][0]
        self.assertIn("bound_not_recorded", case["reasons"])
        path.unlink()
        case = self.collect()["groups"]["F3_up_to_10"]["cases"][0]
        self.assertIn("plan_artifact_missing_or_invalid", case["reasons"])
        row["task_sha256"] = "0"*64
        case = self.collect()["groups"]["F3_up_to_10"]["cases"][0]
        self.assertIn("committed_input_sha_mismatch_or_missing", case["reasons"])

    def test_distance_prior_input_and_old_output_are_not_qualified_evidence(self):
        job, _, _ = self.case("F1", 6)
        task = Path(job["task"])
        task.write_text(task.read_text(encoding="utf-8").replace("ik_qualified", "distance_prior"), encoding="utf-8")
        case = self.collect()["groups"]["F1_5_6"]["cases"][0]
        self.assertIn("input_not_ik_qualified", case["reasons"])
        job["output"] = str(self.project/"old_v132/planning")
        self.save(Path(job["output"])/"plan.json", {"status":"solved", "passed":True,
            "elapsed_seconds":1.,"bound":1.,"proposal_enabled":False,"witness_loaded":False})
        case = self.collect()["groups"]["F1_5_6"]["cases"][0]
        self.assertIn("output_outside_current_v133", case["reasons"])

    def test_uncommitted_tail_is_ignored_and_conflicting_records_are_not_adopted(self):
        job, row, _ = self.case("F5", 7, committed=False)
        self.collect()
        journal = self.base/"states/planning200/completed.jsonl"
        journal.write_text(json.dumps(row), encoding="utf-8")
        case = report.collect(self.base, project=self.project)["groups"]["F5_7"]["cases"][0]
        self.assertEqual(case["acceptance_status"], "pending")
        self.records = [row, {**row, "status": "budget", "passed": False}]
        metrics = self.collect()
        self.assertFalse(metrics["passed"])
        self.assertIn(job["id"], metrics["conflicting_record_ids"])

    def test_selected_new_manifest_and_journal_do_not_fall_back_to_old_trial(self):
        self.case("F1", 5)
        self.collect()
        self.save(self.base/"experiment_selection.json", {"selection_id": "c_scale_retry1",
            "stages": {"planning200": "planning200_retry1"}, "diagnostic_trials": ["planning200"]})
        metrics = report.collect(self.base, project=self.project)
        self.assertEqual(metrics["status"], "pending")
        self.assertEqual(metrics["selected_stage"], "planning200_retry1")
        self.assertIn("manifests/planning200_retry1.json", metrics["manifest"])
        self.assertIn("states/planning200_retry1/", metrics["journal"])
        self.assertEqual(metrics["groups"]["F1_5_6"]["solved_within_60s"], 0)


if __name__ == "__main__":
    unittest.main()
