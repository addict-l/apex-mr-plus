"""Pure selection parsing and namespace identity, never execute a stage."""
import json
from pathlib import Path
import tempfile
import unittest

from tools import v133_experiments as selection


class ExperimentSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=selection.ROOT/"out")
        self.addCleanup(temp.cleanup)
        self.project = Path(temp.name)
        self.base = self.project/"results/sim_v1_3/v133"

    def save(self, **changes):
        value = {"selection_id": "c_scale_retry1", "stages": {"planning200": "planning200_retry1"},
            "diagnostic_trials": ["planning200"], "regressions": {
                "full": "out/v133_c_scale_retry1_full.json",
                "original11": "results/sim_v1_3/v133/original11_c_scale_retry1.json"},
            "planning_code_sha256": {"skillnet/planning/search.py": "a"*64},
            "input_cohort_sha256": "b"*64, "source_manifest_sha256": "c"*64, **changes}
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base/"experiment_selection.json").write_text(json.dumps(value), encoding="utf-8")
        return value

    def test_missing_file_keeps_identity_and_does_not_create_files(self):
        value = selection.load_selection(self.base, project=self.project)
        self.assertFalse(value["selection_present"])
        self.assertEqual(selection.resolve_stage("planning200", self.base, project=self.project), "planning200")
        self.assertFalse(self.base.exists())

    def test_selection_snapshot_and_real_source_metadata_are_explicit(self):
        self.save()
        value = selection.load_selection(self.base, project=self.project)
        self.assertTrue(value["selection_present"])
        self.assertEqual(len(value["selection_sha256"]), 64)
        self.assertEqual(selection.resolve_stage("planning200", self.base, selection=value, project=self.project), "planning200_retry1")
        self.assertEqual(selection.resolve_stage("qualify200", self.base, selection=value, project=self.project), "qualify200")
        self.save(stages={"planning200": "planning200_retry2"})
        self.assertEqual(selection.resolve_stage("planning200", self.base, selection=value, project=self.project), "planning200_retry1")

    def test_stage_paths_and_qualification_remapping_are_rejected(self):
        for stages in ({"planning200": "../old"}, {"planning200": "C:/old"}, {"qualify200": "qualify200_retry1"},
                       {"planning200": "qualify200"}, {}):
            with self.subTest(stages=stages):
                self.save(stages=stages)
                with self.assertRaises(ValueError):
                    selection.load_selection(self.base, project=self.project)

    def test_regression_paths_cannot_escape_project_and_invalid_file_never_falls_back(self):
        self.save(regressions={"full": "../outside.json"})
        with self.assertRaises(ValueError):
            selection.load_selection(self.base, project=self.project)
        (self.base/"experiment_selection.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            selection.resolve_stage("planning200", self.base, project=self.project)


if __name__ == "__main__":
    unittest.main()
