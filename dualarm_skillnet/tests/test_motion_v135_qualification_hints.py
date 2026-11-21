"""Import point-IK observations as hints, never trajectory certificates."""
from copy import deepcopy
import math
import unittest

from skillnet.capability.oracle import Status
from skillnet.core.logic import parse_fact
from tests.test_coppelia_backend import backend_fixture


class QualificationHintImportTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.geometry, self.backend = backend_fixture()

    def record(self, *, pose_mode="catalog"):
        pose = self.backend.frames.grasp_pose("red", "stock_l1", "A")
        result = {"fact": "reach(A, stock_l1)", "status": "VALID",
            "target_contexts": [{"block": "red", "type": self.backend.frames.blocks["red"]["type"],
                "place": "stock_l1", "status": "VALID", "stages": [
                    {"label": "grasp", "pose": pose,
                     "q": [.1, .2, .3, .4, .5, .6], "status": "VALID"},
                    {"label": "approach", "pose": self.backend.frames.approach_pose(pose),
                     "q": [.1, .2, .4, .4, .5, .6], "status": "VALID"}]}]}
        if pose_mode is not None:
            result["pose_mode"] = pose_mode
        return result

    def test_old_wrapped_log_without_pose_mode_preserves_point_observations(self):
        record = self.record(pose_mode=None)
        progress = {"records": [{"fact": record["fact"], "arm": "A", "place": "stock_l1",
                                 "backend_record": record}]}
        receipt = self.backend.import_qualification_hints(progress, source="qualification/F1/000/qualify_log.json")
        self.assertEqual(receipt["imported"], 2)
        self.assertEqual(len(self.backend.qualification_hints), 2)
        for hint in self.backend.qualification_hints:
            self.assertEqual(hint["arm"], "A")
            self.assertEqual(hint["part"], "red")
            self.assertEqual(hint["block_type"], self.backend.frames.blocks["red"]["type"])
            self.assertEqual(hint["place"], "stock_l1")
            self.assertEqual(hint["pose_mode"], "catalog")
            self.assertEqual(hint["grasp_strategy"], "catalog_top")
            self.assertTrue(hint["legacy_pose_mode_assumed"])
            self.assertIn("point_ik", hint["qualification_scope"])
            self.assertEqual(hint["source"], "qualification/F1/000/qualify_log.json")
            self.assertIn(hint["stage"], {"grasp", "approach"})
        self.assertFalse(self.geometry.calls)

    def test_current_backend_record_list_is_imported_and_updates_main_compiler(self):
        receipt = self.backend.import_qualification_hints([self.record()], source="current")
        self.assertEqual(receipt["imported"], 2)
        self.assertEqual(self.backend.compiler.qualification_hints, self.backend.qualification_hints)
        self.assertFalse(self.geometry.calls)

    def test_explicit_side_pitch_probes_are_skipped_even_for_sixty_degrees(self):
        receipt = self.backend.import_qualification_hints(
            [self.record(pose_mode="side_30"), self.record(pose_mode="side_45"),
             self.record(pose_mode="side_60")])
        self.assertEqual(receipt["imported"], 0)
        self.assertGreater(receipt["skipped"], 0)
        self.assertFalse(self.backend.qualification_hints)
        self.assertFalse(self.geometry.calls)

    def test_only_valid_stages_with_finite_six_joint_and_seven_pose_values_import(self):
        record = self.record()
        valid = record["target_contexts"][0]["stages"][0]
        rows = [deepcopy(valid) for _ in range(6)]
        rows[1]["status"] = "UNKNOWN"
        rows[2]["q"][0] = math.nan
        rows[3]["pose"][0] = math.inf
        rows[4]["q"] = [0.]*5
        rows[5]["pose"] = [0.]*6
        record["target_contexts"][0]["stages"] = rows
        receipt = self.backend.import_qualification_hints([record])
        self.assertEqual(receipt["imported"], 1)
        self.assertEqual(len(self.backend.qualification_hints), 1)
        self.assertFalse(self.geometry.calls)

    def test_reimport_deduplicates_and_does_not_alias_source_arrays(self):
        record = self.record()
        self.backend.import_qualification_hints([record], source="first")
        expected = deepcopy(self.backend.qualification_hints)
        receipt = self.backend.import_qualification_hints([record], source="second")
        self.assertEqual(receipt["imported"], 0)
        self.assertEqual(receipt["duplicates"], 2)
        record["target_contexts"][0]["stages"][0]["q"][0] += 1.
        self.assertEqual(self.backend.qualification_hints, expected)

    def test_unknown_part_type_arm_or_place_does_not_enter_candidate_pool(self):
        rows = [self.record() for _ in range(4)]
        rows[0]["target_contexts"][0]["block"] = "not_this_task"
        rows[1]["target_contexts"][0]["type"] = "not_this_block_type"
        rows[2]["fact"] = "reach(C, stock_l1)"
        rows[3]["fact"] = "reach(A, missing_place)"
        rows[3]["target_contexts"][0]["place"] = "missing_place"
        receipt = self.backend.import_qualification_hints(rows)
        self.assertEqual(receipt["imported"], 0)
        self.assertFalse(self.backend.qualification_hints)
        self.assertFalse(self.geometry.calls)

    def test_partial_qualification_keeps_valid_point_only_without_claiming_action_validity(self):
        record = self.record()
        record["status"] = "UNKNOWN"
        record["target_contexts"][0]["status"] = "UNKNOWN"
        record["target_contexts"][0]["stages"][1]["status"] = "UNKNOWN"
        receipt = self.backend.import_qualification_hints([record])
        self.assertEqual(receipt["imported"], 1)
        hint = self.backend.qualification_hints[0]
        self.assertIn("point_ik", hint["qualification_scope"])
        self.assertNotIn("trajectory_certificate", hint)

    def test_handover_task_labels_catalog_observations_with_existing_side_strategy(self):
        engine, geometry, backend = backend_fixture("s1_reach_handover")
        part = next(iter(backend.frames.blocks))
        place = backend.frames.blocks[part].get("at")
        pose = backend.frames.grasp_pose(part, place, "B")
        record = {"fact": f"reach(B, {place})", "pose_mode": "catalog", "status": "VALID",
            "target_contexts": [{"block": part, "type": backend.frames.blocks[part]["type"],
                "place": place, "stages": [{"label": "grasp", "status": "VALID",
                                            "pose": pose, "q": [0.]*6}]}]}
        receipt = backend.import_qualification_hints([record])
        self.assertEqual(receipt["imported"], 1)
        self.assertEqual(backend.qualification_hints[0]["grasp_strategy"], "side_opposed")
        self.assertFalse(geometry.calls)

    def test_successful_live_qualification_registers_points_for_subsequent_compilation(self):
        self.assertEqual(self.backend.qualify(parse_fact("reach(A, stock_l1)")), Status.VALID)
        self.assertEqual(len(self.backend.qualification_hints), 2)
        self.assertEqual(self.backend.compiler.qualification_hints, self.backend.qualification_hints)
        self.assertTrue(all(hint["pose_mode"] == "catalog" for hint in self.backend.qualification_hints))
        self.assertTrue(all("point_ik" in hint["qualification_scope"] for hint in self.backend.qualification_hints))

    def test_missing_qualification_observations_leave_the_normal_solver_available(self):
        receipt = self.backend.import_qualification_hints({}, source="no-record")
        self.assertEqual(receipt["imported"], 0)
        self.assertFalse(self.backend.qualification_hints)
        self.assertFalse(self.geometry.calls)

    def test_importing_a_new_hint_reopens_the_same_geometry_candidate_strategy(self):
        self.geometry.ik_failure = {"ok": False, "solutions": []}
        action = self.engine.parse_action("pick(A, red, stock_l1)")
        self.backend.begin_reference(1)
        self.assertEqual(self.backend.validate(action, self.engine.initial).status, Status.UNKNOWN)
        first = deepcopy(self.backend.validation_context)
        calls = len(self.geometry.calls)
        self.backend.import_qualification_hints([self.record()], source="new-observation")
        self.backend.begin_reference(2)
        self.assertEqual(self.backend.validate(action, self.engine.initial).status, Status.UNKNOWN)
        second = self.backend.validation_context
        self.assertEqual(second["context_id"], first["context_id"])
        self.assertNotEqual(second["candidate_id"], first["candidate_id"])
        self.assertNotEqual(second["candidate_version"], first["candidate_version"])
        self.assertFalse(second["deduplicated"])
        self.assertGreater(len(self.geometry.calls), calls)


if __name__ == "__main__":
    unittest.main()
