"""Measure fixed D geometry without starting a simulator or hiding conflicts."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

from tools.v133_support_geometry import measure


class SupportGeometryRevisionTests(unittest.TestCase):
    def test_top_grasp_shoe_clears_brick_by_at_least_five_millimetres(self):
        report = measure()
        for arm, row in report["arms"].items():
            with self.subTest(arm=arm):
                self.assertGreaterEqual(row["top_grasp"]["shoe_brick_clearance_m"], .005)
                self.assertAlmostEqual(row["sensor_in_palm"][2], .035)
                self.assertEqual(report["original_gripper"]["tcp_in_palm"], [0, 0, .071])

    def test_support_keeps_contact_and_finger_clearance_after_exact_axial_compensation(self):
        report = measure()
        for arm, row in report["arms"].items():
            with self.subTest(arm=arm):
                support = row["support"]
                self.assertAlmostEqual(support["tcp_axial_compensation_m"], .036)
                self.assertAlmostEqual(support["shoe_top_to_beam_bottom_m"], 0., places=10)
                self.assertAlmostEqual(support["shoe_top_xy_offset_from_support_m"], 0., places=10)
                self.assertGreaterEqual(support["finger_upper_edge_clearance_m"], .008)
                self.assertTrue(row["sensor_parent_is_palm"])
                self.assertTrue(row["shoe_parent_is_sensor"])
                self.assertEqual(row["shoe_in_sensor"], [0, 0, 0, 0, 0, 0, 1])

    def test_nearest_entity_face_and_tip_centre_beam_projection_meet_their_stated_limits(self):
        report = measure()
        for arm, row in report["arms"].items():
            for number, tip in enumerate(row["support"]["tips"]):
                with self.subTest(arm=arm, finger=number):
                    self.assertLessEqual(tip["nearest_tip_face_distance_to_support_m"], .080)
                    self.assertTrue(tip["centre_in_beam_width"])
                    self.assertAlmostEqual(tip["nearest_tip_face_distance_to_support_m"], .07867814181842373, places=10)

    def test_strict_centre_distance_and_whole_face_width_conflicts_are_explicit(self):
        report = measure()
        self.assertTrue(report["fixed_shift_clearance_and_contact_passed"])
        self.assertFalse(report["strict_tip_centre_80mm_passed"])
        self.assertFalse(report["whole_tip_faces_in_beam_width"])
        self.assertFalse(report["exact_geometry_contract_passed"])
        for row in report["arms"].values():
            for tip in row["support"]["tips"]:
                # Centre uses the centre of the physical +z end face, rather
                # than substituting its nearer inner/upper edge.
                self.assertAlmostEqual(tip["face_center_distance_to_support_m"], .08449408263304598, places=10)
                self.assertGreater(tip["face_center_distance_to_support_m"], .080)
                self.assertFalse(tip["centre_within_80mm"])
        geometry = report["original_gripper"]
        self.assertEqual(geometry["finger_size"], [.010, .016, .055])
        self.assertEqual(geometry["finger_centres_in_palm"], [[-.075, 0, .042], [.075, 0, .042]])

    def test_other_frames_methods_match_the_d_before_backup_ast(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root/"results/sim_v1_3/v133/d_backup/manifest.json").read_text(encoding="utf-8"))
        tree = ast.parse((root/"simbridge/frames.py").read_text(encoding="utf-8"))
        frames = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Frames")
        methods = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                   for node in frames.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        changed = [name for name, digest in manifest["frames_method_ast_sha256"].items() if methods[name] != digest]
        self.assertEqual(changed, ["support_pose"])


if __name__ == "__main__":
    unittest.main()
