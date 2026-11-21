"""Extra platform task is a declared affordance variant, not an IK fiction."""
from copy import deepcopy
import unittest

from tools.v133_platform_required_check import check_required_platform


class RequiredPlatformFixtureTests(unittest.TestCase):
    def test_only_text_and_manual_support_affordance_change(self):
        report = check_required_platform()
        self.assertTrue(report["checks"]["only_declared_differences"])
        self.assertTrue(report["checks"]["metadata_descriptions_changed"])
        self.assertTrue(report["checks"]["beam_manual_support_disabled"])
        self.assertTrue(report["checks"]["no_handwritten_reach"])

    def test_geometry_spacing_overhang_and_board_boundaries_are_preserved(self):
        report = check_required_platform()
        self.assertTrue(report["valid"], report["checks"])
        before, after = deepcopy(report["geometry"])
        before.pop("task")
        after.pop("task")
        self.assertEqual(before, after)
        self.assertGreaterEqual(after["metrics"]["minimum_reference_spacing_m"], .20-1e-9)
        self.assertAlmostEqual(after["metrics"]["overhangs"]["beam"]["length_m"], .32)
        self.assertEqual(report["existing_workcell_tool_selection"]["original_horizontal_support"], True)
        self.assertEqual(report["existing_workcell_tool_selection"]["required_horizontal_support"], False)
        self.assertFalse(report["search_performed"])
        self.assertFalse(report["ik_performed"])
        self.assertFalse(report["dynamics_performed"])

    def test_rules_require_both_platform_levels_then_final_cleanup(self):
        report = check_required_platform()
        self.assertEqual(report["grounded_beam_support_actions"], [])
        self.assertEqual(report["partial_platform_has_support"], {"tp1": False, "tp2": False})
        for check in ("complete_platform_is_support", "unanchored_beam_without_platform_rejected",
                      "unanchored_beam_with_complete_platform_valid", "both_temporary_slots_must_end_empty"):
            self.assertTrue(report["checks"][check], check)


if __name__ == "__main__":
    unittest.main()
