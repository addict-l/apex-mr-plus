"""Reject the demo's observed out-and-back rotation without banning a wrist flip."""
import math
import unittest

from recovery.path_quality import compact_route, path_metrics


def pose(base=0., wrist=0.):
    return [math.radians(base), 0., 0., 0., 0., math.radians(wrist)]


class DemoPathQualityTests(unittest.TestCase):
    def test_equivalent_end_angles_do_not_hide_a_full_base_turn(self):
        route = [pose(), pose(360), pose(0)]
        metrics = path_metrics(route)
        self.assertAlmostEqual(metrics["joint_travel_deg"][0], 720.)
        self.assertAlmostEqual(metrics["joint_excess_deg"][0], 720.)
        self.assertFalse(compact_route(route))

    def test_small_endpoint_change_does_not_hide_wrist_backtracking(self):
        route = [pose(wrist=10), pose(wrist=-239), pose(wrist=28)]
        self.assertFalse(compact_route(route))

    def test_required_wrist_flip_is_allowed(self):
        route = [pose(-60, 0), pose(-90, 90), pose(-120, 180)]
        self.assertTrue(compact_route(route))
        self.assertAlmostEqual(path_metrics(route)["joint_excess_deg"][5], 0.)

    def test_invalid_or_nonfinite_poses_are_rejected(self):
        for route in ([], [[0] * 5], [pose(), [float("nan")] * 6]):
            with self.subTest(route=route), self.assertRaises(ValueError):
                path_metrics(route)


if __name__ == "__main__":
    unittest.main()
