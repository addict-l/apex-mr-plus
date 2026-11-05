"""Synthetic wrap/long-turn distinction, not simulation evidence."""
import math
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_base_rotation import analyze, measured_window, compare_windows, GROUPS


def sample(t, q_degrees, velocity):
    return {'time': t, 'qB': [math.radians(q_degrees), 0, 0, 0, 0, 0],
            'B_joint_velocity_rad_s': [velocity, 0, 0, 0, 0, 0]}


class BaseRotationTests(unittest.TestCase):
    def test_sensor_wrap_is_two_degrees_not_358(self):
        v = math.radians(2)/.1
        report = measured_window([sample(0, 179, v), sample(.1, -179, v)], 'B', 0, .1)
        self.assertEqual(report['raw_jump_count'], 1)
        self.assertAlmostEqual(report['total_variation_deg'], 2)
        self.assertTrue(report['unwrap_inference_reliable'])
        self.assertTrue(report['velocity_integral_agrees_with_position'])

    def test_true_long_rotation_preserved_across_dense_samples(self):
        v = math.radians(237)/20
        rows = [sample(i*.1, -78+237*i/200, v) for i in range(201)]
        report = measured_window(rows, 'B', 0, 20)
        self.assertEqual(report['raw_jump_count'], 0)
        self.assertAlmostEqual(report['total_variation_deg'], 237)
        self.assertAlmostEqual(report['shortest_endpoint_circular_arc_deg'], 123)

    def test_planned_winding_is_not_replaced_with_shortest_circular_arc(self):
        task = {'id': 'pick_support_approach', 'arm': 'B', 'kind': 'transit',
                'q_path': [[math.radians(x), 0, 0, 0, 0, 0] for x in [-78, 0, 159]]}
        report = analyze([task])
        self.assertEqual(report['red_tasks'], ['pick_support_approach'])
        self.assertAlmostEqual(report['focus_tasks'][0]['planned_reference']['total_variation_deg'], 237)

    def test_velocity_disagreement_is_visible_without_replacing_positions(self):
        rows = [sample(i*.1, 1.5*i, math.radians(10)) for i in range(21)]
        report = measured_window(rows, 'B', 0, 2)
        self.assertAlmostEqual(report['total_variation_deg'], 30)
        self.assertFalse(report['velocity_integral_agrees_with_position'])

    def test_splitting_or_relocating_angle_without_reducing_window_fails(self):
        def report(total, maximum):
            return {'scope': 'planned_candidate_only', 'limit_deg': 180., 'non_relocation_windows': {
                label: {'first_task': 'first', 'last_task': 'last', 'planned_total_variation_deg': total,
                        'max_planned_transit_variation_deg': maximum} for label in GROUPS}}
        baseline = report(250., 237.)
        split_only = compare_windows(report(250., 150.), baseline)
        self.assertFalse(split_only['passed'])
        self.assertTrue(all(r['all_candidate_transits_within_limit'] for r in split_only['windows']))
        self.assertTrue(compare_windows(report(170., 130.), baseline)['passed'])


if __name__ == '__main__': unittest.main()
