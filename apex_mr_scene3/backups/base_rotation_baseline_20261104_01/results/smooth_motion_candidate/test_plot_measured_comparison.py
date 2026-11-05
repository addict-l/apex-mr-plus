"""Synthetic plotting checks; not simulated or measured experiment evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_measured_comparison import all_pose_boundaries, ordinary_boundaries, velocity_series


class VelocityPlotDataTests(unittest.TestCase):
    def setUp(self):
        self.graph = {'nodes': [
            {'id': 'p1', 'task_id': 'A_exit_from_support_first', 'arm': 'A'},
            {'id': 'p2', 'task_id': 'A_exit_from_support_second', 'arm': 'A'},
            {'id': 'other', 'task_id': 'A_exit_from_supporting_other', 'arm': 'A'}]}
        self.execution = {'events': [
            {'kind': 'node_started', 'node': 'p1', 'time': 7.},
            {'kind': 'node_completed', 'node': 'p1', 'time': 7.1},
            {'kind': 'node_started', 'node': 'p2', 'time': 7.12},
            {'kind': 'node_completed', 'node': 'p2', 'time': 7.2}],
            'trace': [
                {'time': 6.9, 'A_joint_velocity_rad_s': [9]*6},
                {'time': 7., 'A_joint_velocity_rad_s': [.01, -.04, 0, 0, 0, 0], 'A_target_velocity_rad_s': [9]*6},
                {'time': 7.1, 'A_joint_velocity_rad_s': [.02]*6},
                {'time': 7.2, 'A_joint_velocity_rad_s': [.03]*6},
                {'time': 7.3, 'A_joint_velocity_rad_s': [9]*6}]}

    def test_actual_start_alignment_and_raw_maximum_only(self):
        out = velocity_series(self.graph, self.execution, 'A_exit_from_support')
        self.assertEqual(out['actual_start_s'], 7.)
        self.assertAlmostEqual(out['actual_elapsed_s'], .2)
        self.assertEqual(out['measured_sample_count'], 3)
        self.assertEqual(out['time_from_actual_start_s'][0], 0.)
        self.assertEqual(out['max_abs_measured_joint_velocity_rad_s'], [.04, .02, .03])

    def test_target_velocity_cannot_fill_missing_measurements(self):
        for row in self.execution['trace']:
            row.pop('A_joint_velocity_rad_s')
            row['A_target_velocity_rad_s'] = [.1]*6
        with self.assertRaisesRegex(ValueError, 'No measured joint velocity'):
            velocity_series(self.graph, self.execution, 'A_exit_from_support')

    def test_missing_actual_completion_rejected(self):
        self.execution['events'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing actual start/completion'):
            velocity_series(self.graph, self.execution, 'A_exit_from_support')

    def test_gap_and_invalid_velocity_are_visible_breaks(self):
        self.execution['trace'] = [
            {'time': 7., 'A_joint_velocity_rad_s': [.1]*6},
            {'time': 7.18, 'A_joint_velocity_rad_s': [float('nan')]*6},
            {'time': 7.2, 'A_joint_velocity_rad_s': [.2]*6}]
        result = velocity_series(self.graph, self.execution, 'A_exit_from_support')
        self.assertEqual(result['max_abs_measured_joint_velocity_rad_s'], [.1, None, None, .2])
        self.assertEqual(result['missing_velocity_samples'], 1)
        self.assertAlmostEqual(result['maximum_observed_sample_gap_s'], .2)


class BoundaryPlotDataTests(unittest.TestCase):
    def setUp(self):
        self.graph = {'nodes': [{'id': str(i), 'metadata': {}} for i in range(8)],
                      'time_parameterization': {'max_velocity_rad_s': .6, 'max_acceleration_rad_s2': 1.2}}
        rows = [{'from': str(i), 'to': str(i+1), 'dependency_wait_s': 0., 'near_zero': value}
                for i, value in enumerate([True, False, True, True, False, None, True])]
        self.metrics = {'boundaries': {'boundaries': rows, 'eligible_boundaries': 7,
            'observed_boundaries': 6, 'near_zero_count': 4, 'speed_threshold_rad_s': .01,
            'sample_window_s': .06}}

    def test_all_boundaries_keep_slow_guarded_and_waiting_denominator(self):
        self.graph['nodes'][0]['metadata']['max_velocity_rad_s'] = .1
        self.graph['nodes'][2]['metadata']['assembly_force_guard'] = True
        self.metrics['boundaries']['boundaries'][4]['dependency_wait_s'] = 1.
        all_rows = all_pose_boundaries(self.metrics)
        ordinary = ordinary_boundaries(self.graph, self.metrics)
        self.assertEqual(all_rows['eligible'], 7)
        self.assertEqual(all_rows['observed'], 6)
        self.assertEqual(all_rows['near_zero'], 4)
        self.assertAlmostEqual(all_rows['near_zero_fraction'], 4/6)
        self.assertEqual(ordinary['eligible'], 3)
        self.assertEqual(ordinary['observed'], 2)
        self.assertEqual(ordinary['near_zero'], 2)

    def test_unobserved_boundaries_not_counted_as_nonzero(self):
        self.metrics['boundaries']['observed_boundaries'] = 0
        self.metrics['boundaries']['near_zero_count'] = 0
        result = all_pose_boundaries(self.metrics)
        self.assertIsNone(result['near_zero_fraction'])
        self.assertEqual(result['coverage_fraction'], 0.)


if __name__ == '__main__': unittest.main()
