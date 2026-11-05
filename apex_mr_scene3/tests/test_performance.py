"""Synthetic measurement regressions, never performance evidence for Scene3."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from apex.performance import boundary_metrics, compare_runs, summarize_run, velocity_metrics
from test_apex_tpg import required_physics_audit_fixture


def complete_fixture():
    graph, execution, req = required_physics_audit_fixture()
    graph['validation_requirements'] = req
    graph['time_parameterization'] = {'max_velocity_rad_s': .6, 'max_acceleration_rad_s2': 1.2}
    graph['scene_signature'] = 'code-inclusive-signature-1'
    graph['reference_signature'] = 'reference-signature'
    graph['certificate'].update(dag=True, motion_overlap_pairs_checked=True, collision_oracle_supplied=True,
        all_nodes_have_geometry=True, waiting_endpoints_verified=True, nominal_world_state_verified=True)
    execution.update(makespan=9., peak_velocity_rad_s=0., run_id='synthetic',
                     scene_signature=graph['scene_signature'], reference_signature=graph['reference_signature'])
    final = deepcopy(execution['trace'][-1])
    execution['trace'].extend({**deepcopy(final), 'time': i/10} for i in range(51, 90))
    execution['trace'].sort(key=lambda r: r['time'])
    for row in execution['trace']:
        row['A_joint_velocity_rad_s'] = [0.]*6
        row['B_joint_velocity_rad_s'] = [0.]*6
    manifest = {'input_fingerprint': {'files': {'scenes/10_Scene3_Dynamic.ttt': 'same-physical-source'}},
                'environment': {'coppeliasim_program_version': 1, 'packages': {}}}
    return graph, execution, manifest


def boundary_fixture():
    nodes = [dict(id=key, task_id='move', arm='A', node_type=kind, duration=end-start,
                  metadata={}, q_samples=[[start]*6, [end]*6])
             for key, kind, start, end in [('p0', 'pose', 0., 1.), ('p1', 'pose', 1.02, 2.),
                                           ('skill', 'skill', 2.02, 2.4), ('p2', 'pose', 2.42, 3.)]]
    graph = {'nodes': nodes, 'edges': [{'source': a['id'], 'target': b['id']} for a, b in zip(nodes, nodes[1:])]}
    events = []
    for key, start, end in [('p0', 0., 1.), ('p1', 1.02, 2.), ('skill', 2.02, 2.4), ('p2', 2.42, 3.)]:
        events += [{'kind': 'node_started', 'node': key, 'time': start},
                   {'kind': 'node_completed', 'node': key, 'time': end}]
    execution = {'makespan': 3., 'events': events,
                 'trace': [{'time': 1., 'A_joint_velocity_rad_s': [.005]*6}]}
    return graph, execution


class PerformanceTests(unittest.TestCase):
    def test_nonuniform_velocity_derivatives_use_actual_interval_midpoints(self):
        times = [0., .02, .09, .20]
        trace = [{'time': t, 'A_joint_velocity_rad_s': [t*t, 0., 0., 0., 0., 0.]} for t in times]
        result = velocity_metrics(trace, 'A')
        self.assertEqual(result['acceleration_estimate_count'], 3)
        self.assertEqual(result['jerk_estimate_count'], 2)
        self.assertAlmostEqual(result['estimated_max_abs_acceleration_rad_s2'], .29)
        self.assertAlmostEqual(result['estimated_max_abs_jerk_rad_s3'], 2.)
        self.assertAlmostEqual(result['sample_intervals_s']['maximum'], .11)

    def test_missing_samples_and_long_gaps_break_derivative_chain(self):
        trace = [{'time': 0., 'A_joint_velocity_rad_s': [0.]*6},
                 {'time': .02, 'A_joint_velocity_rad_s': [.02]*6},
                 {'time': .03},
                 {'time': .04, 'A_joint_velocity_rad_s': [1.]*6},
                 {'time': .8, 'A_joint_velocity_rad_s': [2.]*6}]
        result = velocity_metrics(trace, 'A')
        self.assertEqual(result['invalid_samples'], 1)
        self.assertEqual(result['excluded_long_gap_count'], 1)
        self.assertEqual(result['acceleration_estimate_count'], 1)
        self.assertEqual(result['jerk_estimate_count'], 0)

    def test_only_adjacent_pose_boundaries_count_and_missing_observation_is_unknown(self):
        graph, execution = boundary_fixture()
        graph['nodes'][0]['metadata']['performance_boundary'] = {'kind': 'pass_through', 'reason': 'collinear'}
        result = boundary_metrics(graph, execution)
        self.assertEqual(result['eligible_boundaries'], 1)
        self.assertEqual(result['near_zero_count'], 1)
        self.assertEqual(result['classification_counts'], {'pass_through': 1})
        execution['trace'][0]['time'] = .5
        missing = boundary_metrics(graph, execution)
        self.assertEqual(missing['observed_boundaries'], 0)
        self.assertEqual(missing['coverage_fraction'], 0.)
        self.assertIsNone(missing['boundaries'][0]['near_zero'])

    def test_comparison_accepts_same_conditions_without_inventing_speedup(self):
        graph, execution, manifest = complete_fixture()
        before = deepcopy((graph, execution, manifest))
        after_g, after_e = deepcopy(graph), deepcopy(execution)
        after_g['scene_signature'] = after_e['scene_signature'] = 'new-code-same-physical-scene'
        result = compare_runs(graph, execution, after_g, after_e, before_manifest=manifest, after_manifest=manifest)
        self.assertTrue(result['comparison_accepted'], result['comparability_checks'])
        self.assertFalse(result['validated_speedup'])
        self.assertEqual(result['actual_makespan_reduction_s'], 0.)
        self.assertEqual((graph, execution, manifest), before)

    def test_nominal_claim_or_forged_makespan_cannot_replace_actual_execution(self):
        graph, execution, manifest = complete_fixture()
        candidate = deepcopy(execution); candidate['makespan'] = 4.
        result = compare_runs(graph, execution, graph, candidate, before_manifest=manifest, after_manifest=manifest)
        self.assertFalse(result['comparison_accepted'])
        self.assertFalse(result['comparability_checks']['complete_actual_event_coverage'])
        self.assertFalse(result['validated_speedup'])

    def test_looser_limits_or_goals_are_not_accepted_as_speedup(self):
        graph, execution, manifest = complete_fixture()
        for mutate, key in [
            (lambda g: g['time_parameterization'].update(max_velocity_rad_s=1.), 'same_global_and_task_motion_limits'),
            (lambda g: g['validation_requirements'].update(goal_position_tolerance_m=.05), 'same_acceptance_requirements'),
            (lambda g: g['nodes'][0]['metadata'].update(max_acceleration_rad_s2=.2), 'same_global_and_task_motion_limits')]:
            with self.subTest(key=key):
                after = deepcopy(graph); mutate(after)
                result = compare_runs(graph, execution, after, execution, before_manifest=manifest, after_manifest=manifest)
                self.assertFalse(result['comparability_checks'][key])
                self.assertFalse(result['comparison_accepted'])

    def test_missing_or_changed_physical_provenance_prevents_verified_improvement(self):
        graph, execution, manifest = complete_fixture()
        result = compare_runs(graph, execution, graph, execution)
        self.assertIsNone(result['comparability_checks']['same_physical_source_scene'])
        self.assertFalse(result['comparison_accepted'])
        changed = deepcopy(manifest); changed['input_fingerprint']['files']['scenes/10_Scene3_Dynamic.ttt'] = 'other-mass-or-layout'
        result = compare_runs(graph, execution, graph, execution, before_manifest=manifest, after_manifest=changed)
        self.assertFalse(result['comparability_checks']['same_physical_source_scene'])

    def test_deleted_guard_dependency_fails_even_if_old_timestamps_remain_ordered(self):
        graph, execution, manifest = complete_fixture()
        after = deepcopy(graph)
        first, second = (graph['nodes'][i]['id'] for i in (0, 1))
        after['edges'] = [edge for edge in after['edges']
                          if (edge['source'], edge['target']) != (first, second)]
        result = compare_runs(graph, execution, after, execution, before_manifest=manifest, after_manifest=manifest)
        self.assertFalse(result['comparability_checks']['all_original_dependencies_retained'])
        self.assertTrue(result['dependency_retention']['lost_dependencies'])

    def test_removing_only_a_transitively_redundant_edge_is_allowed(self):
        graph, execution, manifest = complete_fixture()
        before = deepcopy(graph)
        before['edges'].append({'source': graph['nodes'][0]['id'], 'target': graph['nodes'][2]['id'],
                                'type': 'type2', 'reason': 'task_dependency'})
        result = compare_runs(before, execution, graph, execution, before_manifest=manifest, after_manifest=manifest)
        self.assertTrue(result['comparability_checks']['all_original_dependencies_retained'])

    def test_actual_load_proof_is_reaudited_instead_of_trusting_success_flag(self):
        graph, execution, manifest = complete_fixture()
        candidate = deepcopy(execution)
        for row in candidate['trace']: row['applied_load_substeps'] = 0
        result = compare_runs(graph, execution, graph, candidate, before_manifest=manifest, after_manifest=manifest)
        self.assertFalse(result['comparability_checks']['after_actual_audit_passed'])
        self.assertFalse(result['validated_speedup'])

    def test_sparse_after_velocity_trace_cannot_fake_better_sampling(self):
        graph, execution, manifest = complete_fixture()
        candidate = deepcopy(execution)
        del candidate['trace'][10]['A_joint_velocity_rad_s']
        result = compare_runs(graph, execution, graph, candidate, before_manifest=manifest, after_manifest=manifest)
        self.assertFalse(result['comparability_checks']['A_velocity_sampling_comparable'])


if __name__ == '__main__':
    unittest.main()
