"""Rebuild conflicts against changed curved node geometry, not old evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex.tpg import build_tpg, rebuild_motion_conflicts


def q(x, y=0.):
    return [x, y, 0., 0., 0., 0.]


def fixture():
    tasks = [
        {'id': 'a_move', 'arm': 'A', 'kind': 'transit', 'duration': 1.,
         'q_path': [q(0.), q(1.)], 'reference_order': 0, 'dependencies': []},
        {'id': 'b_wait', 'arm': 'B', 'kind': 'hold', 'duration': .4,
         'q_path': [q(.5, 1.), q(.5, 1.)], 'reference_order': 1, 'dependencies': []},
        {'id': 'a_follow', 'arm': 'A', 'kind': 'transit', 'duration': .5,
         'q_path': [q(1.), q(1.2)], 'reference_order': 2, 'dependencies': ['b_wait']},
    ]
    graph = build_tpg(tasks, lambda _a, _b: False, max_joint_span=2.)
    # Existing time-parameterization fields must be retained byte-for-value;
    # this function must not invoke any particular trajectory parameterizer.
    for node in graph['nodes']:
        node['motion_segments'] = [{'q0': node['q_start'][:], 'q1': node['q_end'][:],
                                    'duration': node['duration'], 'profile_phase': [.1, .9]}]
    graph['time_parameterization'] = {'type': 'existing_profile', 'max_velocity_rad_s': .6}
    graph['certificate'].update(waiting_endpoints_verified=True, nominal_world_state_verified=True)
    return graph


def intersects_samples(a, b):
    hit = any(max(abs(x-y) for x, y in zip(qa, qb)) < 1e-8
              for qa in a['q_samples'] for qb in b['q_samples'])
    return {'conflict': hit, 'evidence': {'oracle': 'strict synthetic current-sample geometry'}}


class RebuildMotionConflictTests(unittest.TestCase):
    def test_new_interior_curve_is_queried_without_changing_nodes_or_timing(self):
        graph = fixture()
        graph['nodes'][0]['q_samples'] = [q(0.), q(.5, 1.), q(1.)]
        frozen_nodes = deepcopy(graph['nodes']); frozen_timing = deepcopy(graph['time_parameterization'])
        base_edges = deepcopy(graph['edges']); called = []
        def oracle(a, b):
            called.append((a['id'], b['id']))
            result = intersects_samples(a, b)
            a['q_samples'][0][0] = 999.  # Isolated oracle arguments.
            return result
        result = rebuild_motion_conflicts(graph, oracle)
        self.assertIs(result, graph)
        self.assertEqual(graph['nodes'], frozen_nodes)
        self.assertEqual(graph['time_parameterization'], frozen_timing)
        self.assertEqual(len(graph['nodes']), 3)
        self.assertTrue(all(edge in graph['edges'] for edge in base_edges))
        collisions = [e for e in graph['edges'] if e['reason'] == 'motion_conflict']
        self.assertEqual([(x['source'], x['target']) for x in collisions],
                         [('a_move::p0000', 'b_wait::skill')])
        self.assertEqual(called, [('a_move::p0000', 'b_wait::skill')])
        self.assertEqual(graph['counts']['cross_arm_pairs'], 2)
        self.assertEqual(graph['counts']['already_ordered_pairs_skipped'], 1)
        self.assertTrue(graph['nominal_schedule_requires_refresh'])
        self.assertFalse(graph['certificate']['waiting_endpoints_verified'])
        self.assertFalse(graph['certificate']['nominal_world_state_verified'])

    def test_old_conflict_and_continuous_gates_cannot_hide_new_queries(self):
        graph = fixture(); base_edges = deepcopy(graph['edges'])
        graph['edges'] += [
            {'source': 'a_move::p0000', 'target': 'b_wait::skill', 'type': 'type2',
             'reason': 'motion_conflict', 'evidence': {'stale': True}},
            {'source': 'a_move::p0000', 'target': 'b_wait::skill', 'type': 'type2',
             'reason': 'continuous_run_gate'},
        ]
        graph['geometry_checks'] = [{'stale': True}]
        rebuild_motion_conflicts(graph, intersects_samples)
        self.assertEqual(graph['edges'], base_edges)
        self.assertEqual(graph['counts']['oracle_queries'], 1)
        self.assertEqual(graph['counts']['continuous_gate_edges'], 0)
        self.assertFalse(graph['geometry_checks'][0]['conflict'])
        self.assertNotIn('stale', graph['geometry_checks'][0])

    def test_skip_false_queries_ordered_pairs_and_respects_sorted_reference_indices(self):
        graph = fixture(); graph['nodes'].reverse()
        original_order = [n['id'] for n in graph['nodes']]; queried = []
        rebuild_motion_conflicts(graph, lambda a, b: queried.append((a['reference_index'], b['reference_index'])) or False,
                                 skip_ordered_pairs=False)
        self.assertEqual(queried, [(0, 1), (1, 2)])
        self.assertEqual([n['id'] for n in graph['nodes']], original_order)
        self.assertEqual(graph['counts']['oracle_queries'], 2)

    def test_new_conflict_updates_closure_before_later_pair_queries(self):
        tasks = [{'id': name, 'arm': arm, 'kind': 'transit', 'duration': 1.,
                  'q_path': [q(index), q(index+.1)], 'reference_order': index}
                 for index, (name, arm) in enumerate([('a0', 'A'), ('b0', 'B'), ('b1', 'B'), ('a1', 'A')])]
        graph = build_tpg(tasks, lambda a, b: False, max_joint_span=2.)
        calls = []
        def oracle(a, b):
            calls.append((a['task_id'], b['task_id']))
            return True
        rebuild_motion_conflicts(graph, oracle)
        self.assertIn(('a0', 'b0'), calls)
        self.assertNotIn(('a0', 'b1'), calls)  # New a0->b0 plus existing b0->b1.
        self.assertEqual(graph['counts']['cross_arm_pairs'], 4)
        self.assertEqual(graph['counts']['oracle_queries'], 3)
        self.assertEqual(graph['counts']['already_ordered_pairs_skipped'], 1)

    def test_oracle_rejecting_reference_order_does_not_commit_partial_graph(self):
        graph = fixture(); original = deepcopy(graph)
        with self.assertRaisesRegex(ValueError, 'Reference serialization rejected'):
            rebuild_motion_conflicts(graph, lambda a, b: {'conflict': True, 'allowed_orders': [[b['id'], a['id']]]})
        self.assertEqual(graph, original)

    def test_cycle_and_backward_semantic_dependencies_are_rejected(self):
        for cycle in (True, False):
            with self.subTest(cycle=cycle):
                graph = fixture()
                graph['edges'].append({'source': 'b_wait::skill', 'target': 'a_move::p0000',
                                       'type': 'type2', 'reason': 'task_dependency'})
                if cycle:
                    graph['edges'].append({'source': 'a_move::p0000', 'target': 'b_wait::skill',
                                           'type': 'type2', 'reason': 'task_dependency'})
                original = deepcopy(graph)
                with self.assertRaisesRegex(ValueError, 'cycle' if cycle else 'Reference order violates'):
                    rebuild_motion_conflicts(graph, intersects_samples)
                self.assertEqual(graph, original)

    def test_missing_geometry_cannot_inherit_prior_success_certificate(self):
        graph = fixture(); graph['nodes'][0]['q_samples'] = []
        rebuild_motion_conflicts(graph, intersects_samples)
        self.assertFalse(graph['certificate']['all_nodes_have_geometry'])
        self.assertFalse(graph['certificate']['motion_overlap_pairs_checked'])
        self.assertEqual(graph['counts']['pairs_without_geometry'], 1)

    def test_unknown_dependency_or_missing_oracle_is_not_silently_discarded(self):
        graph = fixture()
        with self.assertRaisesRegex(ValueError, 'requires a collision oracle'):
            rebuild_motion_conflicts(graph, None)
        graph['edges'].append({'source': 'a_move::p0000', 'target': 'b_wait::skill',
                               'type': 'type2', 'reason': 'new_material_safety_guard'})
        original = deepcopy(graph)
        with self.assertRaisesRegex(ValueError, 'Unknown base dependency'):
            rebuild_motion_conflicts(graph, intersects_samples)
        self.assertEqual(graph, original)


if __name__ == '__main__': unittest.main()
