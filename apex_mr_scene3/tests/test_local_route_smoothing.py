"""Synthetic API fixtures for local-route smoothing; no robot proof is implied."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex.blend_validation import validate_profile, local_route_group
from apex.curve import make_profile, eval_at_s
from apex.program import parameterize
from apex.smoothing import smooth_cartesian_macros
from apex.tpg import build_tpg


def q(x, y=0.):
    return [x, y, 0., 0., 0., 0.]


def fixture(span=1.):
    payload = {'part': 'Gate', 'name': 'Gate', 'offset': [0.,0.,.006], 'quaternion': [0.,0.,0.,1.]}
    path = [q(0.), q(.08), q(.08+.08*math.cos(math.radians(5)), .08*math.sin(math.radians(5)))]
    declaration = {'window': 'gate_lift', 'method': 'cartesian_continuation',
                   'certificate_scope': 'native dense fixed-reference-state', 'require_profile_validation': True}
    metadata = {'placed': {'Bridge': [0.,0.,1.,0.,0.,0.,1.]},
                'contact_exclusions': [['A_SupportShoe','Bridge']],
                'options': {'articulations': {'SupportLift': 0.}},
                'local_route_candidate': declaration}
    task = {'id': 'pick_gate_lift', 'arm': 'B', 'kind': 'transit', 'q_path': path,
            'duration': 1., 'dependencies': [], 'metadata': metadata, 'payload': payload}
    graph = build_tpg([task], lambda _a,_b: False, max_joint_span=span)
    cfg = {'arms': {'A': {'home_q': q(.4)}, 'B': {'home_q': q(0.)}},
           'initial_payloads': {'B': payload}}
    return graph, cfg, path


class RecordingPlanner:
    """Deliberately synthetic FK and collision API for integration invariants."""
    def __init__(self, collision=False, fk_scale=1.):
        self.batches = []; self.fk_calls = 0; self.collision = collision; self.scale = fk_scale
        self.info = {'collision_geometry_policy': 'synthetic test fixture only'}

    def validate_samples(self, samples):
        self.batches.extend(deepcopy(samples))
        return [{'ok': not self.collision} for _ in samples]

    def fk(self, arm, configuration):
        self.fk_calls += 1
        return {'ok': True, 'pose': [x*self.scale for x in configuration[:3]]+[0.,0.,0.,1.]}


class LocalRouteSmoothingTests(unittest.TestCase):
    def test_explicit_single_node_route_is_checked_and_keeps_semantics(self):
        graph, cfg, _ = fixture()
        self.assertEqual(len(graph['nodes']), 1)
        original = deepcopy(graph['nodes'][0]); planner = RecordingPlanner()
        report = smooth_cartesian_macros(graph, planner, cfg)
        self.assertEqual(report['accepted_local_route_tasks'], 1)
        self.assertEqual(report['accepted_cartesian_macros'], 0)
        record = report['candidate_macros'][0]
        self.assertEqual(record['macro'], 'pick_gate_lift')
        self.assertEqual(record['group_kind'], 'local_route_task')
        self.assertTrue(record['attempts'][-1]['timing_non_regression'])
        self.assertGreater(len(planner.batches), 2)
        self.assertGreater(planner.fk_calls, 2)
        for sample in planner.batches:
            self.assertEqual(sample['A'], cfg['arms']['A']['home_q'])
            self.assertEqual(sample['payloads']['B'], cfg['initial_payloads']['B'])
            self.assertEqual(sample['options']['articulations'], {'SupportLift': 0.})
            self.assertEqual(sample['options']['contact_exclusions'], original['metadata']['contact_exclusions'])
        node = graph['nodes'][0]
        for key in ('id','task_id','payload','metadata'):
            self.assertEqual(node[key], original[key])
        np.testing.assert_allclose(node['q_start'], original['q_start'], atol=1e-12)
        np.testing.assert_allclose(node['q_end'], original['q_end'], atol=1e-12)
        self.assertEqual(node['original_q_samples'], original['q_samples'])
        parameterize(graph)
        self.assertEqual(node['motion_segments'][0]['geometry_profile'], node['curve_profile'])

    def test_every_fragment_of_one_task_is_in_one_new_group(self):
        graph, cfg, _ = fixture(span=.04)
        self.assertGreater(len(graph['nodes']), 1)
        before = deepcopy(graph)
        report = smooth_cartesian_macros(graph, RecordingPlanner(), cfg)
        self.assertEqual(report['accepted_local_route_tasks'], 1)
        self.assertEqual(report['candidate_macros'][0]['task_ids'], ['pick_gate_lift'])
        parameterize(graph)
        self.assertEqual([n['id'] for n in graph['nodes']], [n['id'] for n in before['nodes']])
        self.assertEqual(graph['nodes'][0]['metadata']['motion_run_members'], [n['id'] for n in graph['nodes']])
        self.assertAlmostEqual(sum(n['duration'] for n in graph['nodes']), graph['nodes'][0]['curve_profile']['duration'])

    def test_unmarked_or_wrongly_declared_task_is_not_smoothed(self):
        for alteration in ('unmarked','unknown_task','wrong_window','no_validation'):
            with self.subTest(alteration=alteration):
                graph, cfg, _ = fixture(); node = graph['nodes'][0]
                if alteration == 'unmarked': node['metadata'].pop('local_route_candidate')
                elif alteration == 'unknown_task': node['task_id'] = 'arbitrary_transit'
                elif alteration == 'wrong_window': node['metadata']['local_route_candidate']['window'] = 'support_pick'
                else: node['metadata']['local_route_candidate']['require_profile_validation'] = False
                before = deepcopy(node); planner = RecordingPlanner()
                report = smooth_cartesian_macros(graph, planner, cfg)
                self.assertEqual(report['accepted_macros'], 0)
                self.assertEqual(node, before)
                self.assertFalse(planner.batches)

    def test_local_group_cannot_cross_task_or_effect_boundary(self):
        graph, cfg, path = fixture()
        profile = make_profile(path, .6, 1.2, 8.)
        first = deepcopy(graph['nodes'][0]); second = deepcopy(first)
        first.pop('fragment_count'); second.pop('fragment_count')
        second['task_id'] = 'close_gate_approach'
        proof = validate_profile(RecordingPlanner(), cfg, [first, second], profile, eval_at_s)
        self.assertFalse(proof['accepted'])
        self.assertEqual(proof['reason'], 'local_route_crosses_task_or_declaration_boundary')
        graph['nodes'][0]['metadata']['effects'] = {'payloads': {'B': None}}
        report = smooth_cartesian_macros(graph, RecordingPlanner(), cfg)
        self.assertEqual(report['accepted_macros'], 0)
        self.assertEqual(report['candidate_macros'][0]['reason'], 'semantic_or_contact_context_changes')

    def test_incomplete_task_declaration_cannot_smooth_only_its_tail(self):
        graph, cfg, _ = fixture(span=.04)
        graph['nodes'][0]['metadata'].pop('local_route_candidate')
        planner = RecordingPlanner()
        report = smooth_cartesian_macros(graph, planner, cfg)
        self.assertEqual(report['accepted_macros'], 0)
        self.assertFalse(planner.batches)
        self.assertEqual(report['candidate_macros'][0]['reason'], 'local_route_does_not_cover_complete_task')

    def test_failed_collision_or_original_fk_corridor_never_updates_path(self):
        for planner in (RecordingPlanner(collision=True), RecordingPlanner(fk_scale=10.)):
            graph, cfg, _ = fixture(); before = deepcopy(graph['nodes'])
            report = smooth_cartesian_macros(graph, planner, cfg)
            self.assertEqual(report['accepted_macros'], 0)
            self.assertEqual(graph['nodes'], before)
            self.assertEqual(report['tcp_corridor_limit_m'], .0005)
            self.assertEqual(report['orientation_limit_rad'], .005)

    def test_profile_time_regression_is_rejected_without_claiming_geometry_check(self):
        graph, cfg, _ = fixture(); before = deepcopy(graph['nodes']); planner = RecordingPlanner()
        def expensive_profile(*args, **kwargs):
            result = make_profile(*args, **kwargs); result['duration'] *= 100.
            return result
        with patch('apex.smoothing.make_profile', side_effect=expensive_profile):
            report = smooth_cartesian_macros(graph, planner, cfg)
        self.assertEqual(report['accepted_macros'], 0)
        self.assertEqual(graph['nodes'], before)
        self.assertFalse(planner.batches)
        self.assertEqual(report['candidate_macros'][0]['reason'], 'profile_slower_than_original_polyline_timing')
        self.assertTrue(all(a['validation']['validation_performed'] is False for a in report['candidate_macros'][0]['attempts']))


if __name__ == '__main__':
    unittest.main()
