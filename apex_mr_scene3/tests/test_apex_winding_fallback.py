"""Compiler goal-winding/path fallback contracts; synthetic planning fixtures."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex.compile import Compiler, CompileFailure


def q(first, second=0.):
    return [first, second, 0., 0., 0., 0.]


class WindingPlanner:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def plan(self, arm, start, end, payload, placed, solve_time, options):
        self.calls.append(deepcopy(dict(arm=arm, start=start, end=end, payload=payload,
                                       placed=placed, solve_time=solve_time, options=options)))
        path = self.answer(start, end)
        if path is None:
            return {'ok': False, 'error': 'Disconnected winding in synthetic obstacle geometry',
                    'problem': {'reason': 'synthetic_connection_collision'}}
        return {'ok': True, 'path': path, 'algorithm': 'CollisionCheckedJointLine',
                'path_source': 'synthetic_test_candidate',
                'stats': {'synthetic_test_oracle': True, 'ompl_solve_called': False}}


class WindingFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='scene3_winding_fallback_')
        c = self.compiler = Compiler.__new__(Compiler)
        c.out = Path(self.temp.name)
        c.cache = c.out/'rrt_paths';c.cache.mkdir()
        c.sig = 'synthetic_winding_contract'
        c.home = {'A': q(.2), 'B': q(0.)}
        c.source = {'TemporarySupport': [0., 0., 0., 0., 0., 0., 1.]}
        c.calls = []
        c.reset()
        self.payload = {'part': 'TemporarySupport', 'offset': [.03, 0., 0.], 'quaternion': [0., 0., 0., 1.]}
        c.payloads['B'] = self.payload
        c.art['SupportLift'] = .01
        self.requested = q(-math.pi/2)
        self.historical = q(3*math.pi/2)
        c.reference_paths = {('B', 'pick_support_approach'): [
            {'source': 'baseline_reference:pick_support_approach',
             'path': [q(0.), q(2.), self.historical]}]}

    def tearDown(self):
        self.temp.cleanup()

    def run_path(self):
        with redirect_stdout(io.StringIO()):
            return self.compiler.path('pick_support_approach', 'B', self.requested,
                                      self.payload, [['B_Finger', 'TemporarySupport']])

    def test_failed_nearest_connection_tries_known_equivalent_before_long_retry(self):
        self.compiler.p = WindingPlanner(lambda start, end: [start[:], end[:]] if end[0]>0 else None)
        path = self.run_path()
        self.assertEqual(path[-1], self.historical)
        self.assertEqual(self.compiler.q['B'], self.historical)
        self.assertEqual(len(self.compiler.tasks), 1)
        self.assertEqual([call['solve_time'] for call in self.compiler.p.calls], [3., 3.])
        for call in self.compiler.p.calls:
            self.assertEqual(call['payload'], self.payload)
            self.assertEqual(call['options']['other_q'], q(.2))
            self.assertEqual(call['options']['articulations'], {'SupportLift': .01})
            self.assertEqual(call['options']['contact_exclusions'],
                             [['A_SupportShoe', 'Bridge'], ['B_Finger', 'TemporarySupport']])
        self.assertIn('candidate_paths', self.compiler.p.calls[-1]['options'])
        stats = self.compiler.calls[-1]['stats']
        self.assertTrue(stats['winding_fallback'])
        self.assertEqual(stats['requested_goal'], self.requested)
        self.assertEqual(stats['resolved_goal'], self.historical)

    def test_successful_but_long_nearest_route_does_not_preempt_shorter_history(self):
        self.compiler.p = WindingPlanner(lambda start, end:
            [start[:], q(0., 6.), end[:]] if end[0]<0 else [start[:], end[:]])
        path = self.run_path()
        self.assertEqual(path[-1], self.historical)
        self.assertEqual(len(self.compiler.p.calls), 2)
        self.assertTrue(all(row['ok'] for row in self.compiler.calls[-1]['stats']['goal_attempts']))

    def test_lower_bound_skips_goal_that_cannot_beat_direct_nearest_solution(self):
        self.compiler.p = WindingPlanner(lambda start, end: [start[:], end[:]])
        path = self.run_path()
        self.assertEqual(path[-1], self.requested)
        self.assertEqual(len(self.compiler.p.calls), 1)
        attempts = self.compiler.calls[-1]['stats']['goal_attempts']
        self.assertEqual(attempts[-1]['skipped'], 'endpoint_distance_not_better_than_best')

    def test_all_colliding_windings_fail_without_mutating_world_or_tasks(self):
        self.compiler.p = WindingPlanner(lambda start, end: None)
        before = deepcopy(self.compiler.q)
        with self.assertRaises(CompileFailure):
            self.run_path()
        self.assertEqual(self.compiler.q, before)
        self.assertEqual(self.compiler.tasks, [])
        self.assertEqual([call['solve_time'] for call in self.compiler.p.calls], [3., 3., 7., 7.])
        self.assertFalse(list(self.compiler.cache.glob('*.json')))

    def test_resolved_goal_cache_matches_actual_path_not_requested_representation(self):
        self.compiler.p = WindingPlanner(lambda start, end: [start[:], end[:]] if end[0]>0 else None)
        self.run_path()
        self.compiler.reset();self.compiler.payloads['B'] = self.payload;self.compiler.art['SupportLift'] = .01
        self.compiler.p.calls.clear()
        path = self.run_path()
        self.assertEqual(path[-1], self.historical)
        self.assertTrue(self.compiler.calls[-1]['cached'])
        self.assertEqual(len(self.compiler.p.calls), 1)  # The failed nearest query is never success-cached.

    def test_goal_variants_stay_equivalent_and_inside_global_planning_range(self):
        for entry in self.compiler.goal_windings('pick_support_approach', 'B', self.requested):
            for a, b in zip(entry['goal'], self.requested):
                self.assertLessEqual(abs(a), 2*math.pi+1e-10)
                self.assertAlmostEqual(math.sin(a), math.sin(b))
                self.assertAlmostEqual(math.cos(a), math.cos(b))

    def test_physical_ik_branch_fallback_only_after_winding_paths_fail(self):
        c = self.compiler
        c.reference_paths[('B', 'pick_support_approach')] = [
            {'source': 'baseline_other_IK_branch', 'path': [q(0.), q(2.)]}]
        c.p = WindingPlanner(lambda start, end: [start[:], end[:]] if abs(end[0]-2.)<1e-8 else None)
        ik_calls = []

        def ik(arm, tcp, payload, contacts, seed=None, label=None):
            ik_calls.append(deepcopy(dict(arm=arm, tcp=tcp, payload=payload, contacts=contacts, seed=seed)))
            return list(seed) if seed is not None else self.requested

        c.ik = ik
        with redirect_stdout(io.StringIO()):
            path = c.move_pose('pick_support_approach', 'B', 'TemporarySupport',
                               [0., 0., 0., 0., 0., 0., 1.], ([0., 0., 0.], [0., 0., 0., 1.]),
                               self.payload, [['B_Finger', 'TemporarySupport']])
        self.assertEqual(path[-1], q(2.))
        self.assertEqual(c.q['B'], q(2.))
        self.assertEqual(len(ik_calls), 2)
        self.assertEqual(ik_calls[0]['tcp'], ik_calls[1]['tcp'])
        self.assertEqual(ik_calls[0]['contacts'], ik_calls[1]['contacts'])
        self.assertEqual(c.tasks[-1]['metadata']['planning_stats']['ik_branch_fallback'], 'baseline_other_IK_branch')


if __name__ == '__main__':
    unittest.main()
