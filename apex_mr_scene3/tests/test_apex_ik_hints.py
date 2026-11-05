"""IK continuity and historical guesses obey the current-geometry contract.

The deterministic oracle checks compiler precedence and context forwarding;
these tests do not substitute for the simulator's IK/collision qualification.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex.compile import Compiler, CompileFailure


class GeometryOracle:
    def __init__(self):
        self.calls = []
        self.reject = lambda request: False

    def solve_ik(self, arm, position, quaternion, payload, placed, options):
        request = deepcopy(dict(arm=arm, position=position, quaternion=quaternion,
                                payload=payload, placed=placed, options=options))
        self.calls.append(request)
        if self.reject(request):
            return {'ok': False, 'problem': 'revised body collides at this IK branch'}
        # Returning a refined value distinct from every supplied historical
        # guess detects accidentally returning the hint without solving.
        return {'ok': True, 'q': [v + .001 for v in options['seed']],
                'stats': {'collision_checks': 1, 'qualified_current_geometry': True}}


class IKHintTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='scene3_ik_hint_')
        self.root = Path(self.temp.name)
        (self.root / 'scenes').mkdir()
        (self.root / 'scenes/10_Scene3_Dynamic.ttt').write_bytes(b'current scene geometry')
        (self.root / 'apex').mkdir()
        (self.root / 'apex/motion_backend.lua').write_bytes(b'current strict IK backend')
        self.out = self.root / 'output'
        self.out.mkdir()
        self.hint = [.6, .5, .4, .3, .2, .1]
        self.pose = [0., 0., 1., 0., 0., 0., 1.]
        self.cfg = {'arms': {'A': {'home_q': [0.] * 6}, 'B': {'home_q': [.1] * 6}},
                    'free_q': {'A': [.2] * 6, 'B': [.3] * 6},
                    'initial': {'K1': self.pose}, 'goal': {'K1': self.pose}}
        self.oracle = GeometryOracle()
        self.root_patch = patch('apex.compile.ROOT', self.root)
        self.root_patch.start()
        self.write_reference([{'id': 'route', 'arm': 'B', 'q_path': [[0.] * 6, self.hint]}])
        self.compiler = self.make_compiler()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def write_reference(self, value):
        (self.out / 'reference_tasks.json').write_text(json.dumps(value), encoding='utf-8')

    def make_compiler(self):
        return Compiler(None, self.oracle, self.cfg, self.out)

    def prime_current_cache(self):
        c = self.compiler
        result = c.ik('B', self.pose, label='unknown_label')
        self.oracle.calls.clear()
        return result

    def test_current_branch_precedes_historical_hint_and_uses_its_exact_cache(self):
        current = self.prime_current_cache()
        result = self.compiler.ik('B', self.pose, label='route')
        self.assertEqual(self.oracle.calls, [])
        self.assertEqual(result, current)
        self.assertNotEqual(result, self.hint)
        self.assertEqual(self.compiler.calls[-1]['seed_source'], 'current')
        self.assertTrue(self.compiler.calls[-1]['cached'])

    def test_cached_distant_hint_cannot_preempt_fresh_current_branch(self):
        self.compiler.ik('B', self.pose, label='route', seed=self.hint)
        self.oracle.calls.clear()
        result = self.compiler.ik('B', self.pose, label='route')
        self.assertEqual(len(self.oracle.calls), 1)
        self.assertEqual(self.oracle.calls[0]['options']['seed'], self.compiler.q['B'])
        self.assertEqual(result, [q + .001 for q in self.compiler.q['B']])
        self.assertEqual(self.compiler.calls[-1]['seed_source'], 'current')

    def test_explicit_seed_overrides_hint_and_fallback_cache(self):
        self.prime_current_cache()
        explicit = [-.4] * 6
        result = self.compiler.ik('B', self.pose, label='route', seed=explicit)
        self.assertEqual(len(self.oracle.calls), 1)
        self.assertEqual(self.oracle.calls[0]['options']['seed'], explicit)
        self.assertEqual(result, [q + .001 for q in explicit])
        self.assertEqual(self.compiler.calls[-1]['seed_source'], 'explicit')

    def test_colliding_current_branch_falls_back_to_qualified_historical_cache(self):
        fallback = self.compiler.ik('B', self.pose, label='route', seed=self.hint)
        self.oracle.calls.clear()
        self.oracle.reject = lambda req: req['options']['seed'] == self.compiler.q['B']
        result = self.compiler.ik('B', self.pose, label='route')
        self.assertEqual(len(self.oracle.calls), 1)
        self.assertEqual(result, fallback)
        self.assertTrue(self.compiler.calls[-1]['cached'])
        self.assertEqual(self.compiler.calls[-1]['seed_source'], 'prior_reference')
        # A failed current-branch solve is never cached as a success.
        self.compiler.ik('B', self.pose, label='route')
        self.assertEqual(len(self.oracle.calls), 2)

    def test_distant_fallback_cache_does_not_skip_untried_historical_seed(self):
        self.compiler.ik('B', self.pose, seed=self.cfg['free_q']['B'])
        self.oracle.calls.clear()
        self.oracle.reject = lambda req: req['options']['seed'] == self.compiler.q['B']
        result = self.compiler.ik('B', self.pose, label='route')
        self.assertEqual([c['options']['seed'] for c in self.oracle.calls],
                         [self.compiler.q['B'], self.hint])
        self.assertEqual(result, [q + .001 for q in self.hint])
        self.assertEqual(self.compiler.calls[-1]['seed_source'], 'prior_reference')

    def test_preferred_request_carries_full_current_geometry_and_exact_cache_key(self):
        c = self.compiler
        payload = {'part': 'K1', 'offset': [0., 0., .03], 'quaternion': [0., 0., 0., 1.]}
        c.payloads['A'] = {'part': 'K2'}
        c.art['SupportLift'] = .010
        c.contact_holds['tool_A'] = [['A_GripperPalm', 'K2']]
        contacts = [['B_GripperPalm', 'K1']]
        result = c.ik('B', self.pose, payload, contacts, label='route')
        request = self.oracle.calls[-1]
        self.assertEqual(request['payload'], payload)
        self.assertEqual(request['placed'], c.placed)
        self.assertEqual(request['options']['other_q'], c.q['A'])
        self.assertEqual(request['options']['other_payload'], c.payloads['A'])
        self.assertEqual(request['options']['articulations'], c.art)
        self.assertIn(contacts[0], request['options']['contact_exclusions'])
        self.assertIn(['A_GripperPalm', 'K2'], request['options']['contact_exclusions'])
        self.assertEqual(c.ik('B', self.pose, payload, contacts, label='route'), result)
        self.assertEqual(len(self.oracle.calls), 1)  # Same preferred current-scene evidence.
        c.placed['K1'][0] += 1e-12
        c.ik('B', self.pose, payload, contacts, label='route')
        self.assertEqual(len(self.oracle.calls), 2)  # No rounding away changed geometry.
        c.sig = 'new-scene-and-backend-signature'
        c.ik('B', self.pose, payload, contacts, label='route')
        self.assertEqual(len(self.oracle.calls), 3)

    def test_all_current_geometry_failures_never_return_historical_q(self):
        self.oracle.reject = lambda req: True
        with self.assertRaises(CompileFailure):
            self.compiler.ik('B', self.pose, label='route')
        self.assertEqual(self.oracle.calls[0]['options']['seed'], self.compiler.q['B'])
        self.assertFalse(list((self.out / 'ik_solutions').glob('*.json')))

    def test_missing_corrupt_and_invalid_reference_records_are_ignored(self):
        reference = self.out / 'reference_tasks.json'
        reference.unlink()
        self.assertEqual(self.make_compiler().reference_ik_seeds, {})
        reference.write_text('{broken', encoding='utf-8')
        self.assertEqual(self.make_compiler().reference_ik_seeds, {})
        self.write_reference([
            {'id': 'good', 'arm': 'A', 'q_end': [.4] * 6},
            {'id': 'bad_size', 'arm': 'B', 'q_end': [.3]},
            {'id': 'bad_bool', 'arm': 'B', 'q_end': [True] * 6},
            {'id': 'bad_nan', 'arm': 'B', 'q_end': [float('nan')] * 6},
            None])
        self.assertEqual(self.make_compiler().reference_ik_seeds,
                         {'good': {'arm': 'A', 'q': [.4] * 6}})
        self.write_reference({'stage': 'incomplete_reference_diagnostic', 'tasks': [
            {'id': 'partial', 'arm': 'B', 'q_end': self.hint}]})
        self.assertEqual(self.make_compiler().reference_ik_seeds, {})

    def test_other_arm_and_unknown_labels_do_not_use_historical_hint(self):
        self.compiler.ik('A', self.pose, label='route')
        self.assertEqual(self.oracle.calls[-1]['options']['seed'], self.compiler.q['A'])
        self.oracle.calls.clear()
        self.prime_current_cache()
        self.compiler.ik('B', self.pose, label='another_unknown')
        self.assertEqual(self.oracle.calls, [])
        self.assertTrue(self.compiler.calls[-1]['cached'])

    def test_historical_path_is_only_a_small_endpoint_corrected_candidate(self):
        target = [q+.001 for q in self.hint]
        candidates = self.compiler.candidate_paths('route', 'B', target)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['path'][0], self.compiler.q['B'])
        self.assertEqual(candidates[0]['path'][-1], target)
        self.assertEqual(candidates[0]['source'], 'previous_reference:route')
        self.assertLessEqual(candidates[0]['max_endpoint_correction_rad'], .12)
        self.assertEqual(self.compiler.reference_paths[('B', 'route')][0]['path'][0], [0.]*6)
        self.assertEqual(self.compiler.candidate_paths('route', 'B', [q+1 for q in target]), [])

    def test_candidate_alignment_preserves_whole_path_winding(self):
        path = [[-1.]*6, [-.8]*6, [-.5]*6]
        self.compiler.reference_paths = {('B', 'wrapped'): [{'path': path, 'source': 'baseline'}]}
        self.compiler.q['B'] = [-1+2*math.pi]*6
        target = [-.5+2*math.pi]*6
        candidates = self.compiler.candidate_paths('wrapped', 'B', target)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]['path'][1][0], -.8+2*math.pi)
        self.assertEqual(path[1], [-.8]*6)

    def test_compiler_loads_baseline_and_previous_geometry_candidates(self):
        file = self.root/'backups/smooth_motion_baseline_20261104_01/results/apex/reference_tasks.json'
        file.parent.mkdir(parents=True)
        file.write_text(json.dumps([{'id': 'route', 'arm': 'B',
                                     'q_path': [[0.]*6, [.2]*6, self.hint]}]), encoding='utf-8')
        candidates = self.make_compiler().reference_paths[('B', 'route')]
        self.assertEqual([c['source'] for c in candidates], ['baseline_reference:route', 'previous_reference:route'])


if __name__ == '__main__':
    unittest.main()
