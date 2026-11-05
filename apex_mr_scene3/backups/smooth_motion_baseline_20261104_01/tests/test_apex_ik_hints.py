"""Historical IK guesses must pass the normal current-geometry solver contract.

The deterministic oracle checks compiler precedence and context forwarding;
these tests do not substitute for the simulator's IK/collision qualification.
"""
from copy import deepcopy
import json
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

    def test_preferred_hint_is_refined_before_existing_fallback_cache(self):
        fallback = self.prime_current_cache()
        result = self.compiler.ik('B', self.pose, label='route')
        self.assertEqual(len(self.oracle.calls), 1)
        self.assertEqual(self.oracle.calls[0]['options']['seed'], self.hint)
        self.assertEqual(result, [q + .001 for q in self.hint])
        self.assertNotEqual(result, self.hint)
        self.assertNotEqual(result, fallback)
        self.assertEqual(self.compiler.calls[-1]['seed_source'], 'prior_reference')
        self.assertFalse(self.compiler.calls[-1]['cached'])

    def test_explicit_seed_overrides_hint_and_fallback_cache(self):
        self.prime_current_cache()
        explicit = [-.4] * 6
        result = self.compiler.ik('B', self.pose, label='route', seed=explicit)
        self.assertEqual(len(self.oracle.calls), 1)
        self.assertEqual(self.oracle.calls[0]['options']['seed'], explicit)
        self.assertEqual(result, [q + .001 for q in explicit])
        self.assertEqual(self.compiler.calls[-1]['seed_source'], 'explicit')

    def test_colliding_hint_is_rejected_before_valid_cached_fallback(self):
        fallback = self.prime_current_cache()
        self.oracle.reject = lambda req: req['options']['seed'] == self.hint
        result = self.compiler.ik('B', self.pose, label='route')
        self.assertEqual(len(self.oracle.calls), 1)
        self.assertEqual(result, fallback)
        self.assertTrue(self.compiler.calls[-1]['cached'])
        # Failure was not cached as accepted historical evidence.
        self.compiler.ik('B', self.pose, label='route')
        self.assertEqual(len(self.oracle.calls), 2)

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
        self.assertEqual(self.oracle.calls[0]['options']['seed'], self.hint)
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


if __name__ == '__main__':
    unittest.main()
