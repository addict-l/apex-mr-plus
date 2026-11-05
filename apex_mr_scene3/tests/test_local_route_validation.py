from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from apex import local_routes as routes


def q(x):
    return [x, 0., 0., 0., 0., 0.]


class NativeAdapter:
    def __init__(self):
        self.valid = True
        self.samples = []
        self.move_skill = False
        self.fk_calls = 0

    def validate_samples(self, samples):
        self.samples.extend(deepcopy(samples))
        return [{'ok': self.valid} for _ in samples]

    def fk(self, arm, joint):
        self.fk_calls += 1
        return {'ok': True, 'pose': [(.01 if self.move_skill and self.fk_calls % 2 == 0 else 0), 0, 0, 0, 0, 0, 1]}


class LocalRouteValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ('scenes/10_Scene3_Dynamic.ttt', 'apex/motion_backend.lua', 'config/runtime.json'):
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic physical input')
        self.original = []
        self.windows = []
        start = 0.
        for index, (name, (first, last)) in enumerate(routes.WINDOWS.items()):
            end = (index+1)*.4
            paths = {first: [q(start), q(-.2)], last: [q(-.2), q(end)]}
            self.windows.append({'name': name, 'paths': paths})
            for ident, kind, path in ((first, 'transit', [q(start), q(4)]),
                                      ('semantic_skill_'+name, 'grasp', [q(4), q(4)]),
                                      (last, 'transit', [q(4), q(end)])):
                self.original.append({'id': ident, 'arm': 'B', 'kind': kind, 'q_path': path,
                    'payload': None, 'duration': 1., 'dependencies': [],
                    'metadata': {'placed': {}, 'contact_exclusions': [['exact_tool', 'exact_part']],
                                 'contact_tolerance': .002, 'options': {'articulations': {'SupportLift': 0.}}}})
            start = end
        self.asset = {'schema_version': 1, 'reference_signature': routes._value_hash(self.original),
                      'physical_input_hashes': {name: routes._sha(self.root/name) for name in
                         ('scenes/10_Scene3_Dynamic.ttt', 'apex/motion_backend.lua', 'config/runtime.json')},
                      'windows': self.windows}
        self.asset_path = self.root/'apex/local_route_candidates.json'
        self.asset_path.write_text(json.dumps(self.asset), encoding='utf-8')
        self.planner = NativeAdapter()
        self.compiler = SimpleNamespace(home={'A': q(0), 'B': q(0)}, p=self.planner)
        self.patch = patch.object(routes, 'ROOT', self.root); self.patch.start()

    def tearDown(self):
        self.patch.stop(); self.temp.cleanup()

    def apply(self):
        return routes.apply_local_routes(self.compiler, {'tasks': self.original}, self.original, self.asset_path)

    def test_current_native_queries_preserve_exact_context_and_semantic_endpoint(self):
        result, evidence = self.apply()
        self.assertEqual(len(result), len(self.original))
        self.assertGreater(evidence['samples'], 100)
        self.assertEqual(self.planner.fk_calls, 6)
        self.assertTrue(all(s['options']['contact_exclusions'] == [['exact_tool', 'exact_part']] for s in self.planner.samples))
        self.assertTrue(all(s['A'] == q(0) and s['payloads'] == {'A': None, 'B': None} for s in self.planner.samples))

    def test_historical_path_does_not_override_a_current_geometry_failure(self):
        self.planner.valid = False
        with self.assertRaisesRegex(ValueError, 'native geometry'):
            self.apply()

    def test_changed_skill_pose_is_rejected_even_when_collision_free(self):
        self.planner.move_skill = True
        with self.assertRaisesRegex(ValueError, 'physical skill endpoint'):
            self.apply()

    def test_changed_scene_or_baseline_identity_rejects_cached_candidate(self):
        (self.root/'scenes/10_Scene3_Dynamic.ttt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'physical inputs'):
            self.apply()
        self.asset['reference_signature'] = 'wrong'
        self.asset_path.write_text(json.dumps(self.asset), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'baseline reference identity'):
            self.apply()


if __name__ == '__main__':
    unittest.main()
