"""Contact-mode regressions using a strict synthetic collision oracle.

These tests check state reconstruction and declared permissions across skill
completion and stationary waits. They are not simulator geometry evidence.
"""
from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apex.compile import Compiler
from apex.geometry import verify_rollout
from apex.program import parameterize
from apex.tpg import build_tpg


class StrictContactOracle:
    """The gripper touches an unheld fixture until it physically retreats.

    While held at the initial endpoint, the fixture also touches its supporting
    table. Either state requires its exact declared contact pair, irrespective
    of whether that arm has an active graph node or is waiting for the other.
    """
    def __init__(self, target='TemporarySupport'):
        self.target = target
        self.samples = []

    def sample_state_pairs(self, samples):
        self.samples.extend(deepcopy(samples))
        answers = []
        for sample in samples:
            pairs = {frozenset(pair) for pair in sample['options']['contact_exclusions']}
            needed = (frozenset([self.target, 'Table']) if sample['payloads'].get('B')
                      else frozenset(['B_GripperPalm', self.target]))
            valid = sample['B'][0] > .05 or needed in pairs
            answers.append({'ok': valid, 'problem': None if valid else
                            'missing exact contact permission at held or released endpoint'})
        return answers


def contact_fixture(output):
    pose = [0., 0., 0., 0., 0., 0., 1.]
    home = [0.] * 6
    payload = {'part': 'TemporarySupport', 'name': 'TemporarySupport',
               'offset': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    compiler = Compiler.__new__(Compiler)
    compiler.out = output
    compiler.home = {'A': home[:], 'B': home[:]}
    compiler.source = {'TemporarySupport': pose}
    compiler.reset()
    compiler.support_contacts = []
    compiler.tool_contacts = lambda arm, part: [[arm + '_GripperPalm', part]]
    compiler.resting_contacts = lambda part: [[part, 'Table']]
    compiler.skill('grasp', 'B', 'grasp', 'TemporarySupport', payload_after=payload,
                   effects={'payloads': {'B': payload}})
    compiler.payloads['B'] = payload
    compiler.skill('release', 'B', 'release_support', 'TemporarySupport', target=pose,
                   effects={'payloads': {'B': None}, 'placed': {'TemporarySupport': pose}})
    compiler.payloads['B'] = None
    compiler.skill('A_wait', 'A', 'wait', dependencies=['release'], dwell_s=.4)
    compiler.add('B_retreat', 'B', 'transit', [home, [1.] * 6],
                 {'placed': deepcopy(compiler.placed), 'contact_exclusions': compiler.contact_context()},
                 dependencies=['A_wait'])
    compiler.clear_contact_hold('B_retreat', 'B')
    graph = build_tpg(compiler.tasks)
    parameterize(graph)
    cfg = {'parts': [{'id': 'TemporarySupport', 'source': pose, 'goal': pose, 'dependencies': []}],
           'arms': {'A': {'home_q': home}, 'B': {'home_q': home}}, 'table': 'Table'}
    return graph, cfg


class ContactModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='scene3_contact_regression_')
        self.output = Path(self.temp.name)
        self.graph, self.cfg = contact_fixture(self.output)

    def tearDown(self):
        self.temp.cleanup()

    def test_grasp_release_and_other_arm_wait_preserve_contact_permissions(self):
        oracle = StrictContactOracle()
        report = verify_rollout(self.graph, oracle, self.cfg, outdir=self.output / 'valid')
        self.assertTrue(report['valid'], report)
        self.assertGreater(report['tested_samples'], 50)
        # At the beginning of A's task, B is stationary at its released fixture.
        waiting = self.graph['task_map']['A_wait']['first_node']
        start = self.graph['nominal_timestamps'][waiting]['start']
        self.assertGreater(start, 0.)
        expected = frozenset(['B_GripperPalm', 'TemporarySupport'])
        wait_metadata = self.graph['task_map']['A_wait']['metadata']
        self.assertIn(expected, {frozenset(pair) for pair in wait_metadata['contact_exclusions']})
        self.assertNotIn(expected, {frozenset(pair) for pair in oracle.samples[-1]['options']['contact_exclusions']})

    def test_old_skill_metadata_fails_the_contact_state_regression(self):
        broken = deepcopy(self.graph)
        for node in broken['nodes']:
            if node['node_type'] == 'skill':
                node['metadata']['contact_exclusions'] = []
                node['metadata'].get('effects', {}).pop('support_contacts', None)
        report = verify_rollout(broken, StrictContactOracle(), self.cfg, outdir=self.output / 'old')
        self.assertFalse(report['valid'])
        self.assertIsNotNone(report['first_violation'])

    def test_release_permission_does_not_exempt_an_unrelated_object(self):
        report = verify_rollout(self.graph, StrictContactOracle('UnrelatedBlock'), self.cfg,
                                outdir=self.output / 'unrelated')
        self.assertFalse(report['valid'])
        self.assertIsNotNone(report['first_violation'])

    def test_clear_effect_is_atomic_after_every_retreat_fragment(self):
        fragments = [node for node in self.graph['nodes'] if node['task_id'] == 'B_retreat']
        self.assertGreater(len(fragments), 1)
        for node in fragments:
            self.assertNotIn('tool_B', node['metadata'].get('effects', {}).get('support_contacts', {}))
        clearing = self.graph['task_map']['B_retreat_contact_clear']
        self.assertEqual(len(clearing['node_ids']), 1)
        self.assertEqual(clearing['metadata']['effects']['support_contacts'], {'tool_B': None})
        clear_node = next(node for node in self.graph['nodes'] if node['id'] == clearing['first_node'])
        self.assertGreater(clear_node['reference_index'], max(n['reference_index'] for n in fragments))


if __name__ == '__main__':
    unittest.main()
