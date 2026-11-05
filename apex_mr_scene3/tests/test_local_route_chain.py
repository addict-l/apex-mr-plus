from copy import deepcopy
import math
import unittest

from apex.local_route_chain import base_travel, replace_window


def q(x):
    return [x, 0., 0., 0., 0., 0.]


def fixture():
    result = []
    for name, kind, path in [('fetch', 'transit', [q(0), q(4)]),
                             ('grasp', 'grasp', [q(4), q(4)]),
                             ('carry', 'transit', [q(4), q(.4)]),
                             ('later', 'transit', [q(.4), q(.8)])]:
        result.append({'id': name, 'arm': 'B', 'kind': kind, 'q_path': path, 'duration': 1.,
                       'payload': {'part': 'test'} if name == 'carry' else None,
                       'dependencies': [result[-1]['id']] if result else [],
                       'metadata': {'effects': {'unchanged': name}, 'contact_exclusions': [['a', 'b']]}})
    return result


class LocalRouteChainTests(unittest.TestCase):
    def test_closed_short_candidate_preserves_semantics_and_never_mutates_input(self):
        original = fixture(); frozen = deepcopy(original)
        candidate, evidence = replace_window(original, 'fetch', 'carry', {
            'fetch': [q(0), q(-.6)], 'carry': [q(-.6), q(.4)]})
        self.assertEqual(original, frozen)
        self.assertEqual(candidate[1]['q_path'], [q(-.6), q(-.6)])
        self.assertEqual(candidate[-1], original[-1])
        for a, b in zip(original, candidate):
            self.assertEqual({k: v for k, v in a.items() if k not in ('q_path', 'duration')},
                             {k: v for k, v in b.items() if k not in ('q_path', 'duration')})
        self.assertEqual(evidence['shifted_skill_postures'], ['grasp'])
        self.assertAlmostEqual(evidence['after_base_travel_deg'], math.degrees(1.6))

    def test_open_window_and_outside_edit_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'final anchor'):
            replace_window(fixture(), 'fetch', 'carry', {'fetch': [q(0), q(-.6)], 'carry': [q(-.6), q(.5)]})
        with self.assertRaisesRegex(ValueError, 'within the window'):
            replace_window(fixture(), 'fetch', 'carry', {'later': [q(.4), q(.5)]})

    def test_long_turn_is_not_hidden_by_winding_or_short_net_displacement(self):
        self.assertAlmostEqual(base_travel([q(0), q(4)]), 4)
        with self.assertRaisesRegex(ValueError, 'above threshold'):
            replace_window(fixture(), 'fetch', 'carry', {'fetch': [q(0), q(3.2)], 'carry': [q(3.2), q(.4)]})
        with self.assertRaisesRegex(ValueError, 'above threshold'):
            replace_window(fixture(), 'fetch', 'carry', {'fetch': [q(0), q(2), q(0)], 'carry': [q(0), q(.4)]})

    def test_unreconnected_next_transit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'open at carry'):
            replace_window(fixture(), 'fetch', 'carry', {'fetch': [q(0), q(-.6)]})


if __name__ == '__main__':
    unittest.main()
