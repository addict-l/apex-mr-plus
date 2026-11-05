"""Mathematical regressions only; no robot geometry or simulation is loaded."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blending_draft import make_profile, eval_profile, eval_at_s


def corner(angle=5., radius=.08):
    a = np.zeros(6)
    center = a.copy(); center[0] = radius
    end = center.copy()
    end[:2] += radius*np.asarray([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    return [a.tolist(), center.tolist(), end.tolist()]


def distance_to_segment(points, a, b):
    points, a, b = np.asarray(points), np.asarray(a), np.asarray(b)
    direction = b-a
    fractions = np.clip((points-a)@direction/(direction@direction), 0., 1.)
    closest = a+fractions[:, None]*direction
    return np.linalg.norm(points-closest, axis=1)


class BlendingDraftTests(unittest.TestCase):
    def test_macro_endpoints_and_all_piece_joins_are_c2(self):
        knots = corner()
        profile = make_profile(knots, .6, 1.2, 8.)
        np.testing.assert_array_equal(eval_profile(profile, 0.), knots[0])
        np.testing.assert_array_equal(eval_profile(profile, profile['duration']), knots[-1])
        for left, right in zip(profile['pieces'], profile['pieces'][1:]):
            a, b = np.asarray(left['controls']), np.asarray(right['controls'])
            la, lb = left['length'], right['length']
            np.testing.assert_allclose(a[-1], b[0], atol=1e-14, rtol=0.)
            np.testing.assert_allclose(5*(a[-1]-a[-2])/la, 5*(b[1]-b[0])/lb, atol=1e-11, rtol=0.)
            np.testing.assert_allclose(20*(a[-1]-2*a[-2]+a[-3])/la**2,
                                       20*(b[2]-2*b[1]+b[0])/lb**2, atol=1e-8, rtol=0.)
        for time in (0., profile['duration']):
            np.testing.assert_allclose(eval_profile(profile, time, 1), 0., atol=1e-14)
            np.testing.assert_allclose(eval_profile(profile, time, 2), 0., atol=1e-14)

    def test_actual_derivatives_stay_below_analytic_bounds_and_joint_limits(self):
        generator = np.random.default_rng(20)
        direction = generator.normal(size=6); direction /= np.linalg.norm(direction)
        points = [np.zeros(6), direction*.055]
        for index in range(5):
            normal = generator.normal(size=6)
            normal -= (normal@direction)*direction; normal /= np.linalg.norm(normal)
            angle = math.radians(2.+index*.7)
            direction = math.cos(angle)*direction+math.sin(angle)*normal
            points.append(points[-1]+(.035+index*.004)*direction)
        limits = [np.linspace(.15, .4, 6), np.linspace(.3, .8, 6), np.linspace(.9, 3., 6)]
        profile = make_profile(points, *limits)
        for order,key,limit in zip((1,2,3), ('velocity','acceleration','jerk'), limits):
            bound = np.asarray(profile['derivative_bounds'][key])
            values = np.asarray([eval_profile(profile, time, order)
                                 for time in np.linspace(0., profile['duration'], 1501)])
            self.assertTrue(np.all(np.max(abs(values), axis=0) <= bound+1e-10))
            self.assertTrue(np.all(bound <= limit+1e-10))

    def test_time_derivatives_match_independent_finite_differences(self):
        profile = make_profile(corner(), .6, 1.2, 8.)
        h = profile['duration']*1e-6
        for fraction in (.19, .43, .5, .61, .83):
            time = fraction*profile['duration']
            for order in (1, 2, 3):
                numerical = (np.asarray(eval_profile(profile, time+h, order-1))-
                             np.asarray(eval_profile(profile, time-h, order-1)))/(2*h)
                np.testing.assert_allclose(eval_profile(profile, time, order), numerical,
                                           atol=2e-7, rtol=2e-6)

    def test_exact_joint_space_maximum_deviation_matches_geometry(self):
        for angle in (.1, 3., 6.):
            with self.subTest(angle=angle):
                profile = make_profile(corner(angle), .6, 1.2, 8.)
                blend = profile['blends'][0]
                points = np.asarray([eval_at_s(profile, blend['s0']+u*blend['length'])
                                     for u in np.linspace(0., 1., 1001)])
                a, center, b = blend['replaced_polyline']
                distances = np.minimum(distance_to_segment(points, a, center),
                                       distance_to_segment(points, center, b))
                self.assertAlmostEqual(float(np.max(distances)),
                                       blend['max_joint_space_deviation_rad'], delta=2e-13)

    def test_collinear_knots_reduce_to_unchanged_line(self):
        knots = [np.zeros(6), np.array([.1, 0, 0, 0, 0, 0]), np.array([.2, 0, 0, 0, 0, 0])]
        profile = make_profile(knots, .6, 1.2, 8.)
        self.assertEqual(profile['blends'], [])
        self.assertEqual(len(profile['pieces']), 1)
        self.assertEqual(profile['max_joint_space_deviation_rad'], 0.)
        for s in np.linspace(0., profile['length'], 21):
            np.testing.assert_allclose(eval_at_s(profile, s), [s,0,0,0,0,0], atol=1e-14)

    def test_radius_cap_nonoverlap_and_length_definition(self):
        profile = make_profile(corner(radius=.2), .6, 1.2, 8.)
        self.assertEqual(profile['blend_radius'], .02)
        expected_length = sum(np.linalg.norm(np.asarray(a)-b) for a,b in zip(profile['input_knots'], profile['input_knots'][1:]))
        self.assertAlmostEqual(profile['length'], expected_length, places=12)
        for left,right in zip(profile['pieces'], profile['pieces'][1:]):
            self.assertAlmostEqual(left['s0']+left['length'], right['s0'], places=12)
        self.assertIn('not_curve_arc_length', profile['length_kind'])

    def test_rejects_sharp_corner_and_invalid_values(self):
        for angle in (6.001, 90., 180.):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                make_profile(corner(angle), .6, 1.2, 8.)
        for kwargs in ({'blend_fraction': .51}, {'blend_fraction': 0.}, {'max_angle_degrees': math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_profile(corner(), .6, 1.2, 8., **kwargs)
        for limit in (0., -1., math.inf, math.nan):
            with self.subTest(jmax=limit), self.assertRaises(ValueError):
                make_profile(corner(), .6, 1.2, limit)

    def test_json_schema_marks_candidate_unvalidated_and_handles_stationary_path(self):
        profile = make_profile([[1.]*6, [1.]*6], .6, 1.2, 8.)
        encoded = json.dumps(profile, allow_nan=False)
        restored = json.loads(encoded)
        self.assertFalse(restored['collision_validated'])
        self.assertIn('not_collision_validated', restored['status'])
        self.assertEqual(restored['removed_duplicate_knots'], 1)
        for time in (-1., 0., profile['duration'], profile['duration']+1.):
            self.assertEqual(eval_profile(restored, time), [1.]*6)
            for order in (1,2,3):
                self.assertEqual(eval_profile(restored, time, order), [0.]*6)


if __name__ == '__main__':
    unittest.main()
