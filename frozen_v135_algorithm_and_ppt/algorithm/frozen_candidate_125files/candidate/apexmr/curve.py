"""Pure-mathematics C2 joint-path candidate; NOT collision validated.

This laboratory module does not read or modify a simulator, planner, or scene.
Six-dimensional joint coordinates must already be consistently unwrapped.
The returned ``length`` is the ORIGINAL polyline's parameter length: a rounded
corner is assigned 2*r even though its true curve arc length is slightly less.

Before execution, every changed curve needs full robot/payload/world collision
checking, task-space clearance checks, regenerated TPG conflicts and milestones,
and dynamic verification. Mathematical limits alone are not a safety approval.
"""
from __future__ import annotations

import math
import numpy as np


RADIUS_CAP_RAD = .02
MIN_DURATION_S = .08


def _positive_vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(6, float(array))
    if array.shape != (6,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(name+' must be a positive finite scalar or six-vector')
    return array


def _order(value):
    if isinstance(value, bool) or value not in (0, 1, 2, 3):
        raise ValueError('derivative must be an integer from 0 through 3')
    return int(value)


def _derivative_controls(controls, length, derivative):
    values = np.asarray(controls, dtype=float)
    for degree in range(5, 5-derivative, -1):
        values = degree*np.diff(values, axis=0)/length
    return values


def _bezier(controls, fraction):
    values = np.asarray(controls, dtype=float).copy()
    for remaining in range(len(values)-1, 0, -1):
        values[:remaining] = ((1.-fraction)*values[:remaining]+
                              fraction*values[1:remaining+1])
    return values[0]


def _smoothstep(u, derivative=0):
    if derivative == 0:
        return u*u*u*(10.+u*(-15.+6.*u))
    if derivative == 1:
        return 30.*u*u*(1.-u)*(1.-u)
    if derivative == 2:
        return 60.*u*(1.-u)*(1.-2.*u)
    return 60.*(1.-6.*u+6.*u*u)


def _inverse_smoothstep(value):
    if value <= 0.:
        return 0.
    if value >= 1.:
        return 1.
    lower, upper = 0., 1.
    for _ in range(65):
        middle = (lower+upper)/2.
        if _smoothstep(middle) < value:
            lower = middle
        else:
            upper = middle
    return (lower+upper)/2.


def _scalar_timing_bounds(beginning, end):
    """Exact extrema of |f'|, |f''|, |f'''| on this phase interval."""
    critical = {
        1: (.5,),
        2: ((3.-math.sqrt(3.))/6., (3.+math.sqrt(3.))/6.),
        3: (.5,),
    }
    return [max(abs(_smoothstep(u, order)) for u in
                [beginning, end]+[v for v in critical[order] if beginning <= v <= end])
            for order in (1, 2, 3)]


def _timing(pieces, length, limits):
    numerators = np.zeros((3, 6))
    parameter_bounds = np.zeros((3, 6))
    for piece in pieces:
        if piece['length'] <= 0.:
            continue
        hull = np.asarray([np.max(abs(_derivative_controls(piece['controls'], piece['length'], order)), axis=0)
                           for order in (1, 2, 3)])
        parameter_bounds = np.maximum(parameter_bounds, hull)
        u0 = _inverse_smoothstep(piece['s0']/length)
        u1 = _inverse_smoothstep((piece['s0']+piece['length'])/length)
        b1, b2, b3 = np.asarray(_scalar_timing_bounds(u0, u1))*length
        m1, m2, m3 = hull
        # Chain rule; convex-hull derivative bounds do not depend on sampling.
        piece_numerators = np.asarray([
            m1*b1,
            m2*b1*b1+m1*b2,
            m3*b1**3+3.*m2*b1*b2+m1*b3,
        ])
        numerators = np.maximum(numerators, piece_numerators)
        piece['parameter_derivative_bounds'] = {
            key: row.tolist() for key,row in zip(('first', 'second', 'third'), hull)}
    duration = max(MIN_DURATION_S,
                   float(np.max(numerators[0]/limits[0])),
                   float(np.sqrt(np.max(numerators[1]/limits[1]))),
                   float(np.cbrt(np.max(numerators[2]/limits[2]))))
    # Small roundoff margin; this does not replace clearance or servo margins.
    duration *= 1.+1e-12
    if not math.isfinite(duration):
        raise ValueError('The requested path/limits produce an unrepresentable duration')
    bounds = numerators/np.asarray([duration, duration**2, duration**3])[:, None]
    return duration, bounds, parameter_bounds


def make_profile(knots, vmax, amax, jmax, blend_fraction=.35, max_angle_degrees=6):
    """Return a JSON-compatible, conservatively retimed C2 Bezier candidate.

    Sharp corners are rejected, never silently smoothed. The caller may split
    a macro at the reported corner and stop there. Scalar or per-joint limits
    are supported. ``derivative_bounds`` contains six-component time-derivative
    bounds, plus ``global_max`` for convenient reporting.

    The symmetric corner control points give q_s=u_in/u_out and q_ss=0 at
    both joins. Their maximum Euclidean distance from the replaced two line
    segments is exactly 3*r*sin(angle)/16 for the accepted small angles.
    This is a JOINT-SPACE deviation in radians, not a TCP clearance bound.
    """
    points = np.asarray(knots, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (6,) or len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError('knots must be a nonempty finite N by 6 array')
    if not math.isfinite(blend_fraction) or not 0. < blend_fraction <= .5:
        raise ValueError('blend_fraction must be in (0, .5] to prevent overlapping trims')
    if not math.isfinite(max_angle_degrees) or not 0. < max_angle_degrees < 90.:
        raise ValueError('max_angle_degrees must be in (0, 90)')
    limits = [_positive_vector(v, name) for v,name in
              ((vmax, 'vmax'), (amax, 'amax'), (jmax, 'jmax'))]
    original = points.copy()
    # Exact duplicate knots carry no geometric event in this math-only helper.
    # The application must separately retain semantic/skill boundaries.
    points = np.asarray([point for index,point in enumerate(points)
                         if index == 0 or not np.array_equal(point, points[index-1])])
    radii = np.zeros(len(points))
    angles = np.zeros(len(points))
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors/lengths[:, None] if len(lengths) else np.empty((0, 6))
    for i in range(1, len(points)-1):
        angle = 2.*math.asin(min(1., float(np.linalg.norm(directions[i]-directions[i-1]))/2.))
        angles[i] = angle
        if math.degrees(angle) > max_angle_degrees+1e-10:
            raise ValueError(f'Corner {i} is {math.degrees(angle):.9g} degrees; split/stop above {max_angle_degrees:g}')
        if angle > 1e-12:
            radii[i] = min(blend_fraction*lengths[i-1], blend_fraction*lengths[i], RADIUS_CAP_RAD)

    pieces, blends = [], []
    cursor = points[0].copy()
    clock = 0.

    def append_piece(controls, length, kind, corner=None):
        nonlocal clock
        piece = {'s0': float(clock), 'length': float(length),
                 'controls': np.asarray(controls).tolist(), 'kind': kind}
        if corner is not None:
            piece['corner_index'] = int(corner)
        pieces.append(piece)
        clock += float(length)

    def append_line(end):
        distance = float(np.linalg.norm(end-cursor))
        if distance > 0.:
            append_piece(np.linspace(cursor, end, 6), distance, 'line')

    for i in range(1, len(points)-1):
        r = float(radii[i])
        if r == 0.:
            continue
        u_prev, u_next = directions[i-1], directions[i]
        a, b = points[i]-r*u_prev, points[i]+r*u_next
        append_line(a)
        controls = [a, a+.4*r*u_prev, a+.8*r*u_prev,
                    b-.8*r*u_next, b-.4*r*u_next, b]
        blend = {'corner_index': i, 'radius_rad': r, 'angle_degrees': math.degrees(angles[i]),
                 's0': clock, 'length': 2.*r,
                 'max_joint_space_deviation_rad': 3.*r*math.sin(angles[i])/16.,
                 'replaced_polyline': [a.tolist(), points[i].tolist(), b.tolist()]}
        blends.append(blend)
        append_piece(controls, 2.*r, 'quintic_blend', i)
        cursor = b.copy()
    append_line(points[-1])
    if not pieces:
        append_piece([points[0]]*6, 0., 'stationary')
    duration, bounds, parameter_bounds = _timing(pieces, clock, limits)
    bound_dict = {key: values.tolist() for key,values in zip(('velocity', 'acceleration', 'jerk'), bounds)}
    bound_dict['global_max'] = {key: float(np.max(values)) for key,values in
                              zip(('velocity', 'acceleration', 'jerk'), bounds)}
    return {
        'schema': 'scene3_c2_joint_bezier_candidate_v1',
        'status': 'mathematical_candidate_not_collision_validated',
        'collision_validated': False,
        'pieces': pieces, 'length': clock, 'duration': duration,
        'length_kind': 'original_polyline_parameter_length_not_curve_arc_length',
        'derivative_bounds': bound_dict,
        'parameter_derivative_bounds': {key: row.tolist() for key,row in
                                        zip(('first', 'second', 'third'), parameter_bounds)},
        'bound_method': 'Bezier derivative convex hull and exact scalar timing extrema per piece; chain rule',
        'limits': {key: row.tolist() for key,row in zip(('velocity', 'acceleration', 'jerk'), limits)},
        'blend_radius': max((b['radius_rad'] for b in blends), default=0.),
        'blend_radius_cap_rad': RADIUS_CAP_RAD, 'blend_fraction': float(blend_fraction),
        'max_angle_degrees': float(max_angle_degrees), 'blends': blends,
        'max_joint_space_deviation_rad': max((b['max_joint_space_deviation_rad'] for b in blends), default=0.),
        'input_knots': original.tolist(), 'effective_knots': points.tolist(),
        'removed_duplicate_knots': len(original)-len(points),
        'timing_law': 's(t)=length*(10u^3-15u^4+6u^5), u=t/duration',
        'required_before_execution': ['full_robot_payload_world_collision_check',
                                      'TCP_clearance_and_task_constraints',
                                      'new_TPG_conflicts_and_progress_milestones',
                                      'measured_dynamic_and_force_validation'],
    }


def eval_at_s(profile, s, derivative=0):
    """Evaluate q or its first three derivatives with respect to path s."""
    derivative = _order(derivative)
    s = float(s)
    if not math.isfinite(s):
        raise ValueError('s must be finite')
    length = float(profile['length'])
    if derivative and (s < 0. or s > length or length <= 0.):
        return [0.]*6
    s = min(length, max(0., s))
    pieces = profile['pieces']
    piece = next((p for p in pieces if s < p['s0']+p['length']), pieces[-1])
    if piece['length'] <= 0.:
        return list(piece['controls'][0]) if derivative == 0 else [0.]*6
    fraction = min(1., max(0., (s-piece['s0'])/piece['length']))
    controls = _derivative_controls(piece['controls'], piece['length'], derivative)
    return _bezier(controls, fraction).tolist()


def eval_profile(profile, elapsed, derivative=0):
    """Evaluate position, velocity, acceleration, or jerk in real seconds.

    Outside [0,duration], position is held and derivatives are zero. At the
    endpoints the interior jerk is returned; C2 does not imply continuous jerk.
    """
    derivative = _order(derivative)
    elapsed = float(elapsed)
    if not math.isfinite(elapsed):
        raise ValueError('elapsed must be finite')
    duration, length = float(profile['duration']), float(profile['length'])
    if derivative and (elapsed < 0. or elapsed > duration or length <= 0.):
        return [0.]*6
    u = min(1., max(0., elapsed/duration))
    s = length*_smoothstep(u)
    if derivative == 0:
        return eval_at_s(profile, s)
    q1 = np.asarray(eval_at_s(profile, s, 1))
    sd = length*_smoothstep(u, 1)/duration
    if derivative == 1:
        return (q1*sd).tolist()
    q2 = np.asarray(eval_at_s(profile, s, 2))
    sdd = length*_smoothstep(u, 2)/duration**2
    if derivative == 2:
        return (q2*sd**2+q1*sdd).tolist()
    q3 = np.asarray(eval_at_s(profile, s, 3))
    sddd = length*_smoothstep(u, 3)/duration**3
    return (q3*sd**3+3.*q2*sd*sdd+q1*sddd).tolist()


__all__ = ['make_profile', 'eval_profile', 'eval_at_s']
