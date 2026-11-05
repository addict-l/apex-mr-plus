"""Validate a proposed C2 macro with a caller-owned, real planning connection.

This module never starts a simulator and never changes production source or
acceptance artifacts. Call ``validate_profile`` with the actual MotionPlanner;
the returned evidence is a fixed-other-arm, sampled geometry/corridor check,
not a substitute for the rebuilt TPG, complete rollout or dynamic execution.

Profile schema: ``pieces=[{s0, length, controls: [six 6D Bezier points]}]``,
``length``. The evaluator has signature ``eval_at_s(profile, s, derivative=0)``.
To avoid silently substituting the other arm's home configuration, provide
``cfg['validation_state']={'q': {'A': ..., 'B': ...},
                          'payloads': {'A': None, 'B': ...}}``
reconstructed immediately before this reference macro. Alternatively every
node may contain complete ``metadata.options.other_q/other_payload`` values.
"""
from copy import deepcopy
import json
import math
import re


KNOWN_MACROS = frozenset((
    'support_slide_in', 'A_exit_from_support', 'install_K1_slide',
    'extract_support_motion', 'close_gate_slide', 'close_gate_retreat',
))
# Same semantic comparison as the frozen apex.program._context. Provenance
# fields not ignored by production remain significant here too.
CONTEXT_IGNORED = frozenset((
    'effects', 'planner', 'path_source', 'planning_stats', 'motion_run_id',
    'motion_run_offset', 'motion_run_duration', 'motion_run_members',
    'continuation_from', 'continuation_to', 'performance_boundary',
))


class Rejected(ValueError):
    def __init__(self, reason, **details):
        self.reason, self.details = reason, details
        super().__init__(reason)


def _norm(vector):
    return math.sqrt(sum(x*x for x in vector))


def _delta(a, b):
    return [x-y for x, y in zip(a, b)]


def _vector(value, size, label):
    try:
        result = [float(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise Rejected('invalid_vector', field=label) from exc
    if len(result) != size or not all(math.isfinite(x) for x in result):
        raise Rejected('invalid_vector', field=label)
    return result


def _positive(value, label, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise Rejected('invalid_bound', field=label, value=value)
    return value


def _context(node):
    metadata = node.get('metadata', {})
    return json.dumps({
        'arm': node['arm'], 'payload': node.get('payload'), 'mode': node.get('mode', {}),
        'metadata': {k: v for k, v in metadata.items() if k not in CONTEXT_IGNORED},
    }, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _reference_nodes(nodes):
    if not nodes:
        raise Rejected('empty_macro')
    context = _context(nodes[0])
    arm = nodes[0]['arm']
    if arm not in ('A', 'B'):
        raise Rejected('invalid_arm')
    original, task_ids, macro = [], [], None
    for index, node in enumerate(nodes):
        task_id = node.get('task_id', node.get('id', '')).split('::')[0]
        match = re.fullmatch(r'(.+)_(\d+)', task_id)
        name = match.group(1) if match else None
        if name not in KNOWN_MACROS or (macro is not None and name != macro):
            raise Rejected('not_one_known_cartesian_macro', task_id=task_id)
        macro = name
        if node.get('node_type', 'pose') != 'pose' or node.get('kind', 'transit') != 'transit':
            raise Rejected('skill_boundary_inside_macro', index=index)
        if _context(node) != context:
            raise Rejected('context_changes_inside_macro', index=index, task_id=task_id)
        if node.get('metadata', {}).get('effects'):
            raise Rejected('mode_effect_inside_macro', index=index, task_id=task_id)
        raw = node.get('q_samples', node.get('q_path'))
        if not raw:
            raise Rejected('missing_reference_geometry', task_id=task_id)
        path = [_vector(q, 6, 'reference_q') for q in raw]
        if original and _norm(_delta(path[0], original[-1])) > 1e-8:
            raise Rejected('discontinuous_reference_geometry', task_id=task_id)
        for q in path:
            if not original or _norm(_delta(q, original[-1])) > 1e-12:
                original.append(q)
        if not task_ids or task_ids[-1] != task_id:
            if task_ids and int(task_id.rsplit('_', 1)[1]) != int(task_ids[-1].rsplit('_', 1)[1])+1:
                raise Rejected('nonconsecutive_macro_tasks', task_id=task_id)
            task_ids.append(task_id)
    if len(original) < 2:
        raise Rejected('zero_length_reference_geometry')
    return arm, macro, task_ids, original


def _curve_samples(profile, evaluator, step_rad, max_samples):
    pieces = profile.get('pieces', [])
    length = _positive(profile.get('length'), 'profile.length')
    if not pieces:
        raise Rejected('empty_profile')
    parameters, endpoints, evidence = [], [], []
    cursor = 0.
    for index, piece in enumerate(pieces):
        start = _positive(piece.get('s0'), 'piece.s0', allow_zero=True)
        span = _positive(piece.get('length'), 'piece.length')
        if abs(start-cursor) > 1e-9:
            raise Rejected('noncontiguous_profile_pieces', piece=index)
        controls = [_vector(q, 6, 'Bezier control') for q in piece.get('controls', [])]
        if len(controls) != 6:
            raise Rejected('expected_quintic_Bezier_controls', piece=index)
        edges = [_norm(_delta(b, a)) for a, b in zip(controls, controls[1:])]
        polygon_length = sum(edges)
        # Uniform parameter sampling at merely polygon_length/step does not
        # bound every gap when controls are uneven. The Bezier derivative is
        # a convex combination of 5*(P[i+1]-P[i]); this stronger bound does.
        divisions = max(1, math.ceil(polygon_length/step_rad),
                        math.ceil(5*max(edges, default=0.)/step_rad))
        if len(parameters)+divisions+1 > max_samples:
            raise Rejected('geometry_sample_budget_exceeded', piece=index,
                           maximum=max_samples, required_at_least=len(parameters)+divisions+1)
        parameters.extend(start+span*k/divisions for k in range(0 if index == 0 else 1, divisions+1))
        endpoints.extend((start, start+span))
        evidence.append({'index': index, 's0': start, 'length': span,
                         'control_polygon_length_rad': polygon_length,
                         'samples': divisions+1,
                         'joint_arclength_gap_bound_rad': 5*max(edges, default=0.)/divisions})
        cursor = start+span
    if abs(cursor-length) > 1e-9:
        raise Rejected('profile_length_mismatch', pieces_end=cursor, length=length)
    samples = [_vector(evaluator(profile, s), 6, 'evaluated_curve_q') for s in parameters]
    # Verify the evaluator and controls agree at every piece boundary. The
    # derivative sampling bound only applies to the supplied Bezier geometry.
    for index, piece in enumerate(pieces):
        for s, expected in ((piece['s0'], piece['controls'][0]),
                            (piece['s0']+piece['length'], piece['controls'][-1])):
            actual = _vector(evaluator(profile, s), 6, 'evaluated_endpoint_q')
            if _norm(_delta(actual, expected)) > 1e-8:
                raise Rejected('evaluator_control_endpoint_mismatch', piece=index, s=s)
    piece_index = 0
    for s, actual in zip(parameters, samples):
        while piece_index+1 < len(pieces) and s > pieces[piece_index]['s0']+pieces[piece_index]['length']+1e-12:
            piece_index += 1
        piece = pieces[piece_index]
        u = max(0., min(1., (s-piece['s0'])/piece['length']))
        expected = [sum(math.comb(5, k)*(1-u)**(5-k)*u**k*piece['controls'][k][j]
                        for k in range(6)) for j in range(6)]
        if _norm(_delta(actual, expected)) > 1e-8:
            raise Rejected('evaluator_control_interior_mismatch', piece=piece_index, s=s)
    return parameters, samples, evidence, sorted(set(endpoints))


def _dense_polyline(path, step):
    result = [path[0]]
    for a, b in zip(path, path[1:]):
        count = max(1, math.ceil(_norm(_delta(b, a))/step))
        result.extend([[x+(y-x)*k/count for x, y in zip(a, b)] for k in range(1, count+1)])
    return result


def _quat(value):
    value = _vector(value, 4, 'FK quaternion')
    norm = _norm(value)
    if norm < 1e-12:
        raise Rejected('zero_FK_quaternion')
    return [x/norm for x in value]


def _slerp(a, b, fraction):
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0:
        b, dot = [-x for x in b], -dot
    dot = max(-1., min(1., dot))
    if dot > .9995:
        return _quat([x+(y-x)*fraction for x, y in zip(a, b)])
    angle = math.acos(dot)
    left = math.sin((1-fraction)*angle)/math.sin(angle)
    right = math.sin(fraction*angle)/math.sin(angle)
    return [left*x+right*y for x, y in zip(a, b)]


def _nearest_corridor(pose, reference, lengths):
    best = None
    for index, (a, b) in enumerate(zip(reference, reference[1:])):
        direction = _delta(b[:3], a[:3])
        denominator = sum(x*x for x in direction)
        fraction = (sum(x*y for x, y in zip(_delta(pose[:3], a[:3]), direction))/denominator
                    if denominator > 1e-20 else 0.)
        fraction = max(0., min(1., fraction))
        closest = [x+fraction*y for x, y in zip(a[:3], direction)]
        distance = _norm(_delta(pose[:3], closest))
        if best is None or distance < best[0]:
            reference_orientation = _slerp(a[3:], b[3:], fraction)
            angle = 2*math.acos(min(1., abs(sum(x*y for x, y in zip(pose[3:], reference_orientation)))))
            progress = lengths[index]+fraction*math.sqrt(denominator)
            best = (distance, angle, progress, index, fraction)
    return best


def validate_profile(planner, cfg, nodes, profile, evaluator, step_rad=.003,
                     max_tcp_deviation_m=.0005, max_orientation_error_rad=.005):
    """Return acceptance/rejection evidence; perform no file or simulator setup.

    Geometry checks use every sampled curve configuration and exact original
    context. FK checks also use every curve sample, plus a dense version of the
    original joint polyline. No failed/unsupported check is treated as success.
    """
    report = {'accepted': False, 'valid': False, 'geometry_valid': False,
              'tcp_corridor_valid': False, 'continuous_collision_guarantee': False,
              'scope': 'sampled curved macro against fixed reference other-arm state; original FK-polyline corridor',
              'geometry_samples_checked': 0, 'geometry_samples_submitted': 0, 'fk_calls': 0,
              'validation_performed': False}
    try:
        step_rad = _positive(step_rad, 'step_rad')
        position_limit = _positive(max_tcp_deviation_m, 'max_tcp_deviation_m')
        angle_limit = _positive(max_orientation_error_rad, 'max_orientation_error_rad')
        limits = cfg.get('blend_validation', {})
        max_samples = int(limits.get('max_samples', 50000))
        batch_size = int(limits.get('batch_size', 128))
        progress_tolerance = _positive(limits.get('progress_tolerance_m', 1e-6),
                                       'progress_tolerance_m', allow_zero=True)
        if max_samples < 2 or batch_size < 1:
            raise Rejected('invalid_sampling_budget')
        report['thresholds'] = {'step_rad': step_rad, 'max_tcp_deviation_m': position_limit,
                                'max_orientation_error_rad': angle_limit,
                                'progress_tolerance_m': progress_tolerance}
        arm, macro, task_ids, original = _reference_nodes(nodes)
        other = 'B' if arm == 'A' else 'A'
        metadata = deepcopy(nodes[0].get('metadata', {}))
        options = deepcopy(metadata.get('options', {}))
        state = cfg.get('validation_state', {})
        state_q = state.get('q', {})
        state_payloads = state.get('payloads', {})
        if other in state_q and other in state_payloads:
            other_q, other_payload = state_q[other], state_payloads[other]
        elif 'other_q' in options and 'other_payload' in options:
            other_q, other_payload = options['other_q'], options['other_payload']
        else:
            raise Rejected('missing_reference_other_arm_state', required_arm=other)
        other_q = _vector(other_q, 6, 'reference_other_arm_q')
        if 'other_q' in options and _norm(_delta(other_q, _vector(options['other_q'], 6, 'options.other_q'))) > 1e-8:
            raise Rejected('conflicting_other_arm_state')
        if 'other_payload' in options and (options['other_payload'] or None) != (other_payload or None):
            raise Rejected('conflicting_other_arm_payload')
        if arm in state_q and _norm(_delta(_vector(state_q[arm], 6, 'reference_active_q'), original[0])) > 1e-8:
            raise Rejected('active_arm_reference_start_mismatch')
        payloads = {arm: deepcopy(nodes[0].get('payload')), other: deepcopy(other_payload)}
        if arm in state_payloads and (state_payloads[arm] or None) != (payloads[arm] or None):
            raise Rejected('active_arm_reference_payload_mismatch')
        placed = deepcopy(metadata.get('placed'))
        if not isinstance(placed, dict):
            raise Rejected('missing_placed_snapshot')
        contacts = deepcopy(metadata.get('contact_exclusions', []))
        if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in contacts):
            raise Rejected('invalid_exact_contact_pairs')
        if 'contact_exclusions' in options and options['contact_exclusions'] != contacts:
            raise Rejected('conflicting_contact_permissions')
        options['contact_exclusions'] = contacts
        # Match the compiler's declared physical contact tolerance, never widen
        # it to make a blend pass. The historical tasks omit this constant from
        # metadata.options; their metadata value takes precedence when present.
        declared_tolerance = metadata.get('contact_tolerance', options.get('contact_tolerance', .002))
        options['contact_tolerance'] = _positive(declared_tolerance, 'contact_tolerance')
        if 'articulations' not in options:
            raise Rejected('missing_articulation_snapshot')
        report.update(macro=macro, arm=arm, task_ids=task_ids, context_unchanged=True,
                      context={'payloads': payloads, 'placed': placed, 'options': options,
                               'other_arm': other, 'other_q': other_q})
        parameters, curve, piece_evidence, piece_ends = _curve_samples(profile, evaluator, step_rad, max_samples)
        if _norm(_delta(curve[0], original[0])) > 1e-8 or _norm(_delta(curve[-1], original[-1])) > 1e-8:
            raise Rejected('profile_changes_macro_endpoints')
        reference_q = _dense_polyline(original, step_rad)
        if len(reference_q) > max_samples:
            raise Rejected('reference_sample_budget_exceeded', required=len(reference_q), maximum=max_samples)
        report.update(curve_sample_count=len(curve), reference_sample_count=len(reference_q),
                      piece_sampling=piece_evidence, included_piece_endpoints=piece_ends,
                      fk_stride=1, max_sample_joint_gap_rad=max(
                          (_norm(_delta(b, a)) for a, b in zip(curve, curve[1:])), default=0.))
        validate = getattr(planner, 'validate_samples', None) or getattr(planner, 'sample_state_pairs', None)
        if not callable(validate) or not callable(getattr(planner, 'fk', None)):
            raise Rejected('planner_missing_required_validation_API')
        report['collision_api'] = 'validate_samples' if callable(getattr(planner, 'validate_samples', None)) else 'sample_state_pairs'
        report['planner_type'] = type(planner).__module__+'.'+type(planner).__name__
        report['collision_geometry_policy'] = getattr(planner, 'info', {}).get('collision_geometry_policy')
        for offset in range(0, len(curve), batch_size):
            batch = [{arm: q, other: other_q, 'payloads': deepcopy(payloads),
                      'placed': deepcopy(placed), 'options': deepcopy(options)}
                     for q in curve[offset:offset+batch_size]]
            answers = validate(batch)
            report['validation_performed'] = True
            report['geometry_samples_submitted'] += len(batch)
            if not isinstance(answers, (list, tuple)) or len(answers) != len(batch):
                raise Rejected('invalid_geometry_API_response', batch_start=offset)
            for index, answer in enumerate(answers):
                report['geometry_samples_checked'] += 1
                if not isinstance(answer, dict) or answer.get('ok') is not True:
                    raise Rejected('curve_collision', sample=offset+index,
                                   s=parameters[offset+index], evidence=answer)
        report['geometry_valid'] = True
        cache = {}

        def fk(q):
            key = tuple(q)  # No rounding away a changed configuration.
            if key not in cache:
                answer = planner.fk(arm, q)
                report['fk_calls'] += 1
                if not isinstance(answer, dict) or answer.get('ok') is not True:
                    raise Rejected('FK_API_failure', evidence=answer)
                pose = _vector(answer.get('pose'), 7, 'FK pose')
                pose[3:] = _quat(pose[3:])
                cache[key] = pose
            return cache[key]

        reference = [fk(q) for q in reference_q]
        lengths = [0.]
        for a, b in zip(reference, reference[1:]):
            lengths.append(lengths[-1]+_norm(_delta(b[:3], a[:3])))
        if lengths[-1] < 1e-8:
            raise Rejected('zero_length_TCP_reference_corridor')
        max_distance, max_angle, max_regression, furthest = 0., 0., 0., None
        worst_distance, worst_angle = None, None
        for index, (s, q) in enumerate(zip(parameters, curve)):
            distance, angle, progress, segment, fraction = _nearest_corridor(fk(q), reference, lengths)
            if distance >= max_distance:
                max_distance, worst_distance = distance, {'sample': index, 's': s, 'reference_segment': segment}
            if angle >= max_angle:
                max_angle, worst_angle = angle, {'sample': index, 's': s, 'reference_segment': segment}
            regression = max(0., furthest-progress) if furthest is not None else 0.
            max_regression = max(max_regression, regression)
            if distance > position_limit or angle > angle_limit or regression > progress_tolerance:
                raise Rejected('TCP_corridor_violation', sample=index, s=s,
                               position_deviation_m=distance, orientation_error_rad=angle,
                               progress_m=progress, regression_m=regression,
                               reference_segment=segment, reference_fraction=fraction)
            furthest = max(furthest, progress) if furthest is not None else progress
        report.update(accepted=True, valid=True, tcp_corridor_valid=True, reason='accepted_sampled_macro',
                      max_tcp_deviation_m=max_distance, max_orientation_error_rad=max_angle,
                      max_progress_regression_m=max_regression, reference_TCP_length_m=lengths[-1],
                      worst_position_sample=worst_distance, worst_orientation_sample=worst_angle,
                      progress_monotone_within_tolerance=True)
    except Rejected as exc:
        report.update(reason=exc.reason, rejection=exc.details)
    except Exception as exc:
        # Fail closed on RPC, evaluator or malformed input errors. Do not claim
        # a scene collision when the actual validation API did not complete.
        report.update(reason='validation_exception', rejection={
            'type': type(exc).__name__, 'message': str(exc)})
    return report

