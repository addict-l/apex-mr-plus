"""Validate small C2 corners of continuous Cartesian macros before TPG checks."""
from copy import deepcopy
import math
import re
import numpy as np
from .curve import make_profile, eval_at_s
from .blend_validation import validate_profile, KNOWN_MACROS, _context, local_route_group
from .program import compress_collinear, _limits, _duration


def _macro(node):
    if 'local_route_candidate' in node.get('metadata', {}):
        return local_route_group(node)
    match = re.fullmatch(r'(.+)_(\d+)', node.get('task_id', ''))
    return match.group(1) if match and match.group(1) in KNOWN_MACROS else None


def _length(path):
    return sum(float(np.linalg.norm(np.asarray(b)-a)) for a, b in zip(path, path[1:]))


def smooth_cartesian_macros(tpg, planner, cfg, vmax=.6, amax=1.2, jmax=8.):
    """Mutate only fully checked candidate macros; keep every semantic node.

    All original contexts, task/skill ids and graph edges stay in place. Small
    interior geometric milestones move onto the accepted curve. Macro first
    and last configurations remain exact. New q_samples are consumed by the
    subsequent full-phase inter-arm oracle and nominal rollout.
    """
    nodes = tpg['nodes']
    q = {a: list(cfg['arms'][a]['home_q']) for a in ('A', 'B')}
    held = deepcopy(cfg.get('initial_payloads', {}))
    report = {'candidate_macros': [], 'accepted_macros': 0, 'accepted_cartesian_macros': 0,
              'accepted_local_route_tasks': 0, 'blended_corners': 0,
              'max_tcp_deviation_m': 0., 'max_orientation_error_rad': 0.,
              'collision_sampling_step_rad': .003,
              'tcp_corridor_limit_m': .0005, 'orientation_limit_rad': .005,
              'continuous_collision_guarantee': False}
    index = 0
    while index < len(nodes):
        first = nodes[index]
        name = _macro(first)
        local_route = local_route_group(first)
        end = index+1
        group = [first]
        if name and first['node_type'] == 'pose':
            while end < len(nodes) and (nodes[end].get('task_id') == local_route if local_route
                                        else _macro(nodes[end]) == name):
                group.append(nodes[end]); end += 1
        if name and (len(group) > 1 or local_route):
            record = {'macro': name, 'arm': first['arm'], 'nodes': len(group),
                      'group_kind': 'local_route_task' if local_route else 'cartesian_macro',
                      'task_ids': list(dict.fromkeys(n['task_id'] for n in group)),
                      'attempts': [], 'accepted': False}
            report['candidate_macros'].append(record)
            if local_route and [n['id'] for n in group] != [n['id'] for n in nodes if n.get('task_id') == local_route]:
                record['reason'] = 'local_route_does_not_cover_complete_task'
            elif any(_context(n) != _context(first) or n['metadata'].get('effects') for n in group):
                record['reason'] = 'semantic_or_contact_context_changes'
            else:
                path = [group[0]['q_samples'][0]]
                for n in group:
                    path.extend(n['q_samples'][1:])
                knots = compress_collinear(path)
                lv, la, lj = _limits(first, vmax, amax, jmax)
                reference_motion_time = sum(_duration(a, b, lv, la, lj) for a, b in zip(knots, knots[1:])
                                            if np.max(abs(np.asarray(a)-b)) > 1e-9)
                record.update(reference_rest_to_rest_motion_duration_s=reference_motion_time,
                              timing_scope='same path knots and joint limits; excludes physical dwell, servo settling and scheduling')
                state_cfg = {**cfg, 'validation_state': {'q': deepcopy(q),
                    'payloads': {arm: deepcopy(held.get(arm)) for arm in ('A', 'B')}}}
                accepted = None
                for radius in (.35, .20, .10):
                    try:
                        profile = make_profile(knots, lv, la, lj, blend_fraction=radius)
                    except ValueError as exc:
                        record['reason'] = str(exc); break
                    if local_route and not profile['blends']:
                        record['reason'] = 'no_small_angle_corner_to_blend'; break
                    timing_ok = profile['duration'] <= reference_motion_time+1e-9
                    if local_route and not timing_ok:
                        record['attempts'].append({'blend_fraction': radius, 'duration_s': profile['duration'],
                                                   'timing_non_regression': False,
                                                   'validation': {'accepted': False, 'validation_performed': False,
                                                                  'reason': 'profile_slower_than_original_polyline_timing'}})
                        record['reason'] = 'profile_slower_than_original_polyline_timing'
                        continue
                    proof = validate_profile(planner, state_cfg, group, profile, eval_at_s)
                    record['attempts'].append({'blend_fraction': radius, 'validation': proof,
                                               'timing_non_regression': timing_ok,
                                               'duration_s': profile['duration']})
                    if proof['accepted']:
                        accepted = (profile, proof)
                        break
                if accepted:
                    profile, proof = accepted
                    profile['collision_validated'] = True
                    profile['status'] = 'sampled_reference_geometry_and_TCP_corridor_validated'
                    cursor = 0.
                    for n in group:
                        source = deepcopy(n['q_samples'])
                        finish = cursor+_length(source)
                        if n is group[-1]:
                            finish = profile['length']
                        stamps = {cursor, finish}
                        count = max(1, math.ceil((finish-cursor)/.003))
                        stamps.update(cursor+(finish-cursor)*i/count for i in range(count+1))
                        for piece in profile['pieces']:
                            for s in (piece['s0'], piece['s0']+piece['length']):
                                if cursor < s < finish: stamps.add(s)
                        n['original_q_samples'] = source
                        n['q_samples'] = [eval_at_s(profile, s) for s in sorted(stamps)]
                        n['q_start'], n['q_end'] = n['q_samples'][0], n['q_samples'][-1]
                        n['curve_profile'] = profile
                        n['curve_range_s'] = [cursor, finish]
                        n['curve_run_id'] = first['id']
                        cursor = finish
                    record.update(accepted=True, duration_s=profile['duration'],
                                  blended_corners=len(profile['blends']),
                                  max_joint_space_deviation_rad=profile['max_joint_space_deviation_rad'])
                    report['accepted_macros'] += 1
                    report['accepted_local_route_tasks' if local_route else 'accepted_cartesian_macros'] += 1
                    report['blended_corners'] += len(profile['blends'])
                    report['max_tcp_deviation_m'] = max(report['max_tcp_deviation_m'], proof.get('max_tcp_deviation_m', 0.))
                    report['max_orientation_error_rad'] = max(report['max_orientation_error_rad'], proof.get('max_orientation_error_rad', 0.))
                else:
                    record.setdefault('reason', 'candidate_geometry_or_TCP_corridor_rejected')
        # Advance the reference world, never the simulator's live robot state.
        for n in group:
            q[n['arm']] = list(n['q_end'])
            for arm, payload in n.get('metadata', {}).get('effects', {}).get('payloads', {}).items():
                if payload: held[arm] = deepcopy(payload)
                else: held.pop(arm, None)
        index = end
    tpg['curve_smoothing'] = report
    if report['accepted_macros']:
        tpg['certificate']['geometric_reference_paths_unchanged'] = False
    return report
