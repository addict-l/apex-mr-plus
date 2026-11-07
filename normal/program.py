"""Continuous TPG timing adapted from Scene3 at commit 70ada87.

Shared profiles, dependency gates and jerk bounds are reused unchanged.
The original Scene1 contact skills retain their motion plus dwell duration.
"""
import json
import math
import numpy as np
from .tpg import simulate_tpg, _graph
from .curve import eval_at_s


def compress_collinear(path):
    if len(path) < 3:
        return [list(q) for q in path]
    out = [list(path[0])]
    for i in range(1, len(path)-1):
        a = np.asarray(path[i])-np.asarray(out[-1])
        b = np.asarray(path[i+1])-np.asarray(path[i])
        if np.linalg.norm(a) < 1e-10:
            continue
        if (np.linalg.norm(b) > 1e-10 and np.dot(a, b) > 0 and
                np.linalg.norm(a/np.linalg.norm(a)-b/np.linalg.norm(b)) < 1e-7):
            continue
        out.append(list(path[i]))
    out.append(list(path[-1]))
    return out


def _duration(q0, q1, vmax, amax, jmax):
    d = max(abs(a-b) for a, b in zip(q0, q1))
    # Quintic maxima: |s'|=1.875, |s''|=10/sqrt(3), |s'''|=60.
    return max(1.875*d/vmax, math.sqrt((10/math.sqrt(3))*d/amax),
               (60*d/jmax)**(1/3), .08)


def _limits(node, vmax, amax, jmax):
    md = node.get('metadata', {})
    result = tuple(min(global_value, float(md.get(key, global_value))) for key, global_value in
                   [('max_velocity_rad_s', vmax), ('max_acceleration_rad_s2', amax),
                    ('max_jerk_rad_s3', jmax)])
    if not all(math.isfinite(v) and v > 0 for v in result):
        raise ValueError('Invalid node motion limits: '+node['id'])
    return result


def _context(node):
    md = node.get('metadata', {})
    ignored = {'effects', 'planner', 'path_source', 'planning_stats', 'motion_run_id',
               'motion_run_offset', 'motion_run_duration', 'motion_run_members',
               'continuation_from', 'continuation_to', 'performance_boundary'}
    return json.dumps({'arm': node['arm'], 'payload': node.get('payload'),
                       'mode': node.get('mode', {}),
                       'metadata': {k: v for k, v in md.items() if k not in ignored}},
                      sort_keys=True, separators=(',', ':'))


def _direction(node):
    points = compress_collinear(node['q_samples'])
    if node['node_type'] != 'pose' or len(points) != 2:
        return None
    d = np.asarray(points[-1])-np.asarray(points[0])
    length = np.linalg.norm(d)
    return d/length if length > 1e-9 else None


def _inverse_quintic(fraction):
    if fraction <= 0:
        return 0.
    if fraction >= 1:
        return 1.
    lo, hi = 0., 1.
    for _ in range(55):
        u = (lo+hi)/2
        if u*u*u*(10+u*(-15+6*u)) < fraction:
            lo = u
        else:
            hi = u
    return (lo+hi)/2


def _gate_run(tpg, run):
    """Acquire future prerequisites before moving; reject cyclic contractions."""
    ids = {n['id'] for n in run}
    first = run[0]['id']
    edges = tpg['edges']
    have = {(e['source'], e['target']) for e in edges}
    extra = []
    for edge in edges:
        if edge['target'] in ids and edge['source'] not in ids:
            pair = (edge['source'], first)
            if pair not in have:
                extra.append({**edge, 'target': first, 'reason': 'continuous_run_gate',
                              'original_target': edge['target'],
                              'original_reason': edge.get('reason')})
                have.add(pair)
    try:
        _graph(tpg['nodes'], edges+extra)
    except ValueError:
        return False
    edges.extend(extra)
    return True


def refresh_schedule(tpg, acquire_runs=True):
    """Refresh timing after conflict reconstruction without changing curves."""
    nodes = tpg['nodes']
    by_id = {n['id']: n for n in nodes}
    if acquire_runs:
        for node in nodes:
            members = node.get('metadata', {}).get('motion_run_members')
            if members and not _gate_run(tpg, [by_id[nid] for nid in members]):
                raise ValueError('New conflict order prevents a continuous run: '+node['id'])
    _, _, _, topology = _graph(nodes, tpg['edges'])
    tpg['topological_order'] = topology
    prediction = simulate_tpg(tpg)
    tpg['nominal_schedule'] = prediction['nodes']
    tpg['nominal_task_schedule'] = prediction['task_schedule']
    tpg['nominal_timestamps'] = prediction['timestamps']
    tpg['nominal_makespan'] = prediction['makespan']
    for node in nodes:
        node['nominal_duration'] = node['duration']
        node.update(prediction['timestamps'][node['id']])
    clock = 0.
    for row in tpg['reference_schedule']:
        duration = sum(by_id[n]['duration'] for n in tpg['task_map'][row['task_id']]['node_ids'])
        row.update(start=clock, end=clock+duration, duration=duration)
        tpg['task_map'][row['task_id']]['duration'] = duration
        cursor = clock
        for nid in tpg['task_map'][row['task_id']]['node_ids']:
            by_id[nid]['reference_start'] = cursor
            cursor += by_id[nid]['duration']
            by_id[nid]['reference_end'] = cursor
        clock += duration
    tpg['reference_makespan'] = clock
    tpg.setdefault('counts', {})['continuous_gate_edges'] = sum(
        e.get('reason') == 'continuous_run_gate' for e in tpg['edges'])
    tpg['counts']['all_edges_after_timing'] = len(tpg['edges'])
    tpg['nominal_schedule_requires_refresh'] = False
    return tpg


def parameterize(tpg, vmax=.60, amax=1.20, jmax=8.0):
    if not all(math.isfinite(v) and v > 0 for v in (vmax, amax, jmax)):
        raise ValueError('Velocity, acceleration and jerk bounds must be positive and finite')
    nodes = tpg['nodes']
    _graph(nodes, tpg['edges'])
    for node in nodes:
        if not node.get('q_samples'):
            raise ValueError('Executable TPG node requires joint geometry: '+node['id'])
        md = node.setdefault('metadata', {})
        for key in ('motion_run_id', 'motion_run_offset', 'motion_run_duration',
                    'motion_run_members', 'continuation_from', 'continuation_to'):
            md.pop(key, None)
        md['performance_boundary'] = {'kind': 'required_pause', 'reason': 'skill_or_motion_endpoint'}
        lv, la, lj = _limits(node, vmax, amax, jmax)
        if node.get('curve_profile'):
            node['motion_segments'] = []
            continue
        points = compress_collinear(node['q_samples'])
        segments = []
        for q0, q1 in zip(points, points[1:]):
            if max(abs(a-b) for a, b in zip(q0, q1)) > 1e-9:
                segments.append({'q0': q0, 'q1': q1, 'duration': _duration(q0, q1, lv, la, lj)})
        node['motion_segments'] = segments
        node['duration'] = max(sum(s['duration'] for s in segments), .08)
        if node['node_type'] == 'skill':
            dwell = float(md.get('dwell_s', .20))
            if not math.isfinite(dwell) or dwell < 0:
                raise ValueError('Invalid node dwell: '+node['id'])
            required = {'verify_support': .30, 'verify_permanent': .65}.get(md.get('action'), 0.)
            node['duration'] += max(dwell, required)
            md['qualified_dwell_s'] = max(dwell, required)

    runs = []
    at = 0
    while at < len(nodes):
        first = nodes[at]
        direction = _direction(first)
        run = [first]
        end = at+1
        if direction is not None and not first.get('curve_profile'):
            context = _context(first)
            while end < len(nodes):
                before, after = run[-1], nodes[end]
                next_direction = _direction(after)
                if (next_direction is None or after.get('curve_profile') or _context(after) != context or
                        before['metadata'].get('effects') or
                        not np.allclose(before['q_end'], after['q_start'], atol=1e-9, rtol=0) or
                        np.linalg.norm(direction-next_direction) >= 1e-7):
                    break
                run.append(after)
                end += 1
        if len(run) > 1 and _gate_run(tpg, run):
            q0, q1 = run[0]['q_start'], run[-1]['q_end']
            lv, la, lj = _limits(first, vmax, amax, jmax)
            total = _duration(q0, q1, lv, la, lj)
            vector = np.asarray(q1)-np.asarray(q0)
            denom = float(vector@vector)
            offset = 0.
            for index, node in enumerate(run):
                fraction = float((np.asarray(node['q_end'])-np.asarray(q0))@vector)/denom
                finish = total*_inverse_quintic(fraction)
                node['motion_segments'] = [{'q0': list(q0), 'q1': list(q1),
                    'duration': finish-offset, 'profile_duration': total, 'profile_offset': offset}]
                node['duration'] = finish-offset
                node['metadata'].update(motion_run_id=first['id'], motion_run_offset=offset,
                                        motion_run_duration=total)
                if index:
                    node['metadata']['continuation_from'] = run[index-1]['id']
                if index < len(run)-1:
                    node['metadata']['continuation_to'] = run[index+1]['id']
                    node['metadata']['performance_boundary'] = {
                        'kind': 'pass_through', 'reason': 'collinear_shared_profile'}
                offset = finish
            first['metadata']['motion_run_members'] = [n['id'] for n in run]
            runs.append({'id': first['id'], 'node_count': len(run), 'duration_s': total})
            at = end
        else:
            at += 1

    curved = {}
    for node in nodes:
        if node.get('curve_profile'):
            curved.setdefault(node['curve_run_id'], []).append(node)
    for run in curved.values():
        first = run[0]
        if not _gate_run(tpg, run):
            raise ValueError('Curve run would cross a dependency cycle: '+first['id'])
        profile = first['curve_profile']
        if not profile.get('collision_validated'):
            raise ValueError('Cannot execute an unchecked curve: '+first['id'])
        for index, node in enumerate(run):
            lv, la, lj = _limits(node, vmax, amax, jmax)
            for key, limit in [('velocity', lv), ('acceleration', la), ('jerk', lj)]:
                if max(profile['derivative_bounds'][key]) > limit*(1+1e-8):
                    raise ValueError('Curved run exceeds '+key+' bound')
            s0, s1 = node['curve_range_s']
            offset = profile['duration']*_inverse_quintic(s0/profile['length'])
            finish = profile['duration']*_inverse_quintic(s1/profile['length'])
            direction = np.asarray(eval_at_s(profile, s1, 1))
            direction /= np.linalg.norm(direction)
            node['duration'] = finish-offset
            node['motion_segments'] = [{'q0': profile['pieces'][0]['controls'][0],
                'q1': profile['pieces'][-1]['controls'][-1], 'duration': finish-offset,
                'profile_duration': profile['duration'], 'profile_offset': offset,
                'geometry_profile': profile, 'progress_direction': direction.tolist()}]
            node['metadata'].update(motion_run_id=first['id'], motion_run_offset=offset,
                                    motion_run_duration=profile['duration'])
            if index:
                node['metadata']['continuation_from'] = run[index-1]['id']
            if index < len(run)-1:
                node['metadata']['continuation_to'] = run[index+1]['id']
                node['metadata']['performance_boundary'] = {
                    'kind': 'pass_through', 'reason': 'validated_C2_macro'}
        first['metadata']['motion_run_members'] = [n['id'] for n in run]
        runs.append({'id': first['id'], 'node_count': len(run),
                     'duration_s': profile['duration'], 'geometry': 'validated_C2_Bezier'})

    refresh_schedule(tpg, acquire_runs=False)
    tpg['time_parameterization'] = {
        'type': 'jerk_bounded_continuous_quintic_and_C2_Bezier',
        'max_velocity_rad_s': vmax, 'max_acceleration_rad_s2': amax,
        'max_jerk_rad_s3': jmax, 'continuous_runs': runs,
        'pass_through_boundaries': sum(bool(n['metadata'].get('continuation_to')) for n in nodes),
        'original_node_ids_and_edges_preserved': True,
        'original_geometric_paths_unchanged': not bool(curved),
        'C2_blended_macros': len(curved),
        'skill_dwell_counted_once': True,
        'extra_runtime_endpoint_settling': 'required endpoints only',
    }
    return tpg
