"""Independent audit of scene-3 actual execution evidence.

This module never runs a simulator or turns nominal TPG timestamps into actual
events. The caller supplies measured poses, dynamic enablement, force samples
and executor events. Missing evidence fails the audit. Contact and connection
models remain assumptions of the simulator; this is not an experimental proof
of a real LEGO snap or continuously collision-free robot motion.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path


DEFAULTS = {
    'goal_position_tolerance_m': .009,
    'goal_orientation_tolerance_rad': .04,
    'support_force_min_N': 1.0,
    'continuity_force_min_N': .15,
    'min_retraction_m': .008,
    'retraction_tolerance_m': .0001,
    'min_support_trace_samples': 2,
    'support_trace_max_gap_s': .25,
    'permanent_force_min_N': 1.0,
    'required_validation_load_N': 8.0,
    'require_actual_applied_load': False,
    'permanent_proof_duration_s': .5,
    'permanent_proof_max_gap_s': .15,
    'unload_force_max_N': .8,
    'lift_tolerance_m': .0001,
    'support_retracted_position_m': 0.0,
    'required_dynamic_fields': ['support_sensor', 'support_lift'],
    'required_dynamic_connections': [],
    'assembly_trace_max_gap_s': .15,
    'timestamp_tolerance_s': 1e-7,
    'required_installed_parts': ['K1', 'K2'],
    'support_part_id': 'Support',
    'require_dynamic_joints': True,
    'require_final_empty_hands': True,
    'gate_blocks_extraction': False,
}

EVENT_ALIASES = {
    'support_ready': 'support_load_verified',
    'snap_connected': 'part_installed',
    'fixture_unloaded': 'support_unloaded',
    'fixture_extracted': 'support_extracted',
    'fixture_returned': 'support_returned',
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sensor_telemetry_valid(sample, part):
    sensors = sample.get('snap', {})
    sensor = sensors.get(part, {}) if isinstance(sensors, Mapping) else {}
    return (isinstance(sensor, Mapping) and sensor.get('valid') is True and
            all(isinstance(sensor.get(key), (list, tuple)) and len(sensor[key]) == 3 and
                all(_finite(value) for value in sensor[key]) for key in ('force_N', 'torque_Nm')))


def _position_error(actual, target):
    if len(actual) < 3 or len(target) < 3 or not all(_finite(x) for x in list(actual) + list(target)):
        raise ValueError('nonfinite_or_incomplete_pose')
    return math.sqrt(sum((actual[i]-target[i])**2 for i in range(3)))


def _orientation_error(actual, target):
    if len(target) < 7:
        return None
    if len(actual) != 7:
        raise ValueError('missing_measured_orientation')
    qa, qt = actual[3:7], target[3:7]
    na, nt = math.sqrt(sum(x*x for x in qa)), math.sqrt(sum(x*x for x in qt))
    if abs(na-1) > .01 or abs(nt-1) > .01:
        raise ValueError('nonunit_quaternion')
    return 2*math.acos(min(1.0, abs(sum(a*b for a,b in zip(qa, qt))/(na*nt))))


def validate_execution(graph: Mapping, execution: Mapping, requirements: Mapping | None = None) -> dict:
    """Return an audit report; never modify ``graph`` or ``execution``.

    Required evidence includes node_started/node_completed events and semantic
    events: support_load_verified, arm_support_released, part_installed for K1
    and K2, permanent_structure_verified, support_unloaded, support_extracted,
    support_returned and gate_closed. Semantic events reference their serving
    node and occur inside its actual interval. Support and permanent acceptance
    events have details.valid=true. Unload records details.retraction_m.

    ``trace`` rows have time, support_force_N and support_valid. Redundant
    permanent bearing is accepted only with permanent_support_valid,
    permanent_force_N, bridge_position_error_m and bridge_orientation_error_rad.
    The permanent verification load command appears as insertion_load_N. When
    require_actual_applied_load is enabled, every accepted proof sample must
    also contain applied_load_N and applied_load_substeps recorded after actual
    force callbacks, plus valid dedicated trace.snap.Bridge sensor vectors.
    Required dynamic connections are checked in final snap_dynamic and the
    latest trace must retain valid dedicated Bridge sensor telemetry. ``part_poses``
    holds measured world poses. Goal poses are explicit in requirements or
    graph.validation_requirements. All joint dynamics flags must be true;
    final_state.held reports both empty hands. Additional required dynamically
    enabled bodies can be listed by requirements.required_dynamic_objects.
    When assembly tool force/torque limits are supplied, assembly_force_guard
    metadata and actual node windows define required observations independently
    of trace contact flags. Every observed guarded wrench must be valid and
    within these experiment limits; missing intervals cannot silently pass.
    """
    req = {**DEFAULTS, **dict(graph.get('validation_requirements', {})), **dict(requirements or {})}
    issues = []

    def violation(code, **context):
        issues.append({'code': code, **context})

    def require(condition, code, **context):
        if not condition:
            violation(code, **context)
        return bool(condition)

    for key in ['goal_position_tolerance_m', 'goal_orientation_tolerance_rad',
                'support_force_min_N', 'continuity_force_min_N', 'permanent_force_min_N', 'required_validation_load_N',
                'permanent_proof_duration_s', 'permanent_proof_max_gap_s', 'unload_force_max_N', 'lift_tolerance_m',
                'min_retraction_m', 'retraction_tolerance_m',
                'support_trace_max_gap_s', 'assembly_trace_max_gap_s', 'timestamp_tolerance_s']:
        if not _finite(req[key]) or req[key] < 0:
            raise ValueError('Invalid validation requirement: '+key)
    assembly_enabled = any(key in req for key in ('assembly_tool_force_limit_N', 'assembly_tool_torque_limit_Nm'))
    if assembly_enabled:
        for key in ('assembly_tool_force_limit_N', 'assembly_tool_torque_limit_Nm'):
            if not _finite(req.get(key)) or req[key] < 0:
                raise ValueError('Invalid validation requirement: '+key)
    tolerance = req['timestamp_tolerance_s']
    nodes = graph.get('nodes', [])
    by_id = {n['id']: n for n in nodes}
    require(bool(nodes) and len(by_id) == len(nodes), 'empty_or_duplicate_graph_nodes')
    events = execution.get('events', [])
    require(isinstance(events, list) and bool(events), 'missing_actual_events')
    if not isinstance(events, list):
        events = []
    timestamps, semantics = {}, {}
    last_time = -math.inf
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or not _finite(event.get('time')):
            violation('invalid_event_time', event_index=index)
            continue
        time = event['time']
        require(time >= -tolerance and time+tolerance >= last_time,
                'nonmonotonic_or_negative_event_time', event_index=index)
        last_time = max(last_time, time)
        kind = EVENT_ALIASES.get(event.get('kind'), event.get('kind'))
        node_id = event.get('node', event.get('node_id'))
        if kind in ('node_started', 'node_completed'):
            if not require(node_id in by_id, 'unknown_actual_node', node=node_id):
                continue
            field = 'start' if kind == 'node_started' else 'end'
            row = timestamps.setdefault(node_id, {})
            require(field not in row, 'duplicate_node_event', node=node_id, field=field)
            row[field] = time
        elif kind in ('support_load_verified', 'arm_support_released', 'part_installed',
                      'permanent_structure_verified', 'support_unloaded',
                      'support_extracted', 'support_returned', 'gate_closed'):
            normalized = {**event, 'kind': kind, 'node': node_id,
                          'details': dict(event.get('details') or {})}
            semantics.setdefault(kind, []).append(normalized)

    for node in nodes:
        interval = timestamps.get(node['id'], {})
        require('start' in interval and 'end' in interval and interval.get('end', -1)+tolerance >= interval.get('start', 0),
                'missing_or_reversed_actual_interval', node=node['id'])
    for edge in graph.get('edges', []):
        first, second = timestamps.get(edge['source'], {}), timestamps.get(edge['target'], {})
        require('end' in first and 'start' in second and first.get('end', math.inf) <= second.get('start', -math.inf)+tolerance,
                'predecessor_not_completed', source=edge['source'], target=edge['target'])
    for side in ['A', 'B']:
        intervals = sorted((timestamps[n['id']]['start'], timestamps[n['id']]['end'], n['id'])
                           for n in nodes if n['arm'] == side and
                           {'start', 'end'} <= timestamps.get(n['id'], {}).keys())
        latest_end = -math.inf
        for start, end, node_id in intervals:
            require(start+tolerance >= latest_end, 'overlapping_arm_service', arm=side, node=node_id)
            latest_end = max(latest_end, end)
    for kind, rows in semantics.items():
        for row in rows:
            interval = timestamps.get(row['node'], {})
            require('start' in interval and 'end' in interval and
                    interval.get('start', math.inf)-tolerance <= row['time'] <= interval.get('end', -math.inf)+tolerance,
                    'semantic_event_outside_serving_node', kind=kind, node=row['node'])

    def one(kind):
        rows = semantics.get(kind, [])
        require(len(rows) == 1, 'missing_or_duplicate_semantic_event', kind=kind, count=len(rows))
        return rows[0] if len(rows) == 1 else None

    handoff = one('support_load_verified')
    released = one('arm_support_released')
    permanent = one('permanent_structure_verified')
    unloaded = one('support_unloaded')
    extracted = one('support_extracted')
    returned = one('support_returned')
    gate = one('gate_closed')

    def precedes(first, second, code):
        return require(first is not None and second is not None and first['time'] <= second['time']+tolerance, code)

    precedes(handoff, released, 'A_released_before_verified_load_transfer')
    precedes(permanent, unloaded, 'unloaded_before_permanent_acceptance')
    precedes(unloaded, extracted, 'extracted_before_unloading')
    precedes(extracted, returned, 'returned_before_extraction')
    precedes(permanent, gate, 'gate_closed_before_permanent_acceptance')
    if req['gate_blocks_extraction']:
        precedes(extracted, gate, 'inner_support_trapped_by_gate')
    if handoff:
        force = handoff['details'].get('force_N')
        require(handoff['details'].get('valid') is True and _finite(force) and force >= req['support_force_min_N'],
                'load_transfer_missing_measured_force')
    if permanent:
        require(permanent['details'].get('valid') is True, 'permanent_acceptance_missing_valid_measurement')
    if unloaded:
        distance = unloaded['details'].get('retraction_m')
        require(_finite(distance) and distance+req['retraction_tolerance_m'] >= req['min_retraction_m'],
                'insufficient_measured_shoe_retraction')

    installed = semantics.get('part_installed', [])
    installed_by_part = {}
    for part in req['required_installed_parts']:
        matches = [e for e in installed if e['details'].get('part', by_id.get(e['node'], {}).get('part')) == part]
        require(len(matches) == 1, 'missing_or_duplicate_install_event', part=part)
        if len(matches) == 1:
            installed_by_part[part] = matches[0]
            precedes(released, matches[0], 'part_installed_before_A_released')
            precedes(matches[0], permanent, 'permanent_acceptance_before_installation')
    if 'K1' in installed_by_part and 'K2' in installed_by_part:
        precedes(installed_by_part['K1'], installed_by_part['K2'], 'K2_installed_before_K1')
    # Explicit compiler metadata allows auditing resource ownership even when a
    # corrupt or incomplete graph accidentally omits the corresponding edge.
    for node in nodes:
        action = node.get('metadata', {}).get('action', node.get('kind'))
        if node['arm'] == 'A' and (action in ('pick', 'grasp', 'fetch') or str(action).startswith('fetch_')):
            interval = timestamps.get(node['id'], {})
            require(released is not None and interval.get('start', -math.inf)+tolerance >= released['time'],
                    'A_fetch_started_while_still_supporting', node=node['id'])

    trace = execution.get('trace', [])
    if not isinstance(trace, list):
        trace = []
    assembly_audit = {'enabled': assembly_enabled, 'intervals': [], 'samples_checked': {'A': 0, 'B': 0},
                      'max_force_N': {'A': None, 'B': None}, 'max_torque_Nm': {'A': None, 'B': None}}
    if assembly_enabled:
        # Serving installation nodes must carry the guard as well as their
        # approach/retreat fragments. Otherwise omitted metadata could disable
        # the independent audit despite a completed installation event.
        for event in installed + ([gate] if gate else []):
            require(by_id.get(event['node'], {}).get('metadata', {}).get('assembly_force_guard') is True,
                    'assembly_event_node_missing_force_guard', node=event['node'])
        timed_samples = sorted((sample for sample in trace if isinstance(sample, Mapping) and _finite(sample.get('time'))),
                               key=lambda sample: sample['time'])
        for side in ('A', 'B'):
            windows = sorted((timestamps[node['id']]['start'], timestamps[node['id']]['end'])
                             for node in nodes if node['arm'] == side and
                             node.get('metadata', {}).get('assembly_force_guard') is True and
                             {'start', 'end'} <= timestamps.get(node['id'], {}).keys())
            merged = []
            for start, end in windows:
                if end < start:
                    continue  # The actual interval audit already rejects this.
                if merged and start <= merged[-1][1]+tolerance:
                    merged[-1][1] = max(end, merged[-1][1])
                else:
                    merged.append([start, end])
            for start, end in merged:
                observed = [sample['time'] for sample in timed_samples if start-tolerance <= sample['time'] <= end+tolerance]
                require(bool(observed), 'missing_assembly_force_samples', arm=side, start=start, end=end)
                points = [start] + observed + [end]
                gap = max((b-a for a,b in zip(points, points[1:])), default=math.inf)
                require(gap <= req['assembly_trace_max_gap_s']+tolerance,
                        'assembly_force_trace_has_unobserved_gap', arm=side, start=start, end=end, max_gap_s=gap)
                assembly_audit['intervals'].append({'arm': side, 'start': start, 'end': end, 'sample_count': len(observed)})
            for sample in trace:
                if not isinstance(sample, Mapping):
                    continue
                time = sample.get('time')
                in_window = _finite(time) and any(start-tolerance <= time <= end+tolerance for start,end in merged)
                if not (in_window or sample.get(side+'_assembly_contact') is True):
                    continue
                force, torque = sample.get(side+'_tool_force_N'), sample.get(side+'_tool_torque_Nm')
                valid = (_finite(time) and sample.get(side+'_tool_sensor_valid') is True and
                         all(isinstance(vector, (list, tuple)) and len(vector) == 3 and
                             all(_finite(value) for value in vector) for vector in (force, torque)))
                assembly_audit['samples_checked'][side] += 1
                if not require(valid, 'invalid_assembly_tool_sensor_sample', arm=side, time=time):
                    continue
                force_norm, torque_norm = math.hypot(*force), math.hypot(*torque)
                assembly_audit['max_force_N'][side] = max(assembly_audit['max_force_N'][side] or 0., force_norm)
                assembly_audit['max_torque_Nm'][side] = max(assembly_audit['max_torque_Nm'][side] or 0., torque_norm)
                require(force_norm <= req['assembly_tool_force_limit_N']+tolerance,
                        'assembly_tool_force_limit_exceeded', arm=side, time=time, force_N=force_norm)
                require(torque_norm <= req['assembly_tool_torque_limit_Nm']+tolerance,
                        'assembly_tool_torque_limit_exceeded', arm=side, time=time, torque_Nm=torque_norm)
    force_samples = []
    def permanent_bearing_measured(sample):
        force = sample.get('permanent_force_N')
        position = sample.get('bridge_position_error_m')
        angle = sample.get('bridge_orientation_error_rad')
        return (sample.get('permanent_support_valid') is True and
                sample.get('permanent_sensor_valid') is True and
                _finite(force) and force >= req['permanent_force_min_N'] and
                _finite(position) and 0 <= position <= req['goal_position_tolerance_m'] and
                _finite(angle) and 0 <= angle <= req['goal_orientation_tolerance_rad'])

    if handoff and permanent and released:
        begin, end = handoff['time'], permanent['time']
        for sample in trace:
            if isinstance(sample, Mapping) and _finite(sample.get('time')) and begin-tolerance <= sample['time'] <= end+tolerance:
                force = sample.get('support_force_N')
                temporary_bearing = sample.get('support_valid') is True and _finite(force) and force >= req['continuity_force_min_N']
                require(temporary_bearing or permanent_bearing_measured(sample),
                        'invalid_or_lost_support_force_sample', time=sample['time'])
                force_samples.append(sample)
        require(len(force_samples) >= req['min_support_trace_samples'], 'insufficient_measured_support_samples', count=len(force_samples))
        points = [begin] + sorted(s['time'] for s in force_samples) + [end]
        gap = max((b-a for a,b in zip(points, points[1:])), default=math.inf)
        require(gap <= req['support_trace_max_gap_s']+tolerance, 'support_trace_has_unobserved_gap', max_gap_s=gap)
    validation_load_samples = []
    require_dedicated_telemetry = (req['require_actual_applied_load'] or
                                   'Bridge' in req['required_dynamic_connections'])
    proof_start, last_proof_time = None, None
    proof_begin = timestamps.get(permanent['node'], {}).get('start', math.inf) if permanent else math.inf
    for sample in sorted(force_samples, key=lambda item: item['time']):
        # A completion-time telemetry row can already have the proof-active
        # flag cleared by the completion event. Only pre-acceptance rows belong
        # to the observation window being independently audited.
        if not permanent or sample['time'] < proof_begin-tolerance or sample['time'] >= permanent['time']-tolerance:
            continue
        lift = sample.get('support_lift_position_m')
        temporary_force = sample.get('support_force_N')
        applied_load = sample.get('applied_load_N')
        applied_substeps = sample.get('applied_load_substeps')
        applied_evidence = (_finite(applied_load) and
                            applied_load+tolerance >= req['required_validation_load_N'] and
                            _finite(applied_substeps) and applied_substeps >= 1 and
                            applied_substeps == int(applied_substeps))
        qualified = (
            sample.get('permanent_verification_active') is True and
            sample.get('bearing_sensor') == 'PermanentBridgeFT' and
            _finite(sample.get('insertion_load_N')) and
            sample['insertion_load_N']+tolerance >= req['required_validation_load_N'] and
            (not req['require_actual_applied_load'] or applied_evidence) and
            (not require_dedicated_telemetry or _sensor_telemetry_valid(sample, 'Bridge')) and
            permanent_bearing_measured(sample) and
            sample.get('support_valid') is True and _finite(temporary_force) and
            temporary_force <= req['unload_force_max_N'] and
            _finite(lift) and abs(lift-req['support_retracted_position_m']) <= req['lift_tolerance_m'])
        if not qualified:
            proof_start, last_proof_time = None, None
            continue
        validation_load_samples.append(sample)
        if last_proof_time is None or sample['time']-last_proof_time > req['permanent_proof_max_gap_s']+tolerance:
            proof_start = sample['time']
        last_proof_time = sample['time']
    proof_duration = last_proof_time-proof_start if proof_start is not None else 0.
    require(bool(validation_load_samples), 'missing_measured_permanent_load_test')
    require(proof_start is not None and proof_duration+tolerance >= req['permanent_proof_duration_s'] and
            permanent is not None and permanent['time']-last_proof_time <= req['permanent_proof_max_gap_s']+tolerance,
            'permanent_proof_duration_or_continuity_not_met', observed_duration_s=proof_duration)

    goals = req.get('goal_poses', {})
    actual_poses = execution.get('part_poses', {})
    require(bool(goals), 'missing_explicit_goal_poses')
    for part in list(req['required_installed_parts']) + ['Gate', req['support_part_id']]:
        require(part in goals, 'missing_required_goal_pose', part=part)
    position_errors, orientation_errors = {}, {}
    for part, target in goals.items():
        if not require(part in actual_poses, 'missing_actual_part_pose', part=part):
            continue
        try:
            position_errors[part] = _position_error(actual_poses[part], target)
            orientation = _orientation_error(actual_poses[part], target)
            if orientation is not None:
                orientation_errors[part] = orientation
            require(position_errors[part] <= req['goal_position_tolerance_m'], 'goal_position_error', part=part, error_m=position_errors[part])
            require(orientation is None or orientation <= req['goal_orientation_tolerance_rad'], 'goal_orientation_error', part=part, error_rad=orientation)
        except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
            violation('invalid_pose_evidence', part=part, detail=str(error))

    dynamics = execution.get('dynamic_enablement', {})
    if req['require_dynamic_joints']:
        for side in ('A', 'B'):
            flags = dynamics.get(side, {}).get('joints', [])
            require(len(flags) == 6 and all(v is True for v in flags), 'dynamic_joints_not_enabled', arm=side)
    for name in req.get('required_dynamic_objects', []):
        require(dynamics.get('objects', {}).get(name) is True, 'required_object_not_dynamic', object=name)
    for name in req['required_dynamic_fields']:
        require(dynamics.get(name) is True, 'required_dynamic_field_not_enabled', field=name)
    connections = execution.get('snap_dynamic', {})
    for name in req['required_dynamic_connections']:
        require(isinstance(connections, Mapping) and connections.get(name) is True,
                'required_connection_not_dynamic', connection=name)
    if require_dedicated_telemetry:
        timed_trace = [sample for sample in trace if isinstance(sample, Mapping) and _finite(sample.get('time'))]
        latest_sample = max(timed_trace, key=lambda sample: sample['time'], default={})
        require(_sensor_telemetry_valid(latest_sample, 'Bridge'), 'invalid_final_dedicated_sensor_telemetry')
    if req['require_final_empty_hands']:
        held = execution.get('final_state', {}).get('held', {})
        require(all(side in held and held[side] in (None, '', False) for side in ('A', 'B')),
                'final_hands_not_known_empty')
    require(execution.get('success') is True and not execution.get('failure'), 'executor_did_not_report_success')
    return {
        'success': not issues,
        'scope': 'actual TPG event, semantic guard, measured support and final pose audit; idealized contact/connection and sampled collision models remain limitations',
        'violations': issues,
        'expected_nodes': len(nodes), 'actual_node_intervals': len(timestamps),
        'actual_node_timestamps': timestamps,
        'graph_edges_checked': len(graph.get('edges', [])),
        'support_trace_samples_checked': len(force_samples),
        'permanent_load_test_samples_checked': len(validation_load_samples),
        'permanent_load_test_observed_duration_s': proof_duration,
        'assembly_contact_audit': assembly_audit,
        'goal_position_errors_m': position_errors,
        'goal_orientation_errors_rad': orientation_errors,
        'max_goal_position_error_m': max(position_errors.values(), default=None),
        'max_goal_orientation_error_rad': max(orientation_errors.values(), default=None),
        'requirements': req,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1] / 'results' / 'apex'
    parser.add_argument('--graph', type=Path, default=base/'tpg.json')
    parser.add_argument('--execution', type=Path, default=base/'execution.json')
    parser.add_argument('--requirements', type=Path)
    parser.add_argument('--output', type=Path, default=base/'execution_validation.json')
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding='utf-8'))
    execution = json.loads(args.execution.read_text(encoding='utf-8'))
    requirements = json.loads(args.requirements.read_text(encoding='utf-8')) if args.requirements else None
    report = validate_execution(graph, execution, requirements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['success'] else 1)


if __name__ == '__main__':
    main()
