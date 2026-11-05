"""Compare measured Scene3 runs without running a simulator or altering evidence.

Sampled velocity differences are interval-average estimates, not certified
continuous acceleration/jerk extrema. ``write_comparison`` is the only writer.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Mapping
import json
import math
from pathlib import Path
import statistics

from .validation import DEFAULTS, validate_execution


def _read(value):
    return value if isinstance(value, Mapping) else json.loads(Path(value).read_text(encoding='utf-8'))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(_finite(x) for x in value)


def _norm(value):
    return math.sqrt(sum(x*x for x in value))


def _stats(values):
    values = sorted(x for x in values if _finite(x))
    return {'count': len(values), 'sum': sum(values), 'minimum': values[0] if values else None,
            'median': statistics.median(values) if values else None,
            'maximum': values[-1] if values else None}


def _intervals(graph, execution):
    starts, ends, issues = {}, {}, []
    for event in execution.get('events', []):
        kind = event.get('kind'); node = event.get('node'); time = event.get('time')
        if kind not in ('node_started', 'node_completed'):
            continue
        if not _finite(time):
            issues.append('nonfinite actual event time'); continue
        target = starts if kind == 'node_started' else ends
        if node in target:
            issues.append('duplicate actual event: ' + str(node))
        target[node] = time
    ids = {n['id'] for n in graph.get('nodes', [])}
    if set(starts) != ids or set(ends) != ids:
        issues.append('actual event coverage differs from graph')
    intervals = {n: (starts[n], ends[n]) for n in ids & starts.keys() & ends.keys()
                 if ends[n] >= starts[n]}
    if len(intervals) != len(ids):
        issues.append('missing or reversed actual node interval')
    makespan = execution.get('makespan')
    if not _finite(makespan) or makespan <= 0:
        issues.append('missing positive actual makespan')
    elif intervals and max(b for a, b in intervals.values()) - min(a for a, b in intervals.values()) > makespan + 1e-7:
        issues.append('reported makespan is shorter than actual event span')
    return intervals, issues


def _union_length(intervals):
    total = 0.; end = -math.inf
    for lo, hi in sorted(intervals):
        total += max(0., hi - max(lo, end)); end = max(end, hi)
    return total


def velocity_metrics(trace, arm, max_difference_gap_s=.15):
    """Differentiate measured velocity at its actual, possibly unequal times.

    Acceleration = (v[i+1]-v[i])/dt at the interval midpoint. Jerk is the
    difference of consecutive acceleration estimates divided by their midpoint
    separation. No interpolation; invalid samples and large gaps break chains.
    """
    key = arm + '_joint_velocity_rad_s'; points = []; accelerations = []; jerks = []
    invalid = 0; bad_time = 0; gaps = []; previous = None; previous_acc = None
    for row in trace:
        t = row.get('time'); v = row.get(key)
        if not _finite(t) or not _vector(v, 6):
            invalid += 1; previous = None; previous_acc = None; continue
        points.append((t, v))
        if previous is not None:
            dt = t - previous[0]
            if dt <= 0:
                bad_time += 1; previous_acc = None
            else:
                gaps.append(dt)
                if dt <= max_difference_gap_s + 1e-9:
                    mid = (t + previous[0])/2
                    a = [(v[j]-previous[1][j])/dt for j in range(6)]
                    accelerations.append(a)
                    if previous_acc is not None:
                        jerks.append([(a[j]-previous_acc[1][j])/(mid-previous_acc[0]) for j in range(6)])
                    previous_acc = (mid, a)
                else:
                    previous_acc = None
        previous = (t, v)
    peak = lambda rows: max((abs(x) for row in rows for x in row), default=None)
    return {'valid_samples': len(points), 'invalid_samples': invalid, 'nonincreasing_time_count': bad_time,
            'sample_intervals_s': _stats(gaps), 'difference_gap_limit_s': max_difference_gap_s,
            'excluded_long_gap_count': sum(x > max_difference_gap_s + 1e-9 for x in gaps),
            'max_abs_velocity_rad_s': peak([v for _, v in points]),
            'estimated_max_abs_acceleration_rad_s2': peak(accelerations),
            'estimated_max_abs_jerk_rad_s3': peak(jerks),
            'acceleration_estimate_count': len(accelerations), 'jerk_estimate_count': len(jerks),
            'derivative_scope': 'Interval averages from measured velocity, nonuniform actual timestamps; no continuous extrema or jerk certification.'}


def boundary_metrics(graph, execution, intervals=None, speed_threshold_rad_s=.01, sample_window_s=.06):
    """Observe consecutive same-arm pose->pose boundaries, excluding skills.

    Optional metadata.performance_boundary on the preceding node describes
    its exit: {kind: required_pause|pass_through, reason: ...}. Undeclared
    boundaries remain unknown unless actual predecessor readiness forced a wait.
    """
    if intervals is None:
        intervals, _ = _intervals(graph, execution)
    nodes = {n['id']: n for n in graph['nodes']}; pred = defaultdict(set); outgoing = defaultdict(set)
    for edge in graph['edges']:
        pred[edge['target']].add(edge['source']); outgoing[edge['source']].add(edge['target'])
    rows = []
    for arm in ('A', 'B'):
        samples = [(r['time'], r[arm+'_joint_velocity_rad_s']) for r in execution.get('trace', [])
                   if _finite(r.get('time')) and _vector(r.get(arm+'_joint_velocity_rad_s'), 6)]
        samples.sort(key=lambda x: x[0]); times = [x[0] for x in samples]
        ordered = sorted((n for n in nodes.values() if n['arm'] == arm and n['id'] in intervals),
                         key=lambda n: intervals[n['id']][0])
        for left, right in zip(ordered, ordered[1:]):
            if left['node_type'] != 'pose' or right['node_type'] != 'pose':
                continue
            lo = intervals[left['id']][1]; hi = intervals[right['id']][0]
            at = bisect_left(times, lo)
            candidates = [i for i in (at-1, at) if 0 <= i < len(samples)
                          and abs(times[i]-lo) <= sample_window_s + 1e-9
                          and times[i] <= intervals[right['id']][1] + 1e-9]
            i = min(candidates, key=lambda j: abs(times[j]-lo)) if candidates else None
            speed = max(abs(v) for v in samples[i][1]) if i is not None else None
            ready = max([lo]+[intervals[p][1] for p in pred[right['id']] if p in intervals])
            declaration = left.get('metadata', {}).get('performance_boundary', {})
            kind = declaration.get('kind', 'unknown')
            if kind not in ('required_pause', 'pass_through'):
                kind = 'required_dependency_pause' if ready-lo > 1e-7 else 'unknown'
            rows.append({'arm': arm, 'from': left['id'], 'to': right['id'],
                         'same_task': left['task_id'] == right['task_id'], 'kind': kind,
                         'declaration_reason': declaration.get('reason'), 'gap_s': hi-lo,
                         'dependency_wait_s': max(0., ready-lo),
                         'cross_arm_edges_at_boundary': any(nodes[p]['arm'] != arm for p in pred[right['id']]) or
                                                        any(nodes[p]['arm'] != arm for p in outgoing[left['id']]),
                         'speed_sample_time_s': times[i] if i is not None else None,
                         'sample_distance_s': abs(times[i]-lo) if i is not None else None,
                         'max_abs_joint_speed_rad_s': speed,
                         'near_zero': speed < speed_threshold_rad_s if speed is not None else None})
    observed = [r for r in rows if r['near_zero'] is not None]
    same = [r for r in observed if r['same_task']]
    return {'definition': 'Consecutive same-arm pose->pose exits; nearest actual velocity sample only.',
            'speed_threshold_rad_s': speed_threshold_rad_s, 'sample_window_s': sample_window_s,
            'eligible_boundaries': len(rows), 'observed_boundaries': len(observed),
            'coverage_fraction': len(observed)/len(rows) if rows else None,
            'near_zero_count': sum(r['near_zero'] for r in observed),
            'same_task_observed': len(same), 'same_task_near_zero_count': sum(r['near_zero'] for r in same),
            'classification_counts': dict(Counter(r['kind'] for r in rows)), 'boundaries': rows}


def summarize_run(tpg, execution):
    graph, execution = _read(tpg), _read(execution)
    intervals, issues = _intervals(graph, execution)
    nodes = {n['id']: n for n in graph['nodes']}; trace = execution.get('trace', [])
    makespan = execution.get('makespan'); makespan = makespan if _finite(makespan) else None
    arms = {}; skills = defaultdict(lambda: {'count': 0, 'actual_service_s': 0., 'nominal_service_s': 0.})
    for arm in ('A', 'B'):
        selected = [n for n in nodes.values() if n['arm'] == arm and n['id'] in intervals]
        measured = [intervals[n['id']] for n in selected]; active = _union_length(measured)
        service = sum(b-a for a, b in measured)
        arms[arm] = {'node_count': len(selected), 'active_union_s': active, 'service_sum_s': service,
                     'nominal_service_s': sum(n['duration'] for n in selected),
                     'overlap_s': service-active, 'node_idle_s': makespan-active if makespan is not None else None,
                     'active_fraction': active/makespan if makespan else None,
                     'idle_scope': 'No active graph node; initial physical support holding is not free idle time.'}
    for n in nodes.values():
        if n['node_type'] == 'skill' and n['id'] in intervals:
            row = skills[n.get('metadata', {}).get('action', n.get('kind', 'unknown'))]
            row['count'] += 1; row['nominal_service_s'] += n['duration']
            row['actual_service_s'] += intervals[n['id']][1]-intervals[n['id']][0]
    paths = []
    for tid, task in graph.get('task_map', {}).items():
        if task.get('kind') != 'transit':
            continue
        ns = [nodes[n] for n in task['node_ids']]; samples = [q for n in ns for q in n.get('q_samples', [])]
        if not samples:
            continue
        length = sum(_norm([b[i]-a[i] for i in range(6)]) for a, b in zip(samples, samples[1:]))
        chord = _norm([samples[-1][i]-samples[0][i] for i in range(6)])
        paths.append({'task': tid, 'arm': task['arm'], 'length_joint_L2_rad': length,
                      'endpoint_chord_joint_L2_rad': chord, 'length_over_chord': length/chord if chord > 1e-9 else None,
                      'nominal_service_s': sum(n['duration'] for n in ns)})
    goals = {}
    for name, wanted in graph.get('validation_requirements', {}).get('goal_poses', {}).items():
        actual = execution.get('part_poses', {}).get(name)
        if not _vector(actual, 7) or not _vector(wanted, 7):
            goals[name] = None; continue
        goals[name] = {'position_error_m': _norm([actual[i]-wanted[i] for i in range(3)]),
                       'orientation_error_rad': 2*math.acos(min(1., abs(sum(actual[i]*wanted[i] for i in range(3, 7)))))}
    contact = {}
    for arm in ('A', 'B'):
        guarded = [intervals[n['id']] for n in nodes.values() if n['arm'] == arm and n['id'] in intervals
                   and n.get('metadata', {}).get('assembly_force_guard')]
        samples = [r for r in trace if r.get(arm+'_assembly_contact') is True or
                   (_finite(r.get('time')) and any(a-1e-7 <= r['time'] <= b+1e-7 for a, b in guarded))]
        valid = [r for r in samples if r.get(arm+'_tool_sensor_valid') is True and
                 _vector(r.get(arm+'_tool_force_N'), 3) and _vector(r.get(arm+'_tool_torque_Nm'), 3)]
        contact[arm] = {'observed_samples': len(samples), 'valid_samples': len(valid),
                        'max_force_N': max((_norm(r[arm+'_tool_force_N']) for r in valid), default=None),
                        'max_torque_Nm': max((_norm(r[arm+'_tool_torque_Nm']) for r in valid), default=None)}
    return {'run_id': execution.get('run_id'), 'actual_makespan_s': makespan,
            'nominal_makespan_s': graph.get('nominal_makespan'), 'event_coverage_issues': issues,
            'node_types': dict(Counter(n['node_type'] for n in nodes.values())),
            'motion_segment_count': sum(len(n.get('motion_segments', [])) for n in nodes.values()),
            'arms': arms, 'skills': dict(skills), 'boundaries': boundary_metrics(graph, execution, intervals),
            'kinematics': {arm: velocity_metrics(trace, arm) for arm in ('A', 'B')},
            'paths': paths, 'path_total_joint_L2_rad': sum(p['length_joint_L2_rad'] for p in paths),
            'path_scope': 'Commanded joint polyline; endpoint chord is not a collision-free shortcut certificate.',
            'assembly_contact': contact, 'goal_errors': goals,
            'controller_peak_velocity_rad_s': execution.get('peak_velocity_rad_s')}


def _limits(graph):
    timing = graph.get('time_parameterization', {})
    v, a = timing.get('max_velocity_rad_s'), timing.get('max_acceleration_rad_s2')
    if not _finite(v) or not _finite(a) or v <= 0 or a <= 0:
        return None
    tasks = defaultdict(set)
    for node in graph['nodes']:
        meta = node.get('metadata', {})
        tasks[node['task_id']].add((min(v, meta.get('max_velocity_rad_s', v)),
                                    min(a, meta.get('max_acceleration_rad_s2', a))))
    return {'global': [v, a], 'per_task': {t: sorted(values) for t, values in tasks.items()}}


def _retained_dependencies(before, after):
    ids = {n['id'] for n in after['nodes']}; adjacency = defaultdict(set); lost = []; cache = {}
    for edge in after['edges']:
        adjacency[edge['source']].add(edge['target'])
    for edge in before['edges']:
        source, target = edge['source'], edge['target']
        if source not in cache:
            seen = set(); pending = list(adjacency[source])
            while pending:
                item = pending.pop()
                if item not in seen:
                    seen.add(item); pending.extend(adjacency[item]-seen)
            cache[source] = seen
        if source not in ids or target not in ids or target not in cache[source]:
            lost.append({'source': source, 'target': target, 'reason': edge.get('reason')})
    return {'checked_original_edges': len(before['edges']), 'lost_dependencies': lost,
            'passed': not lost, 'scope': 'Original node IDs and reachability; unknown remapping fails conservatively.'}


def compare_runs(before_tpg, before_exec, after_tpg, after_exec, *, before_manifest=None, after_manifest=None):
    """Return metrics and a conservative gate for a measured speedup claim.

    Pass pipeline manifests to establish identical physical source scenes.
    Code-inclusive scene_signature values may legitimately differ across runs.
    No manifest means unknown physical equivalence, never verified improvement.
    """
    bg, be, ag, ae = map(_read, (before_tpg, before_exec, after_tpg, after_exec))
    b, a = summarize_run(bg, be), summarize_run(ag, ae)
    br = {**DEFAULTS, **bg.get('validation_requirements', {})}
    ar = {**DEFAULTS, **ag.get('validation_requirements', {})}
    audits = {'before': validate_execution(bg, be, br), 'after': validate_execution(ag, ae, ar)}
    dependencies = _retained_dependencies(bg, ag)
    physical = None; environment = None
    if before_manifest is not None and after_manifest is not None:
        bm, am = _read(before_manifest), _read(after_manifest)
        key = 'scenes/10_Scene3_Dynamic.ttt'
        bh = bm.get('input_fingerprint', {}).get('files', {}).get(key)
        ah = am.get('input_fingerprint', {}).get('files', {}).get(key)
        physical = bool(bh and ah and bh == ah)
        environment = bool(bm.get('environment') and bm.get('environment') == am.get('environment'))
    checks = {'same_physical_source_scene': physical, 'same_environment': environment,
              'same_global_and_task_motion_limits': _limits(bg) is not None and _limits(bg) == _limits(ag),
              'same_acceptance_requirements': br == ar,
              'all_original_dependencies_retained': dependencies['passed'],
              'before_actual_audit_passed': audits['before']['success'],
              'after_actual_audit_passed': audits['after']['success'],
              'complete_actual_event_coverage': not b['event_coverage_issues'] and not a['event_coverage_issues']}
    for label, graph, execution in [('before', bg, be), ('after', ag, ae)]:
        checks[label+'_graph_matches_actual_signature'] = bool(graph.get('scene_signature') and graph.get('scene_signature') == execution.get('scene_signature') and
                                                               graph.get('reference_signature') and graph.get('reference_signature') == execution.get('reference_signature'))
        cert = graph.get('certificate', {})
        checks[label+'_nominal_geometry_audited'] = all(cert.get(k) is True for k in
            ['dag', 'motion_overlap_pairs_checked', 'collision_oracle_supplied', 'all_nodes_have_geometry',
             'waiting_endpoints_verified', 'nominal_world_state_verified'])
    for side in ('A', 'B'):
        bk, ak = b['kinematics'][side], a['kinematics'][side]
        bdt, adt = bk['sample_intervals_s']['maximum'], ak['sample_intervals_s']['maximum']
        checks[side+'_velocity_sampling_comparable'] = bool(bdt is not None and adt is not None and
            bdt <= .15+1e-9 and adt <= bdt+1e-7 and not bk['invalid_samples'] and not ak['invalid_samples'] and
            not bk['nonincreasing_time_count'] and not ak['nonincreasing_time_count'])
    limits = _limits(ag)
    vmax = limits['global'][0] if limits else None
    speeds = [a['kinematics'][side]['max_abs_velocity_rad_s'] for side in ('A', 'B')]
    peak = ae.get('peak_velocity_rad_s')
    checks['observed_velocity_within_global_limit'] = bool(vmax and all(_finite(v) and v <= vmax+1e-6 for v in speeds) and
                                                          _finite(peak) and peak <= vmax+1e-6)
    bcover, acover = b['boundaries']['coverage_fraction'], a['boundaries']['coverage_fraction']
    sampling_comparable = bcover is not None and acover is not None and acover >= bcover-1e-9
    accepted = all(value is True for value in checks.values())
    delta = b['actual_makespan_s']-a['actual_makespan_s'] if b['actual_makespan_s'] and a['actual_makespan_s'] else None
    return {'schema_version': 1, 'before': b, 'after': a,
            'comparability_checks': checks, 'comparison_accepted': accepted,
            'validated_speedup': bool(accepted and delta is not None and delta > 1e-7),
            'actual_makespan_reduction_s': delta,
            'actual_makespan_reduction_fraction': delta/b['actual_makespan_s'] if delta is not None else None,
            'boundary_stop_comparison_has_equal_or_better_coverage': sampling_comparable,
            'dependency_retention': dependencies,
            'actual_audits': {label: {'success': value['success'], 'violations': value['violations']}
                              for label, value in audits.items()},
            'limits': {'before': _limits(bg), 'after': limits},
            'limitations': ['Two actual runs, not a randomized or repeated robustness experiment.',
                            'Joint acceleration and jerk are sampled interval estimates; compare sampling before interpreting peaks.',
                            'Near-zero boundary observations are not exact counts of continuous physical stops.',
                            'Serial reference model time is not used as an experimentally measured baseline.']}


def write_comparison(report, output_dir, *, plot=True):
    """Explicitly export a derived comparison; never overwrite input artifacts."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    file = output/'performance_comparison.json'
    file.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    result = {'json': str(file)}
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        labels = ['Baseline', 'Candidate']; runs = [report['before'], report['after']]
        axes[0].bar(labels, [x['actual_makespan_s'] for x in runs], color=['#536779', '#008b8b'])
        axes[0].set_ylabel('Actual simulation seconds'); axes[0].set_title('Measured makespan')
        for i, run in enumerate(runs):
            axes[1].bar([i-.18, i+.18], [run['arms'][a]['active_union_s'] for a in ('A', 'B')], width=.32,
                        color=['#648aaf', '#e7a14b'])
        axes[1].set_xticks([0, 1], labels); axes[1].set_ylabel('Actual active-node seconds'); axes[1].set_title('Arm A / Arm B activity')
        axes[2].bar(labels, [x['boundaries']['near_zero_count'] for x in runs], color=['#536779', '#008b8b'])
        axes[2].set_title('Observed near-zero pose boundaries'); axes[2].set_ylabel('Observed count (sampling dependent)')
        for axis in axes: axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
        status = 'ACCEPTED COMPARISON' if report['comparison_accepted'] else 'NOT ACCEPTED: check evidence/conditions'
        fig.suptitle(status); fig.tight_layout(); png=output/'performance_comparison.png'
        fig.savefig(png, dpi=160); plt.close(fig); result['png'] = str(png)
    return result
