"""Read-only base-joint quick check; outputs only a separate investigation JSON.

Measured q[0] is the raw sim.getJointPosition value. Only measured samples are
unwrapped, with small-step checks and velocity integration as an independent
cross-check. Planned q_path values retain their original winding. A short
endpoint circular arc is not evidence that the corresponding route is safe.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = ROOT/'backups/base_rotation_baseline_20261104_01/results/apex'
FOCUS = ['pick_support_approach', 'support_route_front', 'return_support_approach',
         'pick_gate_approach', 'pick_gate_lift', 'close_gate_approach']
GROUPS = {'pick_support': ('pick_support_approach', 'pick_support_lift_contact_clear'),
          'return_support': ('return_support_approach', 'return_support_retreat_contact_clear'),
          'final_gate_pick_and_install': ('pick_gate_approach', 'close_gate_retreat_contact_clear')}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def finite(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def angle(x): return math.atan2(math.sin(x), math.cos(x))
def degrees(x): return math.degrees(x)


def angle_statistics(values):
    if not values or not all(finite(x) for x in values): return None
    ds = [b-a for a, b in zip(values, values[1:])]
    net = values[-1]-values[0]; total = sum(abs(d) for d in ds)
    shortest = abs(angle(net))
    return {'start_deg': degrees(values[0]), 'end_deg': degrees(values[-1]),
            'minimum_deg': degrees(min(values)), 'maximum_deg': degrees(max(values)),
            'range_deg': degrees(max(values)-min(values)), 'net_change_deg': degrees(net),
            'total_variation_deg': degrees(total), 'backtracking_deg': degrees(max(0., total-abs(net))),
            'shortest_endpoint_circular_arc_deg': degrees(shortest),
            'variation_excess_over_endpoint_arc_deg': degrees(max(0., total-shortest))}


def measured_window(rows, arm, start, end, vmax=.6):
    times = [r['time'] for r in rows]
    selected = rows[bisect_left(times, start-1e-7):bisect_right(times, end+1e-7)]
    valid = [r for r in selected if isinstance(r.get('q'+arm), list) and len(r['q'+arm]) == 6 and
             finite(r['q'+arm][0]) and isinstance(r.get(arm+'_joint_velocity_rad_s'), list) and
             len(r[arm+'_joint_velocity_rad_s']) == 6 and finite(r[arm+'_joint_velocity_rad_s'][0])]
    if len(valid) < 2:
        return {'available': False, 'sample_count': len(valid), 'reason': 'Fewer than two measured samples inside actual event interval'}
    raw = [r['q'+arm][0] for r in valid]; unwrapped = [raw[0]]; wraps = []; issues = []
    signed_integral = absolute_integral = 0.; gaps = []; residuals = []
    for i, (before, after) in enumerate(zip(valid, valid[1:])):
        dt = after['time']-before['time']; change = raw[i+1]-raw[i]; principal = angle(change)
        if dt <= 0: raise ValueError('Trace timestamps must increase')
        gaps.append(dt)
        if dt > .25+1e-8 or vmax*dt >= math.pi or abs(principal) > vmax*dt+.01:
            issues.append({'time_s': after['time'], 'reason': 'Sampling gap or position step prevents reliable winding inference'})
        if abs(change) > math.pi:
            wraps.append({'time_s': after['time'], 'raw_jump_deg': degrees(change), 'physical_increment_deg': degrees(principal)})
        unwrapped.append(unwrapped[-1]+principal)
        va, vb = before[arm+'_joint_velocity_rad_s'][0], after[arm+'_joint_velocity_rad_s'][0]
        integral = (va+vb)*.5*dt
        signed_integral += integral; absolute_integral += (abs(va)+abs(vb))*.5*dt
        residuals.append(abs(principal-integral))
    result = angle_statistics(unwrapped)
    integral_error = degrees(unwrapped[-1]-unwrapped[0]-signed_integral)
    integral_agrees = abs(integral_error) <= max(.5, .02*abs(result['net_change_deg']))
    result.update(available=True, sample_count=len(valid), invalid_sample_count=len(selected)-len(valid),
        first_sample_s=valid[0]['time'], last_sample_s=valid[-1]['time'],
        start_sample_gap_s=max(0., valid[0]['time']-start), end_sample_gap_s=max(0., end-valid[-1]['time']),
        max_sample_gap_s=max(gaps), raw_start_deg=degrees(raw[0]), raw_end_deg=degrees(raw[-1]),
        raw_range_deg=degrees(max(raw)-min(raw)), raw_jump_count=len(wraps), raw_wrap_events=wraps,
        unwrap_inference_reliable=not issues and len(valid) == len(selected), inference_issues=issues,
        measured_velocity_signed_integral_deg=degrees(signed_integral),
        measured_velocity_absolute_integral_deg=degrees(absolute_integral),
        position_net_minus_velocity_integral_deg=integral_error,
        velocity_integral_agrees_with_position=integral_agrees,
        angle_measurement_basis='Position samples are authoritative; velocity integration is a separate diagnostic, not substituted for displacement',
        max_step_position_velocity_residual_deg=degrees(max(residuals)),
        sample_coverage_scope='Measured samples inside actual start/end events; no interpolation or target substitution')
    return result


def analyze(reference, graph=None, execution=None, limit_deg=180.):
    tasks = reference['tasks'] if isinstance(reference, dict) else reference
    ids = [t['id'] for t in tasks]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate task IDs')
    if graph and graph.get('reference_signature') and graph['reference_signature'] != digest(tasks):
        raise ValueError('TPG reference signature does not match supplied complete tasks')
    actual = execution is not None
    if actual and graph is None: raise ValueError('Actual measurements require matching TPG')
    starts = {}; ends = {}; trace = []
    if actual:
        if not execution.get('success') or not execution.get('done'): raise ValueError('Execution is not completed successfully')
        if execution.get('reference_signature') != graph.get('reference_signature') or execution.get('scene_signature') != graph.get('scene_signature'):
            raise ValueError('Actual execution and TPG signatures differ')
        starts = {e['node']: e['time'] for e in execution['events'] if e['kind'] == 'node_started'}
        ends = {e['node']: e['time'] for e in execution['events'] if e['kind'] == 'node_completed'}
        if set(starts) != set(ends) or set(starts) != {n['id'] for n in graph['nodes']}:
            raise ValueError('Actual node-event coverage incomplete')
        trace = execution['trace']
    nodes_by_task = {}
    for node in (graph or {}).get('nodes', []): nodes_by_task.setdefault(node['task_id'], []).append(node)
    result = []
    vmax = (graph or {}).get('time_parameterization', {}).get('max_velocity_rad_s', .6)
    for task in tasks:
        path = task.get('q_path', []); nodes = nodes_by_task.get(task['id'], [])
        row = {'task': task['id'], 'arm': task['arm'], 'kind': task['kind'], 'part': task.get('part'),
               'reference_index': task.get('reference_order'), 'q_start': path[0] if path else None,
               'q_end': path[-1] if path else None, 'node_ids': [n['id'] for n in nodes],
               'planned_reference': angle_statistics([q[0] for q in path]),
               'planned_tpg': angle_statistics([q[0] for n in nodes for q in n.get('q_samples', [])])}
        if actual:
            lo, hi = min(starts[n['id']] for n in nodes), max(ends[n['id']] for n in nodes)
            row.update(actual_start_s=lo, actual_end_s=hi, actual_elapsed_s=hi-lo,
                       measured=measured_window(trace, task['arm'], lo, hi, vmax))
            row['node_measurements'] = [{'node': n['id'], 'start_s': starts[n['id']], 'end_s': ends[n['id']],
                'measured': measured_window(trace, task['arm'], starts[n['id']], ends[n['id']], vmax)} for n in nodes]
        metric = row.get('measured') if actual else row['planned_reference']
        certain = metric is not None and (not actual or metric.get('available') and metric.get('unwrap_inference_reliable'))
        row['red_reasons'] = [key for key in ('range_deg', 'total_variation_deg') if certain and metric[key] > limit_deg+1e-6]
        row['status'] = 'RED' if row['red_reasons'] else ('OK' if certain else 'UNKNOWN')
        result.append(row)
    groups = {}
    guard_windows = {}
    for label, (first, last) in GROUPS.items():
        if first not in ids or last not in ids: continue
        selected = result[ids.index(first):ids.index(last)+1]
        selected = [r for r in selected if r['arm'] == 'B']
        group = {'arm': 'B', 'task_ids': [r['task'] for r in selected],
                 'planned_reference_total_variation_deg': sum(r['planned_reference']['total_variation_deg'] for r in selected)}
        if actual:
            lo, hi = min(r['actual_start_s'] for r in selected), max(r['actual_end_s'] for r in selected)
            group.update(actual_start_s=lo, actual_end_s=hi, actual_elapsed_s=hi-lo, measured=measured_window(trace, 'B', lo, hi, vmax))
        groups[label] = group
        # Include immediately adjacent same-arm transit tasks, crossing any
        # stationary skill nodes. A turn moved just outside the named action
        # must remain visible; all tasks also retain their independent red flag.
        left, right = ids.index(first), ids.index(last)
        for index in range(left-1, -1, -1):
            if result[index]['arm'] == 'B' and result[index]['kind'] == 'transit':
                left = index; break
        for index in range(right+1, len(result)):
            if result[index]['arm'] == 'B' and result[index]['kind'] == 'transit':
                right = index; break
        window_rows = [r for r in result[left:right+1] if r['arm'] == 'B']
        window = {'arm': 'B', 'task_ids': [r['task'] for r in window_rows],
            'first_task': window_rows[0]['task'], 'last_task': window_rows[-1]['task'],
            'planned_total_variation_deg': sum(r['planned_reference']['total_variation_deg'] for r in window_rows),
            'max_planned_transit_variation_deg': max(r['planned_reference']['total_variation_deg'] for r in window_rows if r['kind'] == 'transit'),
            'scope': 'Named semantic stage plus preceding/following same-arm transit; windows can overlap and must not be summed'}
        if actual:
            lo, hi = min(r['actual_start_s'] for r in window_rows), max(r['actual_end_s'] for r in window_rows)
            measured = measured_window(trace, 'B', lo, hi, vmax)
            transit = [r for r in window_rows if r['kind'] == 'transit']
            covered = all(r['measured'].get('available') and r['measured'].get('unwrap_inference_reliable') for r in transit)
            window.update(actual_start_s=lo, actual_end_s=hi, actual_elapsed_s=hi-lo, measured=measured,
                all_transits_have_continuous_measured_position=covered,
                max_actual_transit_variation_deg=max((r['measured']['total_variation_deg'] for r in transit), default=None) if covered else None)
        guard_windows[label] = window
    return {'schema_version': 1, 'run_id': (execution or {}).get('run_id'),
        'scope': 'actual_completed_run' if actual else 'planned_candidate_only', 'joint': 'UR5 joint 1 (base), q[0]',
        'limit_deg': limit_deg, 'task_count': len(tasks), 'actual_makespan_s': (execution or {}).get('makespan'),
        'allocation': ({k: reference.get('allocation', {}).get(k) for k in ['selected_support', 'task_assignments']} if isinstance(reference, dict) else None),
        'focus_tasks': [r for r in result if r['task'] in FOCUS],
        'red_tasks': [r['task'] for r in result if r['status'] == 'RED'], 'statuses': dict(Counter(r['status'] for r in result)),
        'all_tasks': result, 'semantic_groups': groups, 'non_relocation_windows': guard_windows,
        'limitations': ['RED means observed/planned angular excursion or accumulated angular travel exceeds the specified review threshold, not a collision failure.',
            'Planned q_path winding is preserved. Actual wrapped sensor positions are unwrapped only between densely sampled states. Measured-velocity integration disagreement is reported separately and must not be hidden.',
            'Shortest endpoint circular arc is a lower bound; it does not certify IK branch compatibility, joint limits, tool clearance or a collision-free shortcut.',
            'Actual windows use recorded samples inside node events, so small endpoint sampling gaps are explicitly reported. No simulation is run.']}


def compare_windows(after, before):
    """Like-for-like complete windows: no relocating a long turn next door."""
    actual = after['scope'] == 'actual_completed_run'
    rows = []
    for label in GROUPS:
        a, b = after.get('non_relocation_windows', {}).get(label), before.get('non_relocation_windows', {}).get(label)
        if a is None or b is None:
            rows.append({'window': label, 'passed': False, 'reason': 'Missing semantic window'}); continue
        same_boundary = a['first_task'] == b['first_task'] and a['last_task'] == b['last_task']
        if actual:
            quality = all(x.get('all_transits_have_continuous_measured_position') and
                x.get('measured', {}).get('available') and x['measured'].get('unwrap_inference_reliable') for x in (a, b))
            av, bv = a.get('measured', {}).get('total_variation_deg'), b.get('measured', {}).get('total_variation_deg')
            maximum = a.get('max_actual_transit_variation_deg')
        else:
            quality = True
            av, bv = a['planned_total_variation_deg'], b['planned_total_variation_deg']
            maximum = a['max_planned_transit_variation_deg']
        reduced = finite(av) and finite(bv) and av < bv-1e-6
        bounded = finite(maximum) and maximum <= after['limit_deg']+1e-6
        rows.append({'window': label, 'same_boundary_tasks': same_boundary, 'position_sampling_continuous': quality,
            'before_total_deg': bv, 'after_total_deg': av, 'total_angle_reduced': reduced,
            'maximum_candidate_transit_deg': maximum, 'all_candidate_transits_within_limit': bounded,
            'passed': bool(same_boundary and quality and reduced and bounded)})
    return {'passed': all(r['passed'] for r in rows), 'scope': 'actual q versus actual q' if actual else 'planned q_path versus planned q_path only',
            'windows': rows, 'limit_deg': after['limit_deg'], 'focus_only_comparison': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path); p.add_argument('--reference', type=Path)
    p.add_argument('--tpg', type=Path); p.add_argument('--execution', type=Path)
    p.add_argument('--plan-only', action='store_true'); p.add_argument('--output', type=Path)
    p.add_argument('--baseline-json', type=Path, help='Require lower full-window rotation against this saved check report')
    p.add_argument('--limit-deg', type=float, default=180.); p.add_argument('--fail-on-red', action='store_true')
    args = p.parse_args(); run = args.run_dir
    if run is None and args.reference is None: run = BASELINE
    ref = args.reference or run/'alloc_and_reference.json'
    gp = args.tpg or (run/'tpg.json' if run and not args.plan_only else None)
    ep = args.execution or (run/'execution.json' if run and not args.plan_only else None)
    report = analyze(read(ref), read(gp) if gp else None, read(ep) if ep else None, args.limit_deg)
    report['input_files'] = {str(path.resolve()): sha(path) for path in (ref, gp, ep) if path}
    if args.baseline_json:
        report['window_non_relocation_check'] = compare_windows(report, read(args.baseline_json))
        report['baseline_comparison_source'] = {'path': str(args.baseline_json.resolve()), 'sha256': sha(args.baseline_json)}
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(HERE.resolve()): raise ValueError('Output must stay in results/base_rotation_investigation')
        if str(output) in report['input_files']: raise ValueError('Cannot overwrite raw input evidence')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    compact = {'run_id': report['run_id'], 'scope': report['scope'], 'task_count': report['task_count'],
               'red_tasks': report['red_tasks'], 'focus': [{k: r.get(k) for k in ['task','status','actual_start_s','actual_end_s','actual_elapsed_s']} |
                 {'base': {k: r.get('measured', r['planned_reference']).get(k) for k in ['start_deg','end_deg','range_deg',
                            'total_variation_deg','shortest_endpoint_circular_arc_deg','raw_jump_count','velocity_integral_agrees_with_position']}}
                 for r in report['focus_tasks']],
               'window_non_relocation_check': report.get('window_non_relocation_check'),
               'output': str(args.output) if args.output else None}
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
    if args.fail_on_red and (report['red_tasks'] or report.get('window_non_relocation_check', {}).get('passed') is False): raise SystemExit(2)


if __name__ == '__main__': main()
