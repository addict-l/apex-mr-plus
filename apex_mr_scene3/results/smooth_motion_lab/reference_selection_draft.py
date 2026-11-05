"""Rank complete semantically equivalent references without collision claims.

This is an independent draft, not a pipeline hook. Reading input JSON is the
only I/O in choose_reference; neither original reference is changed. Ranking
uses current build_tpg + parameterize with no collision oracle or C2 smoothing.
The chosen reference still needs fresh curve, all-phase conflict, full nominal
geometry and actual-dynamics audits before it can be accepted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from collections.abc import Mapping

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apex.tpg import build_tpg
from apex.program import parameterize

IGNORED_METADATA = {'planner', 'path_source', 'planning_stats'}
REQUIRED_FUTURE_CHECKS = ['fresh_curve_geometry_validation', 'fresh_all_phase_motion_conflicts',
                          'full_nominal_geometry_and_waiting_pose_validation',
                          'actual_dynamic_execution', 'independent_actual_event_load_and_pose_audit']


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _arg(args, name, default=None):
    return args.get(name, default) if isinstance(args, Mapping) else getattr(args, name, default)


def _source(value, args, label):
    path = None
    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
        provenance = {'source_type': 'in_memory', 'canonical_content_sha256': value_hash(result)}
    else:
        path = Path(value).resolve()
        raw = path.read_bytes()
        result = json.loads(raw)
        provenance = {'source_type': 'file', 'path': str(path), 'file_sha256': hashlib.sha256(raw).hexdigest(),
                      'canonical_content_sha256': value_hash(result)}
    physical_hash = _arg(args, label+'_scene_sha256', _arg(args, 'physical_scene_sha256'))
    if path is not None:
        manifest_path = path.parent/'pipeline_manifest.json'
        if manifest_path.exists():
            manifest_raw = manifest_path.read_bytes()
            manifest = json.loads(manifest_raw)
            provenance['manifest_path'] = str(manifest_path)
            provenance['manifest_sha256'] = hashlib.sha256(manifest_raw).hexdigest()
            declared = manifest.get('input_fingerprint', {}).get('files', {}).get('scenes/10_Scene3_Dynamic.ttt')
            if physical_hash is not None and declared is not None and physical_hash != declared:
                raise ValueError(label+' supplied physical scene hash disagrees with its manifest')
            physical_hash = physical_hash or declared
    provenance['physical_scene_sha256'] = physical_hash
    return result, provenance


def _q(value):
    return isinstance(value, (list, tuple)) and len(value) == 6 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)


def _same_q(a, b, tolerance):
    # Deliberately no modulo 2*pi equivalence: winding affects actual motion.
    return _q(a) and _q(b) and max(abs(x-y) for x, y in zip(a, b)) <= tolerance


def _closure(reference, tolerance):
    tasks = reference.get('tasks', [])
    declared = reference.get('problem', {}).get('initial_joint_configurations') or reference.get('planner_info', {}).get('initialhome_q')
    starts, ends, issues = {}, {}, []
    if not isinstance(declared, Mapping) or set(declared) != {'A', 'B'} or not all(_q(q) for q in declared.values()):
        return {'valid': False, 'violations': ['Missing finite declared initial joints for A and B'], 'initial': {}, 'final': {}}
    current = deepcopy(dict(declared))
    for task in tasks:
        arm, path = task.get('arm'), task.get('q_path')
        if arm not in current or not isinstance(path, list) or not path or not all(_q(q) for q in path):
            issues.append(str(task.get('id'))+': invalid arm or finite q_path'); continue
        starts.setdefault(arm, path[0])
        if not _same_q(current[arm], path[0], tolerance):
            issues.append(task['id']+': joint path does not close with previous same-arm endpoint')
        if task.get('kind') != 'transit' and any(not _same_q(path[0], q, tolerance) for q in path[1:]):
            issues.append(task['id']+': skill unexpectedly contains joint motion')
        current[arm] = list(path[-1]); ends[arm] = list(path[-1])
    if set(starts) != {'A', 'B'}:
        issues.append('Both arms must have complete reference paths')
    return {'valid': not issues, 'violations': issues, 'initial': deepcopy(dict(declared)), 'final': ends,
            'joint_tolerance_rad': tolerance, 'joint_winding_equivalence_used': False}


def _task_semantics(task):
    return {k: deepcopy(v) for k, v in task.items() if k not in {'q_path', 'duration', 'metadata'}} | {
        'metadata': {k: deepcopy(v) for k, v in task.get('metadata', {}).items() if k not in IGNORED_METADATA}}


def _choice_semantics(allocation):
    choices = allocation.get('choices', [])
    if not isinstance(choices, list):
        return None
    return sorted([{k: deepcopy(c.get(k)) for k in
                    ('task', 'arm', 'grasp_offset', 'grasp_quaternion')} for c in choices],
                  key=lambda c: str(c['task']))


def _support_semantics(allocation):
    support = allocation.get('selected_support', {})
    fields = ('candidate', 'removal_order', 'placer_arm', 'remover_arm', 'goal_pose',
              'rated_load_N', 'contact_overlap_m', 'blocked_after_gate')
    result = {key: deepcopy(support.get(key)) for key in fields}
    evidence = support.get('endpoint_evidence', {})
    result['grasp_offset'] = deepcopy(evidence.get('grasp_offset'))
    result['grasp_quaternion'] = deepcopy(evidence.get('grasp_quaternion'))
    return result


def _rank(reference, args):
    graph = build_tpg(reference['tasks'], conflict_oracle=None, max_joint_span=float(_arg(args, 'span', .24)))
    parameterize(graph, vmax=float(_arg(args, 'vmax', .6)), amax=float(_arg(args, 'amax', 1.2)),
                 jmax=float(_arg(args, 'jmax', 8.)))
    collision_edges = [e for e in graph['edges'] if e.get('reason') == 'motion_conflict']
    if collision_edges:
        raise ValueError('Ranking must not inherit collision edges')
    length = sum(math.sqrt(sum((a-b)**2 for a, b in zip(q0, q1)))
                 for task in reference['tasks'] for q0, q1 in zip(task['q_path'], task['q_path'][1:]))
    return {'nominal_makespan_without_collision_edges_s': graph['nominal_makespan'],
            'serial_parameterized_service_proxy_s': graph['reference_makespan'],
            'joint_polyline_length_L2_rad': length, 'task_count': len(reference['tasks']),
            'node_count': len(graph['nodes']), 'collision_edge_count': len(collision_edges),
            'task_duration_proxy_s': {tid: item['duration'] for tid, item in graph['task_map'].items()},
            'collision_validated': False, 'actual_execution_claimed': False,
            'ranking_scope': 'Heuristic lexicographic ranking only; no collision edges, no C2 curve changes, no simulation.'}


def choose_reference(fresh_reference, baseline_reference, args):
    """Return a fresh wrapper with one whole unchanged task list and a report.

    Args accepts span/vmax/amax/jmax, expected_task_count (default 134), and
    joint_tolerance_rad (default 1e-8). Physical scene hashes come from adjacent
    manifests for file inputs; in-memory callers supply physical_scene_sha256
    or fresh_scene_sha256/baseline_scene_sha256. Unknown scene hash disqualifies
    historical tasks. All eligibility failures retain the fresh reference.
    """
    fresh, fresh_source = _source(fresh_reference, args, 'fresh')
    baseline, baseline_source = _source(baseline_reference, args, 'baseline')
    expected = int(_arg(args, 'expected_task_count', 134))
    tolerance = float(_arg(args, 'joint_tolerance_rad', 1e-8))
    if expected <= 0 or not math.isfinite(tolerance) or tolerance < 0 or tolerance > 1e-6:
        raise ValueError('Invalid task count or joint closure tolerance')
    ft, bt = fresh.get('tasks', []), baseline.get('tasks', [])
    fc, bc = _closure(fresh, tolerance), _closure(baseline, tolerance)
    if len(ft) != expected or not fc['valid']:
        raise ValueError('Fresh reference is incomplete or has unclosed joint paths: '+str(fc['violations']))
    fa, ba = fresh.get('allocation', {}), baseline.get('allocation', {})
    checks = {
        'same_known_physical_source_scene': bool(fresh_source['physical_scene_sha256']) and fresh_source['physical_scene_sha256'] == baseline_source['physical_scene_sha256'],
        'complete_expected_task_count': len(bt) == len(ft) == expected,
        'same_ordered_task_ids': [t.get('id') for t in bt] == [t.get('id') for t in ft],
        'same_task_skill_payload_dependency_and_contact_semantics': [_task_semantics(t) for t in bt] == [_task_semantics(t) for t in ft],
        'same_arm_assignments': bool(fa.get('task_assignments')) and fa.get('task_assignments') == ba.get('task_assignments'),
        'same_chosen_grasp_transforms': bool(fa.get('choices')) and _choice_semantics(fa) == _choice_semantics(ba),
        'same_support_candidate_order_arms_goal_and_grasp': bool(fa.get('selected_support')) and _support_semantics(fa) == _support_semantics(ba),
        'same_buffer_lifecycle_and_material_semantics': all(fa.get(k) == ba.get(k) for k in ('handoffs', 'required_buffer_edges', 'required_lifecycle_edges', 'assembly_order', 'buffer_capacity')),
        'baseline_joint_paths_close': bc['valid'],
        'same_declared_initial_joints': all(_same_q(fc['initial'].get(a), bc['initial'].get(a), tolerance) for a in ('A', 'B')),
        'same_final_joint_semantics': all(_same_q(fc['final'].get(a), bc['final'].get(a), tolerance) for a in ('A', 'B')),
        'same_validation_requirements_if_baseline_declares_them': 'validation_requirements' not in baseline or baseline['validation_requirements'] == fresh.get('validation_requirements'),
    }
    ranking = {'fresh': _rank(fresh, args)}
    eligible = all(v is True for v in checks.values())
    if eligible:
        try:
            ranking['baseline'] = _rank(baseline, args)
        except (ValueError, KeyError, TypeError) as error:
            eligible = False
            checks['baseline_parameterization_succeeded'] = False
            ranking['baseline'] = {'ranking_error': str(error)}
    key = lambda row: (row['nominal_makespan_without_collision_edges_s'], row['serial_parameterized_service_proxy_s'])
    chosen_name = 'baseline' if eligible and key(ranking['baseline']) < key(ranking['fresh']) else 'fresh'
    chosen = baseline if chosen_name == 'baseline' else fresh
    selected = deepcopy(fresh)
    selected['tasks'] = deepcopy(chosen['tasks'])
    selected['motion_stats'] = deepcopy(chosen.get('motion_stats', {}))
    selected['reference_signature'] = value_hash(selected['tasks'])
    selected['task_duration_proxy_s'] = deepcopy(ranking[chosen_name]['task_duration_proxy_s'])
    selected['reference_selection_provenance'] = {
        'selected': chosen_name, 'source': deepcopy(baseline_source if chosen_name == 'baseline' else fresh_source),
        'task_content_sha256': selected['reference_signature'], 'tasks_are_unmodified_source_content': True,
        'motion_stats_are_historical_planner_evidence': chosen_name == 'baseline',
        'collision_validated_by_selection': False, 'required_future_checks': REQUIRED_FUTURE_CHECKS[:],
        'wrapper_allocation_problem_validation_requirements': 'fresh reference, preserved verbatim'}
    report = {'schema_version': 1, 'selected': chosen_name, 'baseline_eligible': eligible,
              'eligibility_checks': checks, 'rejection_reasons': [k for k, v in checks.items() if v is not True],
              'candidate_ranking': ranking, 'rank_order': ['nominal_makespan_without_collision_edges_s', 'serial_parameterized_service_proxy_s'],
              'tie_policy': 'fresh', 'settings': {k: float(_arg(args, k, default)) for k, default in [('span', .24), ('vmax', .6), ('amax', 1.2), ('jmax', 8.)]},
              'sources': {'fresh': fresh_source, 'baseline': baseline_source}, 'joint_closure': {'fresh': fc, 'baseline': bc},
              'selected_reference_signature': selected['reference_signature'],
              'heuristic_only': True, 'collision_validated': False, 'actual_execution_claimed': False,
              'required_future_checks': REQUIRED_FUTURE_CHECKS[:],
              'limitations': ['No-conflict makespan is an optimistic ranking proxy, not a collision-certified schedule or measured time.',
                              'The complete historical task list is selected as a unit; local route fragments are never spliced.',
                              'Costs may change after new C2 geometry, conflict edges, load guards and actual dynamics.',
                              'Fresh allocation/problem/validation requirements and source metadata remain authoritative.']}
    return selected, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    for name, default in [('span', .24), ('vmax', .6), ('amax', 1.2), ('jmax', 8.)]:
        parser.add_argument('--'+name, type=float, default=default)
    args = parser.parse_args()
    selected, report = choose_reference(args.fresh, args.baseline, args)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__': main()
