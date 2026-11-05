"""Independent postprocessing for replanned Scene3 runs; no simulator calls.

The frozen apex.performance comparison is retained verbatim. A separately named
semantic_guard_comparison permits new pose partition IDs and new collision
edges, while checking task semantics, selected support order and new geometry.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apex.performance import compare_runs
from apex.validation import DEFAULTS


SCHEMA = {
    'schema_version': 1,
    'legacy_node_comparison': 'Unmodified apex.performance.compare_runs result, including its original rejection.',
    'semantic_guard_comparison': {
        'accepted': 'All separately documented comparison checks passed.',
        'validated_replanned_speedup': 'accepted AND after actual makespan is shorter.',
        'checks': 'Named true/false/unknown comparison conditions.',
        'task_guard_report': 'Task-ID reachability, actual task timing, skill identity, selected support order, buffer events.',
        'geometry_reports': 'New graph sampled-oracle certificate plus independently stored full rollout reports.',
    },
    'scope': 'Two observed full runs. New route geometry and support order may differ; no isolated causal attribution to smoothing.',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def phase(task):
    if task.startswith(('pick_gate', 'close_gate')):
        return 'gate'
    if task.startswith(('unload_support', 'remove_support', 'regrasp_support', 'raise_unloaded_support',
                        'extract_support', 'lift_extracted_support', 'return_support')):
        return 'removal'
    return 'other'


def task_guard_report(before, after, execution, allocation):
    """Check semantic order without pretending old geometric conflicts persist."""
    nodes = {n['id']: n for n in after['nodes']}
    adjacency = defaultdict(set)
    for edge in after['edges']:
        adjacency[edge['source']].add(edge['target'])
    reachable = {}
    def reaches(source, target):
        if source not in reachable:
            seen = set(); pending = list(adjacency[source])
            while pending:
                item = pending.pop()
                if item not in seen:
                    seen.add(item); pending.extend(adjacency[item]-seen)
            reachable[source] = seen
        return source != target and target in reachable[source]
    task_nodes = {t: row['node_ids'] for t, row in after['task_map'].items()}
    starts = {}; ends = {}
    for event in execution.get('events', []):
        if event.get('kind') == 'node_started': starts[event['node']] = event['time']
        if event.get('kind') == 'node_completed': ends[event['node']] = event['time']
    checks = []; issues = []
    def require(ok, message):
        if not ok: issues.append(message)
    def ordered(source, target, reason):
        valid = source in task_nodes and target in task_nodes
        graph_ok = valid and reaches(task_nodes[source][-1], task_nodes[target][0])
        actual_ok = valid and all(n in ends for n in task_nodes[source]) and all(n in starts for n in task_nodes[target])
        if actual_ok:
            actual_ok = max(ends[n] for n in task_nodes[source]) <= min(starts[n] for n in task_nodes[target])+1e-7
        checks.append({'source_task': source, 'target_task': target, 'reason': reason,
                       'graph_reachable': bool(graph_ok), 'actual_ordered': bool(actual_ok)})
        require(graph_ok and actual_ok, f'{reason}: {source} -> {target}')

    # Compiler task_dependency edges include the chosen before/after Gate
    # serialization. Only the cross-phase edges in this explicit list can vary.
    old_nodes = {n['id']: n for n in before['nodes']}
    preserved = set(); selection_sensitive = []
    for edge in before['edges']:
        if edge.get('reason') != 'task_dependency': continue
        source = edge.get('source_task') or old_nodes[edge['source']]['task_id']
        target = edge.get('target_task') or old_nodes[edge['target']]['task_id']
        if source == target: continue
        if {phase(source), phase(target)} == {'gate', 'removal'}:
            selection_sensitive.append({'source_task': source, 'target_task': target})
        else: preserved.add((source, target))
    for source, target in sorted(preserved): ordered(source, target, 'retained_task_dependency')

    # Skill identity persists even when transit paths have new fragment IDs.
    old_skills = {n['task_id']: n for n in before['nodes'] if n['node_type'] == 'skill'}
    new_skills = defaultdict(list)
    for node in nodes.values():
        if node['node_type'] == 'skill': new_skills[node['task_id']].append(node)
    for task, old in old_skills.items():
        matches = new_skills[task]
        require(len(matches) == 1, 'missing/duplicate skill: '+task)
        if len(matches) == 1:
            new = matches[0]
            require((new.get('part'), new.get('metadata', {}).get('action')) ==
                    (old.get('part'), old.get('metadata', {}).get('action')), 'changed skill semantics: '+task)
    guarded_tasks = {n['task_id'] for n in before['nodes'] if n.get('metadata', {}).get('assembly_force_guard')}
    for task in guarded_tasks:
        require(task in task_nodes and all(nodes[n].get('metadata', {}).get('assembly_force_guard') is True
                                          for n in task_nodes.get(task, [])), 'assembly force guard lost: '+task)

    core = [('place_support', 'verify_support'), ('verify_support', 'release_A'),
            ('release_A', 'supply_K1_approach'), ('release_A', 'supply_K2_approach'),
            ('install_K1', 'install_K2_approach'), ('install_K2', 'verify_permanent'),
            ('verify_permanent', 'unload_support'), ('unload_support', 'remove_support_approach'),
            ('unload_support', 'regrasp_support'), ('regrasp_support', 'extract_support'),
            ('extract_support', 'return_support'), ('verify_permanent', 'pick_gate_approach')]
    for source, target in core: ordered(source, target, 'lifecycle_invariant')
    selected = allocation.get('selected_support') or {}
    candidate, order = selected.get('candidate'), selected.get('removal_order')
    require(candidate in ('outer_front', 'inner_pocket', 'weak_narrow'), 'missing selected support candidate')
    require(order in ('before_gate', 'after_gate'), 'missing selected removal order')
    if order == 'before_gate': ordered('return_support', 'pick_gate_approach', 'selected_before_gate_order')
    if order == 'after_gate': ordered('close_gate', 'unload_support', 'selected_after_gate_order')
    if candidate == 'inner_pocket':
        require(order == 'before_gate', 'inner support cannot be removed after gate closure')
        ordered('extract_support', 'pick_gate_approach', 'inner_support_extraction_clearance')
    require(after.get('validation_requirements', {}).get('gate_blocks_extraction') == (candidate == 'inner_pocket'),
            'gate-extraction audit requirement disagrees with selected geometry')

    handoffs = allocation.get('handoffs', [])
    parts = [h['part'] for h in handoffs]
    for part in parts:
        for source, target in [(f'supply_{part}', f'deposit_{part}_approach'),
                               (f'deposit_{part}', f'pick_{part}_approach'),
                               (f'deposit_{part}_retreat_contact_clear', f'pick_{part}_approach'),
                               (f'pick_{part}', f'install_{part}_approach')]:
            ordered(source, target, 'material_handoff')
    for prior, following in zip(parts, parts[1:]):
        ordered('pick_'+prior, 'deposit_'+following+'_approach', 'buffer_capacity_one')
    # Inspect actual buffer ownership independently of declared graph edges.
    slot = None; deposited = Counter(); received = Counter(); source_grasps = Counter()
    for event in sorted(execution.get('events', []), key=lambda e: e['time']):
        kind = event.get('kind'); part = event.get('details', {}).get('part')
        if kind == 'buffer_deposit':
            require(part in parts and slot is None, 'actual buffer deposit violates part/capacity')
            slot = part; deposited[part] += 1
        elif kind == 'grasp' and part in ('K1', 'K2'):
            if slot == part:
                chosen = next((h for h in handoffs if h['part'] == part), {})
                require(event.get('arm') == chosen.get('to_arm'), 'wrong buffer receiver: '+part)
                slot = None; received[part] += 1
            else:
                require(event.get('arm') == 'A', 'A-exclusive actual source grasp violated: '+part)
                source_grasps[part] += 1
    require(slot is None, 'buffer occupied at final state')
    require(all(deposited[p] == received[p] == 1 for p in parts), 'missing/duplicate actual buffer handoff')
    require(all(source_grasps[p] == 1 for p in ('K1', 'K2')), 'missing/duplicate actual A source grasp')
    semantics = defaultdict(list)
    for event in execution.get('events', []): semantics[event.get('kind')].append(event)
    first, second = ('support_returned', 'gate_closed') if order == 'before_gate' else ('gate_closed', 'support_unloaded')
    actual_selected_order = len(semantics[first]) == len(semantics[second]) == 1 and semantics[first][0]['time'] <= semantics[second][0]['time']+1e-7
    require(actual_selected_order, 'actual semantic events violate selected support order')
    return {'passed': not issues, 'violations': issues, 'checks': checks,
            'selected_support': {'candidate': candidate, 'removal_order': order},
            'preserved_skill_count': len(old_skills), 'order_specific_old_edges_replaced': selection_sensitive,
            'old_motion_conflict_edges_not_preserved_by_ID': sum(e.get('reason') == 'motion_conflict' for e in before['edges']),
            'buffer': {'source_grasps': dict(source_grasps), 'deposits': dict(deposited), 'receives': dict(received), 'final_occupant': slot}}


def geometry_evidence(graph, rollout):
    cert = graph.get('certificate', {})
    flags = ['dag', 'motion_overlap_pairs_checked', 'collision_oracle_supplied', 'all_nodes_have_geometry',
             'waiting_endpoints_verified', 'nominal_world_state_verified']
    good = all(cert.get(k) is True for k in flags)
    good = good and all(rollout.get(k) is True for k in ['valid', 'coverage_complete', 'no_sampled_collision',
                                                       'nominal_world_state_reconstructed', 'stationary_waiting_poses_included'])
    good = good and rollout.get('planned_samples', 0) > 0 and rollout.get('tested_samples') == rollout.get('planned_samples')
    empty = lambda value: value == [] or (type(value) is int and value == 0)
    good = good and rollout.get('first_violation') is None and empty(rollout.get('world_state_gaps')) and empty(rollout.get('missing_source_poses'))
    good = good and abs(rollout.get('nominal_makespan_s', -1)-graph.get('nominal_makespan', -2)) < 1e-6
    return {'passed': bool(good), 'certificate': cert,
            'rollout': {k: rollout.get(k) for k in ['valid', 'step_s', 'tested_samples', 'planned_samples',
                         'coverage_complete', 'stationary_waiting_poses_included', 'first_violation']},
            'scope': 'Fresh sampled swept-path oracle plus full nominal world-state/idle rollout; not continuous collision certification.'}


def compare_replanned(before_dir, after_dir):
    before_dir, after_dir = Path(before_dir), Path(after_dir)
    names = ['tpg.json', 'execution.json', 'pipeline_manifest.json', 'allocation.json', 'nominal_geometry_validation.json']
    bundle = [{name: read(directory/name) for name in names} for directory in [before_dir, after_dir]]
    b, a = bundle
    legacy = compare_runs(b['tpg.json'], b['execution.json'], a['tpg.json'], a['execution.json'],
                          before_manifest=b['pipeline_manifest.json'], after_manifest=a['pipeline_manifest.json'])
    guard = task_guard_report(b['tpg.json'], a['tpg.json'], a['execution.json'], a['allocation.json'])
    geometry = {label: geometry_evidence(x['tpg.json'], x['nominal_geometry_validation.json']) for label, x in [('before', b), ('after', a)]}
    # Keep this separate: the old node-level result is neither patched nor
    # relabelled accepted. Explain exactly which new comparison rules apply.
    inherited = ['same_physical_source_scene', 'same_environment', 'same_global_and_task_motion_limits',
                 'before_actual_audit_passed', 'after_actual_audit_passed', 'complete_actual_event_coverage',
                 'before_graph_matches_actual_signature', 'after_graph_matches_actual_signature',
                 'A_velocity_sampling_comparable', 'B_velocity_sampling_comparable', 'observed_velocity_within_global_limit']
    checks = {key: legacy['comparability_checks'][key] for key in inherited}
    br = {**DEFAULTS, **b['tpg.json'].get('validation_requirements', {})}
    ar = {**DEFAULTS, **a['tpg.json'].get('validation_requirements', {})}
    br.pop('gate_blocks_extraction'); ar.pop('gate_blocks_extraction')
    checks['same_numeric_and_evidence_acceptance_requirements'] = br == ar
    checks['material_support_buffer_and_selected_gate_guards'] = guard['passed']
    checks['fresh_geometry_certificates_and_complete_rollouts'] = all(x['passed'] for x in geometry.values())
    bm, am = b['pipeline_manifest.json'], a['pipeline_manifest.json']
    bf, af = bm.get('input_fingerprint', {}).get('files', {}), am.get('input_fingerprint', {}).get('files', {})
    config_keys = {k for k in bf if k.startswith('config/')}
    checks['same_configuration_file_hashes'] = bool(config_keys) and all(bf[k] == af.get(k) for k in config_keys)
    checks['completed_manifest_matches_actual_runs'] = all(x['pipeline_manifest.json'].get('status') == 'complete' and
        x['pipeline_manifest.json'].get('run_id') == x['execution.json'].get('run_id') for x in bundle)
    checks['sampling_resolution_not_relaxed'] = all(am.get('planning_settings', {}).get(k, float('inf')) <= bm.get('planning_settings', {}).get(k, -1)
                                                   for k in ['joint_conflict_step_rad', 'rollout_step_s']) and (
        a['nominal_geometry_validation.json'].get('step_s', float('inf')) <= b['nominal_geometry_validation.json'].get('step_s', -1))
    stages = {x['name']: x.get('status') for x in am.get('stages', [])}
    checks['new_pipeline_geometry_and_actual_stages_passed'] = all(stages.get(k) == 'passed' for k in
        ['full_nominal_geometry_and_waiting_pose_validation', 'actual_dynamic_execution', 'independent_actual_event_load_and_pose_audit'])
    accepted = all(value is True for value in checks.values())
    reduction = legacy['actual_makespan_reduction_s']
    jerk_limits = {label: x['tpg.json'].get('time_parameterization', {}).get('max_jerk_rad_s3')
                   for label, x in [('before', b), ('after', a)]}
    return {'schema_version': 1, 'schema': SCHEMA, 'legacy_node_comparison': legacy,
            'semantic_guard_comparison': {'accepted': accepted, 'checks': checks, 'task_guard_report': guard,
                'geometry_reports': geometry, 'validated_replanned_speedup': bool(accepted and reduction is not None and reduction > 1e-7),
                'actual_makespan_reduction_s': reduction,
                'actual_makespan_reduction_fraction': legacy['actual_makespan_reduction_fraction'],
                'configured_jerk_limits_rad_s3': jerk_limits,
                'old_geometry_edge_policy': 'Old motion_conflict edges are not required under changed paths. New actual edges and new sampled geometry/complete rollout must pass.',
                'support_order_policy': 'New selected before_gate/after_gate policy is enforced; old choice-specific Gate/removal serialization is not immutable.'},
            'inputs': {str(directory/name): hashlib.sha256((directory/name).read_bytes()).hexdigest()
                       for directory in [before_dir, after_dir] for name in names},
            'limitations': ['New geometry and support order can change; speedup cannot be attributed to smoothing alone.',
                            'Boundary counts have different denominators after repartition; report observations and coverage, not only raw stop-count reduction.',
                            'Same physical/configuration hashes and actual feedback audits do not provide continuous collision or jerk guarantees.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('replanned_comparison.json'))
    parser.add_argument('--describe-schema', action='store_true')
    args = parser.parse_args()
    if args.describe_schema:
        print(json.dumps(SCHEMA, ensure_ascii=False, indent=2)); return
    if not args.before or not args.after: parser.error('--before and --after are required')
    report = compare_replanned(args.before, args.after)
    forbidden = {path.resolve() for directory in [args.before, args.after] for path in directory.glob('*.json')}
    if args.output.resolve() in forbidden: parser.error('Output must not overwrite a source evidence artifact')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'semantic_accepted': report['semantic_guard_comparison']['accepted'],
                      'legacy_accepted': report['legacy_node_comparison']['comparison_accepted']}, indent=2))


if __name__ == '__main__':
    main()
