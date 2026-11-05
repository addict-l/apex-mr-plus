"""Read raw completed-run evidence; write only the independent audit report."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
from collections import Counter
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apex.validation import DEFAULTS, validate_execution


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def finite(value): return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
def vector(value, n): return isinstance(value, (list, tuple)) and len(value) == n and all(finite(x) for x in value)
def sensor(row, part):
    v = row.get('snap', {}).get(part, {})
    return v.get('valid') is True and vector(v.get('force_N'), 3) and vector(v.get('torque_Nm'), 3)


def audit():
    out = ROOT/'results/apex'; base = ROOT/'backups/smooth_motion_baseline_20261104_01/results/apex'
    graph, execution, manifest = [read(out/name) for name in ('tpg.json', 'execution.json', 'pipeline_manifest.json')]
    summary, delivery, curves, rollout = [read(out/name) for name in ('summary.json', 'delivery.json', 'curve_smoothing_validation.json', 'nominal_geometry_validation.json')]
    comparison = read(ROOT/'results/smooth_motion_candidate/replanned_comparison.json')
    figure = read(ROOT/'results/smooth_motion_candidate/figures/figure_evidence.json')
    reference = read(out/'alloc_and_reference.json'); baseline_manifest = read(base/'pipeline_manifest.json')
    source_signature_data = read(out/'source_signature.json')
    expected_run = '20261104T163855Z-0804c434'
    req = {**DEFAULTS, **graph.get('validation_requirements', {})}
    reaudited = validate_execution(graph, execution, req)
    events = execution['events']; trace = execution['trace']; nodes = {n['id']: n for n in graph['nodes']}
    starts = [e for e in events if e['kind'] == 'node_started']; ends = [e for e in events if e['kind'] == 'node_completed']
    start_map = {e['node']: e['time'] for e in starts}; end_map = {e['node']: e['time'] for e in ends}
    complete = [e for e in events if e['kind'] == 'assembly_completed']
    last_completed = max(e['time'] for e in ends)
    edge_violations = [e for e in graph['edges'] if e['source'] not in end_map or e['target'] not in start_map or end_map[e['source']] > start_map[e['target']]+1e-7]
    node_coverage = (set(start_map) == set(end_map) == set(nodes) and len(starts) == len(ends) == len(nodes))
    permanent = [e for e in events if e['kind'] == 'permanent_structure_verified']
    proof_event = permanent[0] if len(permanent) == 1 else None
    proof_rows = [r for r in trace if r.get('permanent_verification_active') is True and proof_event and r['time'] <= proof_event['time']+1e-7]
    def qualified(r):
        return (r.get('bearing_sensor') == 'PermanentBridgeFT' and r.get('permanent_sensor_valid') is True and
            r.get('permanent_support_valid') is True and sensor(r, 'Bridge') and
            r.get('applied_load_N', -math.inf) >= req['required_validation_load_N']-1e-7 and
            r.get('applied_load_substeps', 0) >= 1 and
            r.get('permanent_force_N', -math.inf) >= req['permanent_force_min_N'] and
            abs(r.get('support_lift_position_m', math.inf)-req['support_retracted_position_m']) <= req['lift_tolerance_m'] and
            abs(r.get('support_force_N', math.inf)) <= req['unload_force_max_N'] and
            r.get('bridge_position_error_m', math.inf) <= req['goal_position_tolerance_m'] and
            r.get('bridge_orientation_error_rad', math.inf) <= req['goal_orientation_tolerance_rad'])
    final_proof = []
    for row in proof_rows:
        if not qualified(row): final_proof = []; continue
        if final_proof and row['time']-final_proof[-1]['time'] > req['permanent_proof_max_gap_s']+1e-7: final_proof = []
        final_proof.append(row)
    proof_duration = final_proof[-1]['time']-final_proof[0]['time'] if len(final_proof) > 1 else 0.
    fields = ['time', 'bearing_sensor', 'permanent_verification_active', 'permanent_sensor_valid', 'permanent_support_valid',
              'applied_load_N', 'applied_load_substeps', 'insertion_load_N', 'permanent_force_N', 'support_force_N',
              'support_lift_position_m', 'bridge_position_error_m', 'bridge_orientation_error_rad']
    proof = {'passed': bool(proof_event and final_proof and proof_duration+1e-7 >= req['permanent_proof_duration_s'] and
                           proof_event['time']-final_proof[-1]['time'] <= req['permanent_proof_max_gap_s']+1e-7),
             'dedicated_sensor': 'PermanentBridgeFT', 'total_proof_active_samples': len(proof_rows),
             'final_contiguous_qualified_samples': len(final_proof), 'qualified_observed_duration_s': proof_duration,
             'qualified_start_s': final_proof[0]['time'] if final_proof else None,
             'qualified_end_s': final_proof[-1]['time'] if final_proof else None,
             'sample_max_gap_s': max((b['time']-a['time'] for a, b in zip(final_proof, final_proof[1:])), default=None),
             'qualified_samples': [{k: r.get(k) for k in fields} for r in final_proof], 'verified_event': proof_event}
    input_checks = []
    for name, expected in manifest['input_fingerprint']['files'].items():
        path = ROOT/name; actual = sha(path) if path.exists() else None
        input_checks.append({'path': name, 'expected_sha256': expected, 'actual_sha256': actual, 'matches': actual == expected})
    fingerprint_dictionary_hash = digest(manifest['input_fingerprint']['files'])
    old_files = baseline_manifest['input_fingerprint']['files']; now_files = manifest['input_fingerprint']['files']
    config_checks = {k: {'before': v, 'after': now_files.get(k), 'same': v == now_files.get(k)} for k, v in old_files.items() if k.startswith('config/')}
    curve_rows = []
    for macro in curves['candidate_macros']:
        successful = [a['validation'] for a in macro['attempts'] if a.get('validation', {}).get('accepted')]
        v = successful[-1] if successful else {}
        okay = macro.get('accepted') is True and all(v.get(k) is True for k in ['accepted', 'valid', 'geometry_valid', 'tcp_corridor_valid', 'validation_performed', 'context_unchanged', 'progress_monotone_within_tolerance'])
        okay = okay and v.get('geometry_samples_checked', 0) == v.get('geometry_samples_submitted', -1) and v.get('geometry_samples_checked', 0) > 0
        curve_rows.append({'macro': macro['macro'], 'passed': okay, 'samples': v.get('geometry_samples_checked'),
            'max_tcp_deviation_m': v.get('max_tcp_deviation_m'), 'max_orientation_error_rad': v.get('max_orientation_error_rad'),
            'max_sample_joint_gap_rad': v.get('max_sample_joint_gap_rad'), 'thresholds': v.get('thresholds')})
    curve_profiles = {n['curve_run_id']: n['curve_profile'] for n in nodes.values() if n.get('curve_profile')}
    profile_checks = []
    for run, p in curve_profiles.items():
        limits = p['limits']; bounds = p['derivative_bounds']
        profile_checks.append({'run': run, 'collision_validated': p['collision_validated'], 'limits': limits,
                               'global_derivative_bounds': bounds['global_max']})
    main_path = Path(delivery['main_scene']); planned = Path(manifest['planned_scene'])
    main_hash, planned_hash = sha(main_path), sha(planned)
    identities = [execution.get('run_id'), manifest.get('run_id'), summary.get('run_id'), delivery.get('run_id'), figure['after_run_id']]
    dynamic = execution['dynamic_enablement']
    checks = {
        'run_identity_matches_all_evidence': all(r == expected_run for r in identities),
        'manifest_complete_and_promoted': manifest['status'] == 'complete' and manifest.get('main_scene_promoted') is True,
        'all_actual_nodes_start_and_complete_once': node_coverage,
        'all_new_graph_edges_observed_in_actual_order': not edge_violations,
        'completion_time_matches_raw_events': len(complete) == 1 and abs(complete[0]['time']-last_completed) < 1e-7 and abs(execution['makespan']-last_completed) < 1e-7 and abs(summary['actual_makespan_s']-last_completed) < 1e-7,
        'actual_validator_reaudit_passed': reaudited['success'],
        'dedicated_8N_actual_callback_continuous_proof_passed': proof['passed'],
        'all_12_arm_joints_dynamic': all(len(dynamic[a]['joints']) == 6 and all(v is True for v in dynamic[a]['joints']) for a in ('A', 'B')),
        'all_4_connections_dynamic_and_final_force_telemetry_valid': all(execution['snap_dynamic'].get(p) is True and sensor(trace[-1], p) for p in ('K1', 'K2', 'Gate', 'Bridge')),
        'support_sensor_and_lift_dynamic': dynamic.get('support_sensor') is True and dynamic.get('support_lift') is True,
        'all_frozen_input_file_hashes_match': all(x['matches'] for x in input_checks),
        'input_dictionary_hash_matches_declared_fingerprint': fingerprint_dictionary_hash == manifest['input_fingerprint']['sha256'],
        'full_resolved_source_signature_matches_all_run_artifacts': digest(source_signature_data) == manifest['scene_signature'] == execution['scene_signature'] == graph['scene_signature'],
        'resolved_source_signature_uses_same_inputs_settings_environment': source_signature_data['inputs'] == manifest['input_fingerprint'] and source_signature_data['settings'] == manifest['planning_settings'] and source_signature_data['environment'] == manifest['environment'],
        'reference_signature_matches_actual_chosen_tasks': digest(reference['tasks']) == graph['reference_signature'] == execution['reference_signature'] == manifest['reference_signature'],
        'physical_scene_hash_unchanged': old_files['scenes/10_Scene3_Dynamic.ttt'] == now_files['scenes/10_Scene3_Dynamic.ttt'],
        'configuration_hashes_unchanged': bool(config_checks) and all(x['same'] for x in config_checks.values()),
        'all_6_curved_macros_have_geometry_and_corridor_evidence': len(curve_rows) == curves['accepted_macros'] == 6 and all(r['passed'] for r in curve_rows),
        'all_curved_profiles_geometry_validated': len(curve_profiles) == 6 and all(p['collision_validated'] is True for p in curve_profiles.values()),
        'curved_profile_command_derivative_bounds_within_declared_limits': all(
            all(vector(p['derivative_bounds'].get(k), 6) and vector(p['limits'].get(k), 6) and
                all(0 <= bound <= limit*(1+1e-8) for bound, limit in zip(p['derivative_bounds'][k], p['limits'][k]))
                for k in ('velocity', 'acceleration', 'jerk')) for p in curve_profiles.values()),
        'full_new_geometry_rollout_complete_and_valid': rollout['valid'] is True and rollout['coverage_complete'] is True and rollout['tested_samples'] == rollout['planned_samples'] and not rollout['world_state_gaps'] and not rollout['missing_source_poses'],
        'all_phase_conflicts_rebuilt_for_new_geometry': graph['certificate'].get('conflicts_rebuilt_for_current_node_geometry') is True and graph['certificate'].get('motion_overlap_pairs_checked') is True,
        'semantic_guard_comparison_passed': comparison['semantic_guard_comparison']['accepted'],
        'delivery_main_planned_and_manifest_hashes_match': main_hash == planned_hash == delivery['main_scene_sha256'] == manifest['planned_scene_sha256'],
    }
    report = {'schema_version': 2, 'run_id': expected_run, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'success': all(v is True for v in checks.values()), 'checks': checks,
        'raw_event_timing': {'first_node_started_s': min(start_map.values()), 'last_node_completed_s': last_completed,
            'assembly_completed_s': complete[0]['time'], 'reported_actual_makespan_s': execution['makespan'],
            'final_get_result_snapshot_time_s': execution.get('time'), 'last_trace_time_s': trace[-1]['time'],
            'time_basis': 'Elapsed simulation time from 0 to actual assembly_completed; includes initial 0.6 s settling. Later polling/trace samples are not makespan.',
            'expected_nodes': len(nodes), 'started_events': len(starts), 'completed_events': len(ends),
            'actual_edges_checked': len(graph['edges']), 'edge_violations': edge_violations},
        'raw_semantic_events': [e for e in events if e['kind'] not in ('node_started', 'node_completed', 'grasp', 'buffer_deposit')],
        'raw_qualified_permanent_proof': proof,
        'dynamic_enablement': dynamic, 'snap_dynamic': execution['snap_dynamic'], 'final_snap_telemetry': trace[-1]['snap'],
        'final_state': execution['final_state'],
        'actual_validator_reaudit': {k: v for k, v in reaudited.items() if k != 'actual_node_timestamps'},
        'baseline_comparison': {'baseline_run_id': comparison['legacy_node_comparison']['before']['run_id'],
            'legacy_accepted': comparison['legacy_node_comparison']['comparison_accepted'],
            'legacy_failed_checks': [k for k, v in comparison['legacy_node_comparison']['comparability_checks'].items() if v is not True],
            'legacy_why': 'Old node-ID reachability retains 37 conservative failures after changed geometry/repartition. All task semantics, new graph dependencies and new geometry certificates pass separately; legacy result was not overwritten.',
            'semantic_accepted': comparison['semantic_guard_comparison']['accepted'],
            'actual_makespan_reduction_s': comparison['semantic_guard_comparison']['actual_makespan_reduction_s'],
            'actual_makespan_reduction_fraction': comparison['semantic_guard_comparison']['actual_makespan_reduction_fraction'],
            'all_pose_boundaries': figure['all_pose_boundary_statistics'], 'ordinary_high_speed_boundaries': figure['ordinary_boundary_statistics'],
            'metrics': {label: {k: v for k, v in comparison['legacy_node_comparison'][label].items() if k in
                ('actual_makespan_s', 'arms', 'skills', 'kinematics', 'assembly_contact', 'goal_errors', 'path_total_joint_L2_rad')} for label in ('before', 'after')}},
        'curve_geometry_checks': curve_rows, 'curved_profile_command_bounds': profile_checks,
        'new_geometry_certificate': graph['certificate'], 'nominal_geometry_rollout': {k: v for k, v in rollout.items() if k != 'batch_statistics'},
        'input_integrity': {'count': len(input_checks), 'dictionary_sha256': fingerprint_dictionary_hash, 'files': input_checks, 'same_configs': config_checks},
        'delivery': {'main_scene': str(main_path), 'actual_main_sha256': main_hash, 'actual_planned_sha256': planned_hash,
                     'original_physical_scene_sha256': now_files['scenes/10_Scene3_Dynamic.ttt']},
        'artifact_hashes': {name: sha(out/name) for name in ('execution.json', 'tpg.json', 'summary.json', 'pipeline_manifest.json', 'delivery.json', 'curve_smoothing_validation.json', 'nominal_geometry_validation.json', 'execution_validation.json', 'alloc_and_reference.json', 'source_signature.json')},
        'limitations': ['Two simulation runs, no repeated robustness study and no real-hardware validation.',
            'Actual velocity differentiation uses unequal 0.02–0.10 s intervals; acceleration and jerk are interval estimates, not continuous extrema.',
            'The new commanded profile jerk limit is 8 rad/s^3. Actual sampled jerk estimates exceed 8 (A about 13.30, B about 11.46); no claim that measured physical jerk is bounded by that command limit.',
            'Changed path geometry, C2 curves, continuous execution and skill timing all contribute; the total speedup is not isolated smoothing causality.',
            'All collision evidence is sampled; ideal rigid grasp and snap constraints remain simulation assumptions.',
            'ILP objective values use normalization/proxy costs and are not an execution-time improvement metric.']}
    target = out/'independent_final_audit.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(target), 'run_id': expected_run, 'success': report['success'],
        'failed_checks': [k for k, v in checks.items() if v is not True], 'event_finish_s': last_completed,
        'proof_duration_s': proof_duration, 'proof_samples': len(final_proof), 'input_files_checked': len(input_checks)}, indent=2))


if __name__ == '__main__': audit()
