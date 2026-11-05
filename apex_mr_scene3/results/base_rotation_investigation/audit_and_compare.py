"""Audit a NEW completed base-rotation run and export independent comparison.

Never writes production files or results/apex. Neither node nor C2-group counts
are fixed. Raw measured positions determine rotation; velocity integration is
only a separately reported diagnostic. No simulator calls are made.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'results/smooth_motion_candidate'))
from check_base_rotation import analyze, compare_windows, read, sha, digest, finite, angle, BASELINE, GROUPS
from replanned_comparison import task_guard_report, geometry_evidence
from apex.performance import summarize_run
from apex.validation import DEFAULTS, validate_execution
from apex.curve import _timing

OLD_RUN = '20261104T163855Z-0804c434'


def require_new_completed(manifest, execution, expected_run, baseline_run=OLD_RUN):
    if not expected_run or expected_run == baseline_run:
        raise ValueError('Expected run must be a new run, not the 317.08-second baseline')
    if manifest.get('run_id') != expected_run or execution.get('run_id') != expected_run:
        raise ValueError('Requested run ID differs from manifest/execution')
    if manifest.get('status') != 'complete' or not manifest.get('main_scene_promoted'):
        raise ValueError('New run is not complete and promoted')
    if execution.get('success') is not True or execution.get('done') is not True:
        raise ValueError('New actual execution has not completed successfully')


def command_bounds(graph):
    """Recompute quintic extrema and C2 control-hull bounds from geometry."""
    import numpy as np
    timing = graph['time_parameterization']
    names = ['velocity', 'acceleration', 'jerk']
    keys = ['max_velocity_rad_s', 'max_acceleration_rad_s2', 'max_jerk_rad_s3']
    globals_ = [timing[k] for k in keys]
    if not all(finite(x) and x > 0 for x in globals_): raise ValueError('Invalid global motion limits')
    failures = []; profiles = {}; curve_rows = []; straight_count = 0; maxima = [0., 0., 0.]
    for node in graph['nodes']:
        limits = [min(v, node.get('metadata', {}).get(k, v)) for k, v in zip(keys, globals_)]
        if node.get('curve_profile'):
            profile = node['curve_profile']; key = node['curve_run_id']; signature = digest(profile)
            if key in profiles and profiles[key]['signature'] != signature:
                failures.append(node['id']+': inconsistent curve geometry within run'); continue
            if key not in profiles:
                nominal, bounds, _ = _timing(deepcopy(profile['pieces']), profile['length'],
                    np.asarray([profile['limits'][n] for n in names]))
                duration = profile['duration']
                if not finite(duration) or duration <= 0: raise ValueError('Invalid curve duration')
                recomputed = bounds*np.asarray([(nominal/duration)**order for order in (1, 2, 3)])[:, None]
                reported = np.asarray([profile['derivative_bounds'][n] for n in names])
                okay = bool(np.all(np.isfinite(recomputed)) and np.all(recomputed <= reported*(1+1e-8)+1e-9))
                okay = okay and profile.get('collision_validated') is True
                profiles[key] = {'signature': signature, 'bounds': recomputed, 'passed': okay}
                curve_rows.append({'curve_run': key, 'recomputed_maxima': {n: float(max(row)) for n, row in zip(names, recomputed)},
                    'recorded_bounds_not_understated': okay, 'duration_s': duration})
                if not okay: failures.append(key+': curve bound or geometry qualification invalid')
            values = [float(max(row)) for row in profiles[key]['bounds']]
            if any(v > limit*(1+1e-8)+1e-9 for v, limit in zip(values, limits)):
                failures.append(node['id']+': curve exceeds node motion limits')
            maxima = [max(a, b) for a, b in zip(maxima, values)]
        else:
            for segment in node.get('motion_segments', []):
                duration = segment.get('profile_duration', segment['duration'])
                if not finite(duration) or duration <= 0: raise ValueError('Invalid straight-profile duration')
                displacement = max(abs(b-a) for a, b in zip(segment['q0'], segment['q1']))
                values = [1.875*displacement/duration, (10/math.sqrt(3))*displacement/duration**2,
                          60*displacement/duration**3]
                straight_count += 1
                if any(v > limit*(1+1e-8)+1e-9 for v, limit in zip(values, limits)):
                    failures.append(node['id']+': quintic exceeds node motion limits')
                maxima = [max(a, b) for a, b in zip(maxima, values)]
    return {'passed': not failures, 'violations': failures, 'straight_segments_checked': straight_count,
            'curve_runs_checked': len(profiles), 'curve_reports': curve_rows,
            'recomputed_command_maxima': dict(zip(names, maxima)),
            'scope': 'Analytic commanded-profile extrema/control-hull bounds, not measured physical jerk certification'}


def curve_geometry(graph, evidence):
    curves = {n['curve_run_id']: n for n in graph['nodes'] if n.get('curve_profile')}
    rows = []
    for macro in evidence.get('candidate_macros', []):
        if not macro.get('accepted'): continue
        choices = [a['validation'] for a in macro.get('attempts', []) if a.get('validation', {}).get('accepted')]
        v = choices[-1] if choices else {}
        flags = ['accepted', 'valid', 'geometry_valid', 'tcp_corridor_valid', 'validation_performed', 'context_unchanged']
        okay = all(v.get(k) is True for k in flags)
        okay = okay and v.get('geometry_samples_checked', 0) > 0 and v.get('geometry_samples_checked') == v.get('geometry_samples_submitted')
        thresholds = v.get('thresholds', {})
        for field, limit in [('max_tcp_deviation_m', 'max_tcp_deviation_m'), ('max_orientation_error_rad', 'max_orientation_error_rad')]:
            okay = okay and finite(v.get(field)) and finite(thresholds.get(limit)) and v[field] <= thresholds[limit]+1e-8
        rows.append({'macro': macro['macro'], 'passed': bool(okay), 'task_ids': macro.get('task_ids', []),
                     'samples': v.get('geometry_samples_checked'), 'max_tcp_deviation_m': v.get('max_tcp_deviation_m'),
                     'max_orientation_error_rad': v.get('max_orientation_error_rad'), 'thresholds': thresholds})
    matching = all(any(node['task_id'] in row['task_ids'] for row in rows) for node in curves.values())
    return {'passed': len(rows) == len(curves) == evidence.get('accepted_macros', 0) and matching and all(r['passed'] for r in rows),
            'executed_curve_count': len(curves), 'accepted_evidence_count': len(rows), 'macros': rows,
            'scope': 'Curve count is derived from actual TPG; zero or varying counts are supported'}


def build_report(before_dir, after_dir, expected_run):
    before_dir, after_dir = Path(before_dir), Path(after_dir)
    m, e = read(after_dir/'pipeline_manifest.json'), read(after_dir/'execution.json')
    be = read(before_dir/'execution.json')
    require_new_completed(m, e, expected_run, be['run_id'])
    g, ref, allocation, rollout, curves, delivery, source = [read(after_dir/name) for name in
        ['tpg.json', 'alloc_and_reference.json', 'allocation.json', 'nominal_geometry_validation.json',
         'curve_smoothing_validation.json', 'delivery.json', 'source_signature.json']]
    bg, br, bm = [read(before_dir/name) for name in ['tpg.json', 'alloc_and_reference.json', 'pipeline_manifest.json']]
    req = {**DEFAULTS, **g.get('validation_requirements', {})}; old_req = {**DEFAULTS, **bg.get('validation_requirements', {})}
    actual = validate_execution(g, e, req); old_actual = validate_execution(bg, be, old_req)
    before_rotation, after_rotation = analyze(br, bg, be), analyze(ref, g, e)
    relocation = compare_windows(after_rotation, before_rotation)
    semantic = task_guard_report(bg, g, e, allocation)
    geometry = geometry_evidence(g, rollout); bounds = command_bounds(g); curve_check = curve_geometry(g, curves)
    fingerprints = []
    for name, wanted in m['input_fingerprint']['files'].items():
        path = ROOT/name; found = sha(path) if path.is_file() else None
        fingerprints.append({'path': name, 'expected': wanted, 'actual': found, 'matches': found == wanted})
    old_files, new_files = bm['input_fingerprint']['files'], m['input_fingerprint']['files']
    configs = {k: old_files[k] == new_files.get(k) for k in old_files if k.startswith('config/')}
    starts = [x for x in e['events'] if x['kind'] == 'node_started']; ends = [x for x in e['events'] if x['kind'] == 'node_completed']
    final_events = [x for x in e['events'] if x['kind'] == 'assembly_completed']; finish = max(x['time'] for x in ends)
    old_req.pop('gate_blocks_extraction', None); current_req = dict(req); current_req.pop('gate_blocks_extraction', None)
    def task_limits(graph):
        names = ['max_velocity_rad_s', 'max_acceleration_rad_s2', 'max_jerk_rad_s3']
        defaults = graph['time_parameterization']; result = {}
        for n in graph['nodes']:
            limits = tuple(min(defaults[k], n.get('metadata', {}).get(k, defaults[k])) for k in names)
            result.setdefault(n['task_id'], set()).add(limits)
        return result
    main_path, planned_path = Path(delivery['main_scene']), Path(m['planned_scene'])
    main_hash = sha(main_path); planned_hash = sha(planned_path)
    timing_equal = all(g['time_parameterization'].get(k) == bg['time_parameterization'].get(k) for k in
                       ['max_velocity_rad_s', 'max_acceleration_rad_s2', 'max_jerk_rad_s3'])
    pose_rows = [r for r in after_rotation['all_tasks'] if r['kind'] == 'transit']
    checks = {
        'new_successful_completed_run': True,
        'baseline_actual_audit_passes': old_actual['success'], 'new_actual_audit_passes': actual['success'],
        'actual_event_finish_matches_makespan': len(final_events) == 1 and abs(finish-final_events[0]['time']) <= 1e-7 and abs(finish-e['makespan']) <= 1e-7,
        'all_current_dependencies_and_semantic_guards_pass': semantic['passed'],
        'same_physical_source_scene': old_files['scenes/10_Scene3_Dynamic.ttt'] == new_files['scenes/10_Scene3_Dynamic.ttt'],
        'same_configuration_hashes': bool(configs) and all(configs.values()),
        'same_numeric_and_evidence_requirements': old_req == current_req,
        'same_global_and_task_command_limits': timing_equal and task_limits(bg) == task_limits(g),
        'sampled_conflicts_and_full_geometry_rollout_pass': geometry['passed'],
        'conflicts_rebuilt_on_current_geometry': g.get('certificate', {}).get('conflicts_rebuilt_for_current_node_geometry') is True,
        'geometry_sampling_not_coarsened': all(m['planning_settings'][k] <= bm['planning_settings'][k]+1e-12 for k in ['joint_conflict_step_rad','rollout_step_s']),
        'executed_curve_count_and_geometry_evidence_match': curve_check['passed'],
        'analytic_command_derivative_bounds_pass': bounds['passed'],
        'all_transit_position_windows_have_reliable_sampling': all(r.get('measured', {}).get('available') and r['measured'].get('unwrap_inference_reliable') for r in pose_rows),
        'no_actual_transit_exceeds_180deg': not after_rotation['red_tasks'],
        'three_complete_rotation_windows_improved_without_relocation': relocation['passed'],
        'frozen_input_hashes_match': all(x['matches'] for x in fingerprints),
        'fingerprint_dictionary_hash_matches': digest(m['input_fingerprint']['files']) == m['input_fingerprint']['sha256'],
        'resolved_source_signature_matches': digest(source) == m['scene_signature'] == e['scene_signature'] == g['scene_signature'],
        'source_record_matches_inputs_settings_environment': source['inputs'] == m['input_fingerprint'] and source['settings'] == m['planning_settings'] and source['environment'] == m['environment'],
        'chosen_tasks_match_reference_signature': digest(ref['tasks']) == m['reference_signature'] == e['reference_signature'] == g['reference_signature'],
        'main_scene_delivery_matches': main_hash == planned_hash == delivery['main_scene_sha256'] == m['planned_scene_sha256'],
        'delivery_matches_run_identity': delivery['run_id'] == expected_run and delivery['scene_signature'] == m['scene_signature'] and delivery['reference_signature'] == m['reference_signature'] and delivery['actual_execution_audited'] is True,
    }
    metrics = {'before': summarize_run(bg, be), 'after': summarize_run(g, e)}
    # The full per-node rotation reports are retained separately, avoiding a
    # giant duplicate while preserving exact input hashes and derived counts.
    concise_rotation = lambda r: {k: v for k, v in r.items() if k != 'all_tasks'}
    report = {'schema_version': 1, 'run_id': expected_run, 'baseline_run_id': be['run_id'],
        'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'accepted': all(v is True for v in checks.values()),
        'checks': checks, 'failed_checks': [k for k,v in checks.items() if v is not True],
        'actual_event_evidence': {'start_count': len(starts), 'completion_count': len(ends), 'expected_nodes': len(g['nodes']),
            'edges_checked': len(g['edges']), 'first_start_s': min(x['time'] for x in starts), 'last_completion_s': finish,
            'assembly_completed_s': final_events[0]['time'] if final_events else None, 'makespan_s': e['makespan'],
            'later_result_snapshot_time_s': e.get('time')},
        'actual_audit': {k:v for k,v in actual.items() if k != 'actual_node_timestamps'},
        'semantic_guard_audit': semantic, 'geometry_audit': geometry, 'curved_geometry_audit': curve_check,
        'command_bound_audit': bounds, 'rotation_comparison': {'before': concise_rotation(before_rotation),
            'after': concise_rotation(after_rotation), 'complete_window_non_relocation': relocation},
        'metrics': {label: {k:v for k,v in row.items() if k in ['actual_makespan_s','arms','skills','kinematics','assembly_contact','goal_errors']} for label,row in metrics.items()},
        'input_integrity': {'count': len(fingerprints), 'files': fingerprints, 'same_configs': configs},
        'delivery': {'main_scene_sha256': main_hash, 'planned_scene_sha256': planned_hash,
                     'original_physical_scene_sha256': new_files['scenes/10_Scene3_Dynamic.ttt']},
        'artifact_sha256': {str(d/name): sha(d/name) for d in (before_dir,after_dir) for name in
            ['tpg.json','execution.json','pipeline_manifest.json','alloc_and_reference.json','curve_smoothing_validation.json',
             'nominal_geometry_validation.json','source_signature.json','delivery.json']},
        'limitations': ['Base angle comes from measured positions, not integrating potentially inconsistent saved velocity channels.',
            'Unequal-time velocity differences are interval acceleration/jerk estimates; commanded jerk limits do not certify measured physical jerk.',
            'Collision evidence remains sampled. Grasp and snap connections are idealized simulator constraints.',
            'The 180-degree threshold is a task-quality regression check. Shortest endpoint circular arcs are not collision-free shortcut certificates.',
            'Two simulation runs are compared; no real-hardware or global-optimality claim. Complete guard windows overlap and are not summed.']}
    return report, before_rotation, after_rotation


def plot_comparison(report, before, after, output):
    if report['accepted'] is not True: raise ValueError('Refuse accepted-result plot for a failed independent audit')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists(): font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name() if font.exists() else 'sans-serif',
        'axes.unicode_minus': False, 'svg.fonttype': 'path', 'font.size': 10.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    colors = ['#7C858D','#007F86']; labels = ['修复前','修复后']; keys = list(GROUPS)
    names = ['取支架','放回支架','最后封板取装']; reports = [before,after]
    fig, axes = plt.subplots(1,2,figsize=(13,6.4)); fig.subplots_adjust(left=.075,right=.97,top=.82,bottom=.29,wspace=.28)
    bt, at = before['actual_makespan_s'], after['actual_makespan_s']
    fig.suptitle('基座长转修复：相同语义动作的实际转角与时间',x=.065,ha='left',y=.96,fontsize=15)
    fig.text(.065,.88,f'完整装配仿真时间：{bt:.2f} s → {at:.2f} s；三类动作采用相同任务起止定义。',fontsize=10.5)
    for axis, field, title, unit in [(axes[0],'angle','a  基座实际累计转角','累计转角（°）'),
                                  (axes[1],'time','b  动作实际耗时','仿真时间（s）')]:
        maximum = 0.
        for i,r in enumerate(reports):
            values = [r['semantic_groups'][k]['measured']['total_variation_deg'] if field == 'angle' else r['semantic_groups'][k]['actual_elapsed_s'] for k in keys]
            maximum = max(maximum,max(values)); x = [j+(i-.5)*.34 for j in range(3)]
            axis.bar(x,values,width=.31,color=colors[i],label=labels[i])
            for xx,value in zip(x,values): axis.annotate(f'{value:.1f}',(xx,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
        axis.set_title(title,loc='left',pad=12);axis.set_ylabel(unit);axis.set_xticks(range(3),names)
        axis.set_ylim(0,maximum*1.25);axis.grid(axis='y',color='#D7DDDF',lw=.6);axis.set_axisbelow(True);axis.legend()
    notes = ['转角由实际 qB[0] 连续采样累计绝对增量计算；不使用目标角度或速度积分代替位置。',
             '两次实际执行均通过载荷、依赖、位姿和几何验收；邻接扩展窗口另外检查，防止把长转搬到相邻动作。',
             f"运行编号：修复前 {before['run_id']}；修复后 {after['run_id']}。",
             '阶段累计角度可以超过180°；红测检查每段运动及完整邻接窗口的改善，不能以分拆任务逃避。',
             '结果来自CoppeliaSim动力学日志；不是实机实验或全局最优证明。']
    fig.text(.065,.025,'\n'.join(notes),fontsize=8.3,color='#435057',linespacing=1.6)
    files=[]
    for suffix in ('png','svg'):
        path=output/('base_rotation_comparison.'+suffix);fig.savefig(path,dpi=300 if suffix=='png' else None);files.append(str(path))
    plt.close(fig)
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-run',required=True);p.add_argument('--before',type=Path,default=BASELINE)
    p.add_argument('--after',type=Path,default=ROOT/'results/apex');p.add_argument('--output-dir',type=Path,default=HERE)
    args=p.parse_args();output=args.output_dir.resolve()
    if not output.is_relative_to(HERE.resolve()):raise ValueError('Outputs must stay within base_rotation_investigation')
    report,before,after=build_report(args.before,args.after,args.expected_run)
    output.mkdir(parents=True,exist_ok=True)
    figures=plot_comparison(report,before,after,output) if report['accepted'] else []
    report['figures']=[{'path':name,'sha256':sha(name)}for name in figures]
    for name,value in [('independent_audit.json',report),('candidate_actual_rotation.json',after)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'accepted':report['accepted'],'run_id':report['run_id'],'failed_checks':report['failed_checks'],
        'audit':str(output/'independent_audit.json'),'figures':figures},ensure_ascii=False,indent=2))
    if not report['accepted']:raise SystemExit(2)


if __name__=='__main__':main()
