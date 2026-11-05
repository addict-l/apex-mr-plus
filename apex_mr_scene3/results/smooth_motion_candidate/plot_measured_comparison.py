"""Publication-oriented Chinese figures from two completed simulation runs.

No simulator calls and no target velocities relabelled as measurements.
Produces derived figures only after both actual-execution audits pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from replanned_comparison import compare_replanned, read

COLORS = ['#7C858D', '#007F86']
LABELS = ['基线', '新版']


def finite_vector(value, size=6):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)


def velocity_series(graph, execution, task_prefix):
    """Align raw measured joint velocities to the actual first node start."""
    selected = [n for n in graph['nodes'] if n['task_id'] == task_prefix or n['task_id'].startswith(task_prefix+'_')]
    if not selected:
        raise ValueError('No matching actual task group: '+task_prefix)
    arms = {n['arm'] for n in selected}
    if len(arms) != 1:
        raise ValueError('Velocity panel requires exactly one arm')
    arm = next(iter(arms)); ids = {n['id'] for n in selected}
    starts = {e['node']: e['time'] for e in execution['events'] if e.get('kind') == 'node_started' and e.get('node') in ids}
    ends = {e['node']: e['time'] for e in execution['events'] if e.get('kind') == 'node_completed' and e.get('node') in ids}
    if set(starts) != ids or set(ends) != ids:
        raise ValueError('Missing actual start/completion for '+task_prefix)
    start, stop = min(starts.values()), max(ends.values())
    times = []; speeds = []; valid_times = []; missing = 0; previous = None
    for row in execution['trace']:
        t = row['time']
        if not start-1e-8 <= t <= stop+1e-8:
            continue
        velocity = row.get(arm+'_joint_velocity_rad_s')
        if previous is not None and t <= previous:
            raise ValueError('Nonincreasing actual trace timestamps')
        if previous is not None and t-previous > .15+1e-8:
            times.append((t+previous)/2-start); speeds.append(None)
        times.append(t-start)
        if finite_vector(velocity):
            speeds.append(max(abs(x) for x in velocity)); valid_times.append(t)
        else:
            speeds.append(None); missing += 1
        previous = t
    if not valid_times:
        raise ValueError('No measured joint velocity samples for '+task_prefix)
    return {'task_prefix': task_prefix, 'arm': arm, 'actual_start_s': start, 'actual_end_s': stop,
            'actual_elapsed_s': stop-start, 'time_from_actual_start_s': times,
            'max_abs_measured_joint_velocity_rad_s': speeds,
            'measured_sample_count': len(valid_times), 'missing_velocity_samples': missing,
            'maximum_observed_sample_gap_s': max((b-a for a, b in zip(valid_times, valid_times[1:])), default=None),
            'source_field': arm+'_joint_velocity_rad_s', 'no_interpolation_or_smoothing': True}


def ordinary_boundaries(graph, metrics):
    """Common subset definition, independent of new pass-through declarations.

    Adjacent same-arm pose nodes, both at global v/a, neither guarded assembly
    motion, and no actual predecessor-readiness wait. Skills are already absent
    from performance.boundary_metrics. All-boundary statistics remain reported.
    """
    nodes = {n['id']: n for n in graph['nodes']}
    timing = graph['time_parameterization']
    vmax, amax = timing['max_velocity_rad_s'], timing['max_acceleration_rad_s2']
    chosen = []
    for row in metrics['boundaries']['boundaries']:
        pair = [nodes[row['from']], nodes[row['to']]]
        if row['dependency_wait_s'] > 1e-7:
            continue
        if any(n.get('metadata', {}).get('assembly_force_guard') for n in pair):
            continue
        if any(min(vmax, n.get('metadata', {}).get('max_velocity_rad_s', vmax)) < vmax-1e-9 or
               min(amax, n.get('metadata', {}).get('max_acceleration_rad_s2', amax)) < amax-1e-9 for n in pair):
            continue
        chosen.append(row)
    observed = [r for r in chosen if r['near_zero'] is not None]
    near_zero = sum(bool(r['near_zero']) for r in observed)
    return {'eligible': len(chosen), 'observed': len(observed), 'near_zero': near_zero,
            'near_zero_fraction': near_zero/len(observed) if observed else None,
            'coverage_fraction': len(observed)/len(chosen) if chosen else None,
            'threshold_rad_s': metrics['boundaries']['speed_threshold_rad_s'],
            'sample_window_s': metrics['boundaries']['sample_window_s'],
            'definition': 'Adjacent same-arm poses; global v/a on both; no assembly-force guard; no actual predecessor wait.'}


def all_pose_boundaries(metrics):
    """Keep every eligible non-skill boundary, including slow/guarded motion."""
    source = metrics['boundaries']
    eligible, observed = source['eligible_boundaries'], source['observed_boundaries']
    return {'eligible': eligible, 'observed': observed, 'near_zero': source['near_zero_count'],
            'near_zero_fraction': source['near_zero_count']/observed if observed else None,
            'coverage_fraction': observed/eligible if eligible else None,
            'threshold_rad_s': source['speed_threshold_rad_s'],
            'sample_window_s': source['sample_window_s'],
            'definition': 'All consecutive same-arm pose-to-pose boundaries; includes slow motion, assembly guards and dependency waits; excludes skills.'}


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = ['Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams.update({'font.family': family, 'font.size': 10.5, 'axes.titlesize': 12,
        'axes.labelsize': 10.5, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .75,
        'grid.color': '#D5DBDF', 'grid.linewidth': .55, 'legend.frameon': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    import matplotlib.pyplot as plt
    return plt


def panel(axis, letter, title, ylabel):
    axis.set_title(letter+'  '+title, loc='left', pad=12)
    axis.set_ylabel(ylabel); axis.grid(axis='y'); axis.set_axisbelow(True)


def caption(fig, lines, y=.02):
    fig.text(.065, y, '\n'.join(lines), va='bottom', ha='left', fontsize=8.1,
             color='#3F4A52', linespacing=1.55)


def export(fig, output, name):
    paths = []
    for ext in ('png', 'svg'):
        path = output/(name+'.'+ext)
        fig.savefig(path, dpi=300 if ext == 'png' else None)
        paths.append(str(path))
    return paths


def render_comparison(before_dir, after_dir, output_dir, *, write_evidence=True):
    before_dir, after_dir, output = map(Path, (before_dir, after_dir, output_dir))
    forbidden = {before_dir.resolve(), after_dir.resolve()}
    if output.resolve() in forbidden:
        raise ValueError('Derived figure output must be separate from source artifact directories')
    report = compare_replanned(before_dir, after_dir)
    legacy = report['legacy_node_comparison']; semantic = report['semantic_guard_comparison']
    if not all(v['success'] for v in legacy['actual_audits'].values()):
        raise ValueError('Both actual execution audits must pass before plotting completed-run comparisons')
    graphs = [read(d/'tpg.json') for d in (before_dir, after_dir)]
    executions = [read(d/'execution.json') for d in (before_dir, after_dir)]
    manifests = [read(d/'pipeline_manifest.json') for d in (before_dir, after_dir)]
    if any(m.get('status') != 'complete' or m.get('run_id') != e.get('run_id') for m, e in zip(manifests, executions)):
        raise ValueError('Completed run-matched manifests are required')
    metrics = [legacy['before'], legacy['after']]
    all_boundaries = [all_pose_boundaries(m) for m in metrics]
    ordinary = [ordinary_boundaries(g, m) for g, m in zip(graphs, metrics)]
    series = {task: [velocity_series(g, e, task) for g, e in zip(graphs, executions)]
              for task in ('pick_support_approach', 'A_exit_from_support')}
    runline = '运行编号：基线 '+str(executions[0]['run_id'])+'；新版 '+str(executions[1]['run_id'])
    state = lambda value: '通过' if value is True else ('未知' if value is None else '未通过')
    auditline = ('两次实际执行均验收通过；重规划任务语义比较'+state(semantic['accepted'])+'。节点已重新分片，近零边界同时报告分母。')
    output.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib(); files = []

    # Figure 1: counts explicitly carry their observed denominators. No score
    # cards, no serial-model bar presented as an experimental baseline.
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 6.5), gridspec_kw={'width_ratios': [1, 1.1, 1.1]})
    fig.subplots_adjust(left=.065, right=.975, bottom=.32, top=.85, wspace=.38)
    fig.suptitle('双UR5积木装配：实际仿真工时与运动边界近零速度', x=.065, ha='left', y=.96, fontsize=15)
    panel(ax[0], 'a', '完整装配的实际工时', '仿真时间（s）')
    duration = [m['actual_makespan_s'] for m in metrics]
    ax[0].bar(LABELS, duration, width=.48, color=COLORS)
    ax[0].set_ylim(0, max(duration)*1.22)
    for i, value in enumerate(duration): ax[0].text(i, value+max(duration)*.025, f'{value:.2f} s', ha='center', fontsize=11)
    for axis, letter, title, statistics in (
        (ax[1], 'b', '全部非技能pose边界', all_boundaries),
        (ax[2], 'c', '普通高速pose边界子集', ordinary)):
        panel(axis, letter, title, '近零边界 / 已观测边界（%）')
        fractions = [100*x['near_zero_fraction'] if x['near_zero_fraction'] is not None else float('nan') for x in statistics]
        axis.bar(LABELS, fractions, width=.48, color=COLORS); axis.set_ylim(0, 119)
        axis.set_yticks([0, 25, 50, 75, 100])
        for i, row in enumerate(statistics):
            label = f"{row['near_zero']} / {row['observed']}" if row['observed'] else '无有效观测'
            y = fractions[i] if row['observed'] else 0
            axis.text(i, y+3., label, ha='center', fontsize=11)
        axis.set_xticks(range(len(LABELS)), [f"{LABELS[i]}\n观测 {row['observed']}/{row['eligible']}"
                                            for i, row in enumerate(statistics)])
        axis.tick_params(axis='x', labelsize=9.5, pad=7)
    caption(fig, ['全部边界：相邻同臂pose，保留慢速、装配区和依赖等待；普通高速子集：两侧均使用默认v/a、无装配力守卫、无实际前驱等待。',
                  '近零阈值 max|dq/dt| < 0.01 rad/s；取实际完成点±0.06 s内最近实测样本。分段不同，分母必须同时报告。',
                  '两种统计均排除技能前后边界；近零观测可能包含必要停车，不等于每一处均可消除。', auditline, runline])
    files += export(fig, output, '01_actual_time_and_boundary_stops'); plt.close(fig)

    # Figure 2: only actual joint velocity fields, drawn at actual timestamps.
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.1))
    fig.subplots_adjust(left=.085, right=.97, top=.88, bottom=.22, hspace=.48)
    fig.suptitle('原始实测关节速度：按各动作组实际起点对齐', x=.065, ha='left', y=.97, fontsize=15)
    task_names = [('pick_support_approach', 'B臂接近临时支撑'), ('A_exit_from_support', 'A臂退出初始托举位置')]
    for axis, (letter, (task, label)) in zip(axes, zip(('a', 'b'), task_names)):
        panel(axis, letter, label+'  ·  '+task, 'max |关节角速度|（rad/s）')
        for i, line in enumerate(series[task]):
            y = [x if x is not None else float('nan') for x in line['max_abs_measured_joint_velocity_rad_s']]
            axis.plot(line['time_from_actual_start_s'], y, color=COLORS[i], lw=1.25 if i == 0 else 1.55,
                      alpha=.92, label=f"{LABELS[i]}：实际动作组 {line['actual_elapsed_s']:.2f} s")
        bounds = [max(min(g['time_parameterization']['max_velocity_rad_s'],
                          n.get('metadata', {}).get('max_velocity_rad_s', g['time_parameterization']['max_velocity_rad_s']))
                      for n in g['nodes'] if n['task_id'] == task or n['task_id'].startswith(task+'_')) for g in graphs]
        if bounds[0] == bounds[1]:
            limit_label = '任务局部速度上限' if task == 'A_exit_from_support' else '全局速度上限'
            axis.axhline(bounds[0], color='#6B7379', ls='--', lw=.8, label=f'{limit_label} {bounds[0]:g} rad/s')
            axis.set_ylim(0, bounds[0]*1.1)
        axis.set_xlabel('距该动作组实际 node_started 的时间（s）')
        axis.set_xlim(left=0); axis.set_ylim(bottom=0)
        axis.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', edgecolor='none', framealpha=.96)
    caption(fig, ['速度来自 execution.trace 中 A/B_joint_velocity_rad_s；每一时刻取6关节绝对值最大值，未做插值或平滑。',
                  '两版分别以实际开始事件归零；A_exit包含所有同名前缀子任务。采样间隔不等，图示不构成连续加速度或jerk极值证明。',
                  auditline, runline])
    files += export(fig, output, '02_measured_joint_velocity'); plt.close(fig)

    # Figure 3: four axes with actual physical units and acceptance lines.
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.8))
    fig.subplots_adjust(left=.08, right=.975, bottom=.29, top=.89, hspace=.55, wspace=.3)
    fig.suptitle('实际载荷与最终位姿：保留物理条件和验收阈值', x=.065, ha='left', y=.97, fontsize=15)
    arms = [arm for arm in ('A', 'B') if any(m['assembly_contact'][arm]['observed_samples'] for m in metrics)]
    for axis, letter, field, title, unit, limit_key in [
        (axes[0, 0], 'a', 'max_force_N', '装配期间末端传感器合力峰值', 'N', 'assembly_tool_force_limit_N'),
        (axes[0, 1], 'b', 'max_torque_Nm', '装配期间末端传感器力矩峰值', 'N·m', 'assembly_tool_torque_limit_Nm')]:
        panel(axis, letter, title, unit)
        for i, metric in enumerate(metrics):
            for j, arm in enumerate(arms):
                value = metric['assembly_contact'][arm][field]; x = j+(i-.5)*.32
                if value is not None:
                    axis.bar(x, value, width=.29, color=COLORS[i], label=LABELS[i] if j == 0 else None)
                    axis.text(x, value+.02, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
                else: axis.text(x, .02, '无观测', ha='center', fontsize=8)
        limits = [g['validation_requirements'].get(limit_key) for g in graphs]
        if limits[0] is not None and limits[0] == limits[1]:
            axis.axhline(limits[0], color='#C47732', ls='--', lw=.9, label=f'验收限值 {limits[0]:g} {unit}')
            axis.set_ylim(0, limits[0]*1.16)
        axis.set_xticks(range(len(arms)), [a+'臂' for a in arms]); axis.legend(fontsize=8.5, loc='upper right')
    parts = list(graphs[0]['validation_requirements']['goal_poses'])
    nice = {'TemporarySupport': '临时支撑', 'Bridge': '主梁', 'Gate': '封板'}
    for axis, letter, field, title, unit, factor, limit_key in [
        (axes[1, 0], 'c', 'position_error_m', '最终位置误差', 'mm', 1000., 'goal_position_tolerance_m'),
        (axes[1, 1], 'd', 'orientation_error_rad', '最终姿态误差', 'mrad', 1000., 'goal_orientation_tolerance_rad')]:
        panel(axis, letter, title, unit)
        for i, metric in enumerate(metrics):
            values = [(metric['goal_errors'].get(part) or {}).get(field) for part in parts]
            axis.bar([j+(i-.5)*.32 for j in range(len(parts))], [v*factor if v is not None else float('nan') for v in values],
                     width=.29, color=COLORS[i], label=LABELS[i])
        limits = [g['validation_requirements'].get(limit_key) for g in graphs]
        if limits[0] is not None and limits[0] == limits[1]:
            limit = limits[0]*factor
            axis.axhline(limit, color='#C47732', ls='--', lw=.9, label=f'验收限值 {limit:g} {unit}')
            axis.set_ylim(0, limit*1.13)
        axis.set_xticks(range(len(parts)), [nice.get(p, p) for p in parts]); axis.legend(fontsize=8.5, loc='upper right')
    hashes = [m['input_fingerprint']['files']['scenes/10_Scene3_Dynamic.ttt'] for m in manifests]
    hash_lines = ['共同原始场景 SHA-256：'+hashes[0]] if hashes[0] == hashes[1] else ['基线原始场景 SHA-256：'+hashes[0], '新版原始场景 SHA-256：'+hashes[1]]
    same_limits = semantic['checks']['same_global_and_task_motion_limits']
    velocity, acceleration = legacy['limits']['after']['global']
    caption(fig, ['末端合力包含工具/负载重力；仅展示有装配采样的机械臂，缺测不按0处理。阈值为本实验限制，不是UR5额定值。',
                  f'全局 vmax={velocity:g} rad/s，amax={acceleration:g} rad/s²；两版全局和任务局部v/a相同：{state(same_limits)}。',
                  *hash_lines, auditline, runline,
                  '结果来自CoppeliaSim实际动力学日志；路径和拆撑顺序可能改变，不代表实机验证、全局最优或仅平滑处理的单独因果收益。'])
    files += export(fig, output, '03_loads_errors_and_conditions'); plt.close(fig)

    evidence = {'schema_version': 1, 'before_run_id': executions[0]['run_id'], 'after_run_id': executions[1]['run_id'],
                'all_pose_boundary_statistics': all_boundaries,
                'ordinary_boundary_statistics': ordinary, 'source_scene_sha256': hashes,
                'actual_makespan_s': duration,
                'legacy_node_comparison_accepted': legacy['comparison_accepted'],
                'semantic_guard_comparison_accepted': semantic['accepted'],
                'comparison_conditions': semantic['checks'], 'configured_jerk_limits_rad_s3': semantic['configured_jerk_limits_rad_s3'],
                'velocity_panels': {task: [{k: v for k, v in item.items() if k not in
                    ['time_from_actual_start_s', 'max_abs_measured_joint_velocity_rad_s']} for item in seq] for task, seq in series.items()},
                'figures': [{'path': name, 'sha256': hashlib.sha256(Path(name).read_bytes()).hexdigest()} for name in files],
                'input_artifact_hashes': report['inputs'],
                'notes': ['Raw sampled velocities only; no target-as-measurement substitution.',
                          'Different node counts imply different stopping denominators; counts are not isolated causal effects.',
                          'Full source scene hash and run IDs printed; source evidence artifacts are not overwritten.']}
    manifest = output/'figure_evidence.json'
    if write_evidence:
        manifest.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return {'figures': files, 'evidence': str(manifest) if write_evidence else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=ROOT/'backups/smooth_motion_baseline_20261104_01/results/apex')
    parser.add_argument('--after', type=Path, default=ROOT/'results/apex')
    parser.add_argument('--output', type=Path, default=HERE/'figures')
    args = parser.parse_args()
    print(json.dumps(render_comparison(args.before, args.after, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
