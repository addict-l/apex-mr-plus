"""Review figures from saved scene-3 TPG and actual executor telemetry.

Nominal prediction and actual events are never substituted for one another.
Missing signal values remain gaps. Incomplete nodes have hatched bars clipped
to the last recorded time and are explicitly labeled incomplete.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ARM_COLORS = {'A': '#dd7c2f', 'B': '#387eb3'}


def _style():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = 'DejaVu Sans'
    plt.rcParams.update({'font.family': family, 'font.size': 9,
                         'axes.unicode_minus': False, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validation_load_trace(trace):
    """Keep applied API-call evidence distinct from an unverified command."""
    if any('applied_load_N' in row or 'applied_load_substeps' in row for row in trace):
        series = []
        for row in trace:
            force, count = row.get('applied_load_N'), row.get('applied_load_substeps')
            recorded = (_finite(force) and _finite(count) and count >= 0 and count == int(count) and
                        ((force == 0 and count == 0) or (force > 0 and count >= 1)))
            series.append(force if recorded else math.nan)
        return series, '外载调用值（含动力学子步记录）', '验证外载的实际调用证据'
    return ([row['insertion_load_N'] if _finite(row.get('insertion_load_N')) else math.nan for row in trace],
            '验证外载命令（未记录施力调用）', '验证外载命令；实际调用证据缺失')


def _save(fig, outdir, stem):
    paths = []
    for suffix in ['png', 'svg']:
        path = outdir / f'{stem}.{suffix}'
        fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
        paths.append(str(path))
    plt.close(fig)
    return paths


def _lane(ax, title):
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_yticks([1, 0], ['A / 左臂', 'B / 右臂'])
    ax.set_ylim(-.55, 1.65)
    ax.set_xlabel('时间 / s')
    ax.grid(axis='x', color='#e8ecef', linewidth=.6)
    ax.set_axisbelow(True)


def _bar(ax, row, label=False):
    start, end = row['start'], row['end']
    y = 1 if row['arm'] == 'A' else 0
    interrupted = row.get('incomplete', False)
    ax.barh(y, max(end-start, .002), left=start, height=.30,
            color=ARM_COLORS[row['arm']], alpha=.82,
            edgecolor='#a12b2b' if interrupted else 'white',
            linewidth=.8, hatch='///' if interrupted else None)
    if interrupted:
        ax.plot(end, y, 'x', color='#a12b2b', markersize=6)
    if label:
        text = row.get('task_id', row.get('id', ''))
        ax.annotate(text, ((start+end)/2, y+.18), rotation=30,
                    fontsize=6.5, va='bottom', ha='left', clip_on=True)


def plot_tpg(graph, outdir):
    """Draw serial reference tasks and the real node/dependency graph."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    _style()
    nodes = graph.get('nodes', [])
    if not nodes:
        raise ValueError('Cannot draw TPG without graph nodes')
    for node in nodes:
        if not all(_finite(node.get(k)) for k in ('start', 'end')):
            raise ValueError('TPG needs nominal event timestamps: '+node['id'])
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5), constrained_layout=True)
    _lane(axes[0], '顺序参考任务（规划时间）')
    _lane(axes[1], '多模态 TPG（规划时间；不是实际执行记录）')
    for row in graph.get('reference_schedule', []):
        _bar(axes[0], row, label=row.get('kind') != 'transit')
    by_id = {node['id']: node for node in nodes}
    for node in nodes:
        _bar(axes[1], node, label=node.get('node_type') == 'skill')
    for edge in graph.get('edges', []):
        if edge['type'] != 'type2':
            continue
        before, after = by_id[edge['source']], by_id[edge['target']]
        first_y, last_y = (1 if before['arm']=='A' else 0), (1 if after['arm']=='A' else 0)
        is_motion = edge.get('reason') == 'motion_conflict'
        axes[1].annotate('', (after['start'], last_y), (before['end'], first_y),
                         arrowprops={'arrowstyle': '-|>', 'color': '#945b93' if is_motion else '#248b82',
                                     'alpha': .30 if is_motion else .65, 'lw': .55,
                                     'connectionstyle': 'arc3,rad=.08'})
    axes[1].legend(handles=[Patch(facecolor='#248b82', label='Type-2 任务依赖'),
                            Patch(facecolor='#945b93', label='Type-2 采样运动冲突'),
                            Patch(facecolor='#70818c', label='同臂连续片段由 Type-1 连接')],
                   loc='upper right', fontsize=8)
    fig.suptitle(f"SCENE 03 | {len(nodes)} 个节点 · {len(graph.get('edges', []))} 条依赖\n"
                 'RRT 路径保持原样；静止等待、世界状态变化及实际承载由独立验证负责', fontsize=12)
    return _save(fig, outdir, 'scene3_tpg')


def actual_node_rows(graph, execution):
    """Extract actual intervals exclusively from node events, preserving gaps."""
    by_id = {n['id']: n for n in graph['nodes']}
    times = {}
    all_times = [r['time'] for r in execution.get('trace', []) if _finite(r.get('time'))]
    for event in execution.get('events', []):
        if not _finite(event.get('time')):
            continue
        all_times.append(event['time'])
        node = event.get('node', event.get('node_id'))
        if node not in by_id or event.get('kind') not in ('node_started', 'node_completed'):
            continue
        key = 'start' if event['kind'] == 'node_started' else 'end'
        row = times.setdefault(node, {})
        if key in row:
            raise ValueError('Duplicate actual node event: '+node)
        row[key] = event['time']
    cutoff = max(all_times, default=None)
    result = []
    for node_id, interval in times.items():
        if 'start' not in interval:
            continue
        end = interval.get('end', cutoff)
        if end is None or end < interval['start']:
            raise ValueError('Invalid actual node interval: '+node_id)
        result.append({**by_id[node_id], **interval, 'end': end,
                       'incomplete': 'end' not in interval})
    return sorted(result, key=lambda r:(r['start'], r['id']))


def plot_execution(graph, execution, outdir, validation=None):
    """Draw observed node intervals and unmodified actual telemetry samples."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    _style()
    rows = actual_node_rows(graph, execution)
    if not rows:
        raise ValueError('No actual node events; refusing to draw a fabricated execution timeline')
    verified = validation is not None and validation.get('success') is True
    status = '独立验收通过' if verified else '实际日志诊断 · 尚未通过独立验收'
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5), constrained_layout=True)
    _lane(axes[0], 'TPG 名义预测（规划时间）')
    _lane(axes[1], '实际节点服务区间（仿真时间；红色斜线表示未完成）')
    for node in graph['nodes']:
        _bar(axes[0], node)
    for node in rows:
        _bar(axes[1], node, label=node.get('node_type') == 'skill')
    missing = len(graph['nodes'])-len(rows)
    fig.suptitle(f'SCENE 03 | {status}\n实际启动 {len(rows)}/{len(graph["nodes"])} 个节点；未启动 {missing} 个节点不绘制为完成', fontsize=12)
    paths = _save(fig, outdir, 'scene3_execution_timeline' if verified else 'scene3_execution_diagnostic')
    trace = execution.get('trace', [])
    if len(trace) < 2:
        return paths
    if any(not _finite(r.get('time')) for r in trace):
        raise ValueError('Trace contains invalid timestamps')
    times = [r['time'] for r in trace]
    if any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('Trace times must strictly increase; duplicates are not silently discarded')

    def values(key, valid_key=None, scale=1.):
        return [scale*r[key] if _finite(r.get(key)) and (valid_key is None or r.get(valid_key) is True)
                else math.nan for r in trace]

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    channels = [
        (axes[0], 'support_force_N', '临时支撑反力', '#c59a00', 'support_valid', 1.),
        (axes[0], 'A_support_force_N', 'A 臂支撑反力', ARM_COLORS['A'], 'A_support_valid', 1.),
        (axes[0], 'permanent_force_N', '永久连接载荷', '#2a9173', 'permanent_sensor_valid', 1.),
        (axes[2], 'bridge_position_error_m', '桥面位置误差', '#387eb3', None, 1000.),
    ]
    for ax, key, label, color, valid_key, scale in channels:
        series = values(key, valid_key, scale)
        if any(math.isfinite(v) for v in series):
            ax.plot(times, series, color=color, linewidth=1., label=label)
        else:
            ax.plot([], [], color=color, label=label+'（缺失）')
    load_series, load_label, load_title = _validation_load_trace(trace)
    if any(math.isfinite(value) for value in load_series):
        axes[1].plot(times, load_series, color='#965fa1', linewidth=1., label=load_label)
    else:
        axes[1].plot([], [], color='#965fa1', label=load_label+'（缺失）')
    titles = ['实测支撑与永久连接载荷', load_title, '实测桥面稳定性']
    for ax, title, unit in zip(axes, titles, ['N', 'N', 'mm']):
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylabel(unit); ax.grid(color='#e8ecef', linewidth=.6)
        ax.legend(loc='upper right', fontsize=8)
    angle_values = values('bridge_orientation_error_rad', scale=180/math.pi)
    if any(math.isfinite(v) for v in angle_values):
        angle_axis = axes[2].twinx()
        angle_axis.plot(times, angle_values, color='#a4633d', linewidth=.8, label='桥面姿态误差')
        angle_axis.set_ylabel('姿态误差 / °', color='#a4633d')
    event_labels = {'support_load_verified': '载荷接替', 'arm_support_released': 'A 释放',
                    'permanent_structure_verified': '永久承载确认', 'support_returned': '支撑归仓'}
    for event in execution.get('events', []):
        if event.get('kind') not in event_labels or not _finite(event.get('time')):
            continue
        for ax in axes:
            ax.axvline(event['time'], color='#7c8b95', lw=.7, alpha=.6, linestyle='--')
        axes[0].annotate(event_labels[event['kind']], (event['time'], 1.),
                         xycoords=('data', 'axes fraction'), xytext=(3,-5),
                         textcoords='offset points', rotation=90, va='top', fontsize=7)
    axes[-1].set_xlabel('实际仿真时间 / s')
    fig.suptitle(f'SCENE 03 | {status}\n直接绘制日志采样；缺失/无效信号保留断点，不插值或补零', fontsize=12)
    paths += _save(fig, outdir, 'scene3_actual_dynamics' if verified else 'scene3_dynamics_diagnostic')
    return paths


def render_figures(graph, outdir, execution=None, validation=None):
    paths = plot_tpg(graph, outdir)
    if execution is not None:
        paths += plot_execution(graph, execution, outdir, validation)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]/'results'/'apex'
    parser.add_argument('--graph', type=Path, default=base/'tpg.json')
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--outdir', type=Path, default=base)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding='utf-8')) if path else None
    print(json.dumps(render_figures(read(args.graph), args.outdir,
                                   read(args.execution), read(args.validation)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
