"""Publication-style figures drawn from saved task, TPG and execution data."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Patch


COLORS = {"transit": "#8d969d", "pick": "#5ca9d7", "place": "#e76f70",
          "support_start": "#e3bc43", "support_end": "#e3bc43",
          "handover": "#b494d3", "wait": "#9656cf"}
LABELS = {"transit": "Transit", "pick": "Pick", "place": "Place",
          "support_start": "Support", "support_end": "Release support", "handover": "Handover"}
ARM_COLOR = {"A": "#e27b27", "B": "#397fb8"}


def _style():
    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 9, "axes.unicode_minus": False,
                         "svg.fonttype": "none", "figure.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})


def _time(row, field):
    candidates = [field, field + "_s", "actual_" + field, "actual_" + field + "_s"]
    for key in candidates:
        if key in row:
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    raise ValueError(f"Schedule row is missing a finite {field}: {row.get('id')}")


def _aggregate(rows: Sequence[Mapping[str, Any]], task_map: Mapping[str, Any],
               *, interrupted=False, cutoff=None) -> list[dict]:
    groups = defaultdict(list)
    for item in rows:
        row = dict(item)
        task_id = row.get("task_id", row.get("id"))
        if task_id:
            groups[task_id].append(row)
    answer = []
    for task_id, group in groups.items():
        definition = task_map.get(task_id, {})
        intervals, incomplete = [], False
        for row in group:
            try:
                beginning = _time(row, "start")
            except ValueError:
                if interrupted:
                    incomplete = True
                    continue
                raise
            try:
                finish = _time(row, "end")
            except ValueError:
                if not interrupted:
                    raise
                incomplete = True
                if cutoff is None or cutoff < beginning:
                    continue
                finish = cutoff
            if finish < beginning:
                if interrupted:
                    incomplete = True
                    continue
                raise ValueError(f"Schedule row ends before start: {task_id}")
            intervals.append((beginning, finish))
        if not intervals:
            continue
        start = min(beginning for beginning, _ in intervals)
        end = max(finish for _, finish in intervals)
        answer.append({"id": task_id, "task_id": task_id,
                       "arm": group[0].get("arm", definition.get("arm")),
                       "kind": group[0].get("kind", definition.get("kind", "transit")),
                       "part": group[0].get("part", definition.get("part")),
                       "start": start, "end": end, "duration": end-start,
                       "incomplete": incomplete,
                       "status": "interrupted" if incomplete else "completed"})
    return sorted(answer, key=lambda row: (row["start"], row["arm"], row["id"]))


def _execution_rows(execution, task_map):
    if execution is None:
        return []
    if isinstance(execution, Sequence) and not isinstance(execution, (str, bytes)):
        return _aggregate(execution, task_map)
    interrupted = bool(execution.get("failure")) or execution.get("success") is False
    cutoff = _execution_cutoff(execution) if interrupted else None
    for key in ("task_schedule", "tasks", "nodes", "schedule"):
        rows = execution.get(key)
        if rows:
            if isinstance(rows, Mapping):
                rows = [{"id": task_id, **row} for task_id, row in rows.items()]
            return _aggregate(rows, task_map, interrupted=interrupted, cutoff=cutoff)
    events = execution.get("events", [])
    pending, completed = {}, []
    for event in events:
        task_id = event.get("task_id", event.get("task", event.get("id")))
        kind = event.get("event", event.get("type", ""))
        stamp = event.get("time", event.get("time_s"))
        if task_id not in task_map or stamp is None:
            continue
        if kind in ("start", "task_start", "started"):
            pending[task_id] = float(stamp)
        elif kind in ("finish", "task_done", "done", "complete", "task_finish") and task_id in pending:
            completed.append({**task_map[task_id], "start": pending[task_id], "end": float(stamp), "task_id": task_id})
    if interrupted and cutoff is not None:
        finished = {row["task_id"] for row in completed}
        for task_id, beginning in pending.items():
            if task_id not in finished:
                # Missing end is intentionally retained so aggregation marks
                # this as a clipped, unfinished interval rather than success.
                completed.append({**task_map[task_id], "start": beginning, "task_id": task_id})
    return _aggregate(completed, task_map, interrupted=interrupted, cutoff=cutoff)


def _execution_cutoff(execution):
    for name in ("time", "time_s", "makespan"):
        try:
            value = float(execution.get(name))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            return value
    return None


def _lane_axes(ax, end, title, xlabel=False):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-max(end, 1.) * .025, max(end, 1.) * 1.03)
    ax.set_ylim(-.55, 1.7)
    ax.set_yticks([1., 0.], ["A 臂", "B 臂"])
    ax.tick_params(axis="y", length=0, pad=12)
    for tick, arm in zip(ax.get_yticklabels(), ("A", "B")):
        tick.set_color(ARM_COLOR[arm]); tick.set_fontweight("bold")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#e7e9eb", linewidth=.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel("时间 / s")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def _bar(ax, row, end, motion_only=False, labels=True):
    y = 1. if row["arm"] == "A" else 0.
    width = max(row["end"] - row["start"], max(end, 1.) * .0015)
    face = "#bfd9c5" if row["arm"] == "A" else "#c6dcef"
    if not motion_only:
        face = COLORS.get(row["kind"], "#b3a1bf")
    ax.add_patch(FancyBboxPatch((row["start"], y-.16), width, .32,
                  boxstyle="round,pad=0.007,rounding_size=0.025", linewidth=.5,
                  facecolor=face, edgecolor="white", zorder=3))
    if row.get("incomplete"):
        ax.add_patch(FancyBboxPatch((row["start"], y-.16), width, .32,
                     boxstyle="round,pad=0.007,rounding_size=0.025", linewidth=1.1,
                     facecolor="none", edgecolor="#a32929", hatch="////", zorder=5))
        ax.plot(row["end"], y, marker="x", color="#a32929", ms=7, mew=1.4, zorder=6)
    if labels and width > max(end, 1.) * .017:
        label = "RRT-Connect" if motion_only else LABELS.get(row["kind"], row["kind"])
        if width < max(end, 1.) * .055:
            label = "RRT" if motion_only else {"Transit": "T", "Support": "S", "Release support": "R"}.get(label, label)
        size = 7 if len(label) > 9 else 8
        ax.text(row["start"]+width/2, y, label, ha="center", va="center", fontsize=size, zorder=4)


def _dependency_arrows(ax, rows, task_map, limit=70):
    row_map = {row["id"]: row for row in rows}
    candidates = []
    for row in rows:
        for dep in task_map.get(row["id"], {}).get("dependencies", []):
            if dep in row_map and row_map[dep]["arm"] != row["arm"]:
                candidates.append((row_map[dep], row))
    for before, after in candidates[:limit]:
        y1 = 1 if before["arm"] == "A" else 0
        y2 = 1 if after["arm"] == "A" else 0
        ax.add_patch(FancyArrowPatch((before["end"], y1-.18 if y1 else y1+.18),
                     (after["start"], y2+.18 if y2 == 0 else y2-.18), arrowstyle="-|>",
                     mutation_scale=7, linestyle=(0, (4, 3)), color="#46556c", linewidth=.8, zorder=2))


def _support_spans(ax, rows):
    """Show the persistent contact state between acquire/release events."""
    pending = {}
    for row in sorted(rows, key=lambda r: (r["start"], r["end"])):
        arm = row["arm"]
        if row["kind"] == "support_start":
            pending[arm] = row
        elif row["kind"] == "support_end" and arm in pending:
            start = pending.pop(arm)
            if row["start"] > start["end"]:
                y = 1. if arm == "A" else 0.
                ax.plot([start["end"], row["start"]], [y, y],
                        color=COLORS["support_start"], lw=7, alpha=.45,
                        solid_capstyle="butt", zorder=2)


def _save(fig, output_dir, name):
    files = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=.20)
        files.append(str(path))
    plt.close(fig)
    return files


def _pipeline_figure(tasks, tpg, output_dir):
    reference = tpg["reference_schedule"]
    nominal = tpg["nominal_schedule"]
    ref_end = float(tpg["reference_makespan"])
    nominal_end = float(tpg["nominal_makespan"])
    task_map = tpg["task_map"]
    figure_width = min(30., max(16., len(reference) * .23))
    fig, axes = plt.subplots(3, 1, figsize=(figure_width, 9.7), gridspec_kw={"hspace": .58})
    fig.suptitle("APEX MR Plus：由规划数据生成的顺序任务图与多模态 TPG", x=.075, ha="left", fontsize=16, fontweight="bold")
    _lane_axes(axes[0], ref_end, "(a) ILP 分配后的顺序任务图")
    for arm, y in [("A", 1.), ("B", 0.)]:
        axes[0].plot([0, ref_end], [y, y], color=COLORS["wait"], lw=1.5, zorder=1)
    _support_spans(axes[0], reference)
    for row in reference:
        _bar(axes[0], row, ref_end)
        if row.get("part") and row["kind"] in ("pick", "place"):
            x = (row["start"]+row["end"])/2
            y = 1. if row["arm"] == "A" else 0.
            axes[0].text(x, y+.32, str(row["part"]), fontsize=7, rotation=30, ha="left", color="#556575")
    _dependency_arrows(axes[0], reference, task_map)
    axes[0].text(.99, .95, f"串行参考：{ref_end:.2f} s", transform=axes[0].transAxes, ha="right", color="#46556c")

    _lane_axes(axes[1], ref_end, "(b) RRT-Connect 参考运动路径与接触技能")
    for row in reference:
        if row["kind"] == "transit":
            _bar(axes[1], row, ref_end, motion_only=True)
        else:
            y = 1. if row["arm"] == "A" else 0.
            axes[1].plot([row["start"], row["end"]], [y, y], color=COLORS.get(row["kind"], "#888888"), lw=5, solid_capstyle="butt")
        axes[0].plot([row["start"], row["start"]], [-.42, 1.45], color="#d7dce1", linestyle=":", lw=.5, zorder=0)
    axes[1].text(.99, .95, "路径按原关节折线保留；技能单独建模", transform=axes[1].transAxes, ha="right", color="#46556c")

    _lane_axes(axes[2], nominal_end, "(c) 多模态 TPG：前驱完成后异步放行", xlabel=True)
    axes[2].set_title("(c) 多模态 TPG：前驱完成后异步放行", loc="left",
                      fontsize=12, fontweight="bold", pad=35)
    by_id = {node["id"]: node for node in tpg["nodes"]}
    times = tpg["nominal_timestamps"]
    for arm, y in [("A", 1.), ("B", 0.)]:
        axes[2].plot([0, nominal_end], [y, y], color=COLORS["wait"], lw=1.2, zorder=1)
    _support_spans(axes[2], tpg.get("nominal_task_schedule", []))
    for node in tpg["nodes"]:
        timing = times[node["id"]]
        x = (timing["start"] + timing["end"]) / 2
        y = 1. if node["arm"] == "A" else 0.
        if node["node_type"] == "pose":
            axes[2].plot(x, y, marker="o", ms=3.0, mfc="white", mec="#657280", mew=.55, zorder=4)
        else:
            width = max(timing["duration"], nominal_end * .016)
            face = COLORS.get(node["kind"], "#af9cc4")
            axes[2].add_patch(Ellipse((x, y), width, .31, facecolor=face, edgecolor="white", lw=.7, zorder=5))
            if node["kind"] in ("pick", "place", "support_start", "support_end"):
                label = LABELS.get(node["kind"], node["kind"])
                if node["kind"] in ("support_start", "support_end"):
                    # Support acquire/release can be close in nominal time.
                    # Put their short Chinese labels on opposite sides of the
                    # arm lane, instead of overlapping two long diagonals.
                    acquiring = node["kind"] == "support_start"
                    label = "建立\n支撑" if acquiring else "解除\n支撑"
                    below = bool(y) if acquiring else not bool(y)
                    offset = (-7 if acquiring else 7, -20 if below else 20)
                    axes[2].annotate(label, (x, y), xytext=offset, textcoords="offset points",
                                     ha="right" if acquiring else "left", va="top" if below else "bottom",
                                     fontsize=7, rotation=0, linespacing=1.05, color="#5c4c20")
                else:
                    axes[2].annotate(label, (x, y), xytext=(0, 17 if y else -22), textcoords="offset points",
                                     ha="center", va="bottom" if y else "top", fontsize=7, rotation=28)
    inter = [edge for edge in tpg["edges"] if edge["type"] == "type2"]
    task_edges = [e for e in inter if e["reason"] == "task_dependency"]
    motion_edges = [e for e in inter if e["reason"] == "motion_conflict"]
    # Dense full graphs remain available in JSON; the explanatory figure draws
    # a uniform subset of motion arrows and every inter-arm task dependency.
    if len(motion_edges) > 100:
        step = math.ceil(len(motion_edges)/100)
        selected = task_edges + motion_edges[::step]
    else:
        selected = inter
    for edge in selected:
        before, after = by_id[edge["source"]], by_id[edge["target"]]
        t1, t2 = times[before["id"]], times[after["id"]]
        y1, y2 = (1. if before["arm"] == "A" else 0.), (1. if after["arm"] == "A" else 0.)
        if y1 == y2:
            continue
        motion = edge["reason"] == "motion_conflict"
        axes[2].add_patch(FancyArrowPatch((t1["end"], y1-.07 if y1 else y1+.07),
                         (t2["start"], y2+.07 if y2 == 0 else y2-.07),
                         arrowstyle="-|>", mutation_scale=6, linestyle=(0, (4, 3)),
                         color="#5e9a49" if motion else "#3f506d", lw=.7, alpha=.65, zorder=2))
    verified = tpg.get("certificate", {}).get("motion_overlap_pairs_checked", False)
    text = f"名义异步：{nominal_end:.2f} s；几何冲突检查：{'已提供' if verified else '未完整提供'}"
    if len(selected) < len(inter):
        text += f"；绘出 {len(selected)}/{len(inter)} 条跨臂边"
    axes[2].text(0., 1.025, text, transform=axes[2].transAxes, ha="left", va="bottom",
                 fontsize=8, color="#46556c")
    legend = [Line2D([], [], marker="o", ms=4, mfc="white", mec="#657280", linestyle="none", label="位姿节点"),
              Patch(facecolor=COLORS["pick"], label="抓取技能"), Patch(facecolor=COLORS["place"], label="安装技能"),
              Patch(facecolor=COLORS["support_start"], label="支撑技能"),
              Line2D([], [], color=COLORS["wait"], lw=1.5, label="等待 / 同臂路径"),
              Line2D([], [], color="#3f506d", ls="--", label="任务依赖"),
              Line2D([], [], color="#5e9a49", ls="--", label="几何避碰依赖")]
    fig.legend(handles=legend, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(.52, -.015), fontsize=8)
    fig.text(.075, .012, "数据源：本次 ILP、RRT-Connect 路径及 TPG JSON。参考序列累计耗时与异步完成时间采用各自时间轴；时序预测不等于真实执行日志。", fontsize=8, color="#586574")
    return _save(fig, output_dir, "normal_pipeline_tpg")


def _gantt_figure(tpg, execution, output_dir):
    actual = _execution_rows(execution, tpg["task_map"])
    interrupted = isinstance(execution, Mapping) and (bool(execution.get("failure")) or execution.get("success") is False)
    cutoff = _execution_cutoff(execution) if interrupted else None
    views = [("顺序参考计划", tpg["reference_schedule"]),
             ("TPG 名义异步预测", tpg.get("nominal_task_schedule", []))]
    if actual:
        prediction = isinstance(execution, Mapping) and execution.get("kind") == "discrete_event_prediction"
        label = "TPG 离散事件预测（含指定延迟）" if prediction else "真实执行日志"
        if interrupted:
            label += "（执行中断；斜线标记未完成任务）"
        views.append((label, actual))
    fig, axes = plt.subplots(len(views), 1, figsize=(18, 2.65*len(views)+1.0), squeeze=False,
                             gridspec_kw={"hspace": .68})
    fig.suptitle("APEX MR Plus：双臂任务时间线", x=.075, ha="left", fontsize=16, fontweight="bold")
    end = max([row["end"] for _, rows in views for row in rows] + ([cutoff] if cutoff is not None else [1.]))
    for index, (label, rows) in enumerate(views):
        ax = axes[index, 0]
        _lane_axes(ax, end, label, xlabel=index == len(views)-1)
        _support_spans(ax, rows)
        for row in rows:
            _bar(ax, row, end)
        _dependency_arrows(ax, rows, tpg["task_map"], limit=35)
        finish = max((row["end"] for row in rows), default=0.)
        failed_view = interrupted and index == len(views)-1 and bool(actual)
        if failed_view and cutoff is not None:
            finish = cutoff
        ax.axvline(finish, color="#a32929" if failed_view else "#495767", linestyle=":", linewidth=.9)
        endpoint = "中断时刻" if failed_view else "完成时间"
        ax.text(.99, .95, f"{endpoint} {finish:.2f} s", transform=ax.transAxes, ha="right", color="#a32929" if failed_view else "#46556c")
        if failed_view and execution.get("failure"):
            ax.text(.01, .95, "中断原因：" + str(execution["failure"])[:145], transform=ax.transAxes,
                    ha="left", fontsize=8, color="#a32929")
    fig.legend(handles=[Patch(facecolor=color, label=LABELS[kind]) for kind, color in COLORS.items() if kind in LABELS],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(.52, -.018))
    fig.text(.075, .010, "任务块表示首次启动至完成的区间，内部可能含等待；黄色细带表示 Support 与 Release support 之间持续保持的支撑状态。", fontsize=8, color="#586574")
    return _save(fig, output_dir, "normal_execution_timeline")


def report_figures(tasks: Sequence[Mapping[str, Any]], tpg: Mapping[str, Any],
                   execution: Mapping[str, Any] | None, outdir: str | Path) -> list[str]:
    """Export PNG/SVG figures. ``execution`` must contain real timestamps.

    A supplied discrete-event prediction is explicitly labeled as a prediction.
    No fake execution row is synthesized when the runtime log is absent.
    """
    _style()
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _pipeline_figure(tasks, tpg, output_dir) + _gantt_figure(tpg, execution, output_dir)


__all__ = ["report_figures"]
