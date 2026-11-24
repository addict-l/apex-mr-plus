"""Render an accepted LEGO bridge run from the actual recorded plan and timing."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from PIL import Image

from bridge_media import sha256


ARM_COLORS = {"A": "#246383", "B": "#bd622f"}


def _arms(row):
    names = row.get("arms", row.get("arm"))
    if names is None:
        raise ValueError(f"Timing row {row.get('id')} has no assigned arm")
    arms = (names,) if isinstance(names, str) else tuple(names)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= ARM_COLORS.keys():
        raise ValueError(f"Invalid assigned arm(s): {arms}")
    return arms


def _span(row):
    start, end = float(row["start_s"]), float(row["end_s"])
    if not all(map(math.isfinite, (start, end))) or not 0 <= start <= end:
        raise ValueError(f"Invalid timing extent for {row.get('id')}: {start}, {end}")
    return start, end


def _title(row):
    for key in ("text", "action", "source_action", "label"):
        if row.get(key):
            return str(row[key])
    return str(row["id"])


def _action_label(row, cap=22):
    title = _title(row)
    title = re.sub(r"\(([AB]),\s*", "(", title)
    title = re.sub(r",\s*([^,()]+)\)$", ")\n" + r"\1", title)
    return title if len(title) <= cap else f"{title[:cap - 1]}…"


def _bars(axis, rows, lanes, *, labels=False, alpha=1.):
    longest = max((_span(row)[1] for row in rows), default=0.)
    lookup = {}
    for row in rows:
        identifier = row["id"]
        if identifier in lookup:
            raise ValueError(f"Duplicate timing row {identifier}")
        start, end = _span(row)
        for arm in _arms(row):
            lane = lanes[arm]
            if end > start:
                axis.barh(lane, end - start, left=start, height=.58, color=ARM_COLORS[arm],
                          alpha=alpha, linewidth=.35, edgecolor="white", zorder=3)
            else:
                axis.scatter([start], [lane], marker="|", color=ARM_COLORS[arm], zorder=4)
            if labels and longest and end - start >= longest * .035:
                axis.text((start + end) / 2, lane, _action_label(row), ha="center", va="center",
                          fontsize=6.5, color="white", linespacing=.9, zorder=4, clip_on=True)
            lookup[identifier] = start, end, lane
    return lookup, longest


def _axis(axis, title, lanes, longest):
    axis.set_title(title, loc="left", fontweight="semibold", color="#172a36", fontsize=11)
    axis.set_yticks(list(lanes.values()), list(lanes.keys()))
    axis.set_ylim(min(lanes.values()) - .6, max(lanes.values()) + .6)
    axis.set_xlim(0, longest * 1.03 if longest > 0 else 1)
    axis.grid(axis="x", color="#d8e3e9", linewidth=.65, zorder=0)
    axis.tick_params(left=False, labelsize=8)
    axis.spines[["right", "left", "top"]].set_visible(False)
    axis.spines["bottom"].set_color("#a0b0ba")
    axis.set_xlabel("Time (s)", loc="right", fontsize=9, labelpad=3)


def _edge(geometry, src, dst):
    left, right = geometry[src], geometry[dst]
    return (left[1], left[2]), (right[0], right[2])


def _draw_edges(axis, geometry, edges):
    for edge in edges:
        start, end = _edge(geometry, edge["source"], edge["target"])
        cross = start[1] != end[1]
        if not cross:
            continue
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6,
                                      color="#446c6a", linestyle="--", linewidth=.6,
                                      alpha=.55, zorder=2, shrinkA=3, shrinkB=3))


def write_bridge_figure(mode_dir: Path, *, plan: list[dict], reference: list[dict],
                        tpg: list[dict], replay: list[dict], edges: list[dict],
                        acceptance: Path, evidence_paths: dict[str, Path]) -> dict:
    """Use their measured seconds, providing a PNG only for a passed replay receipt.

    Row keys: id, arm or arms, start_s, end_s, optional text/action. Edge keys:
    source, target. The caller maps its durable JSON schema and supplies the
    signed plan/reference/TPG/replay inputs as evidence_paths.
    """
    mode_dir = Path(mode_dir).resolve()
    acceptance = Path(acceptance).resolve()
    report = json.loads(acceptance.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("Figure export needs a passed replay acceptance receipt")
    if not all((plan, reference, tpg, replay)) or not set(evidence_paths) >= {
        "plan", "reference", "tpg", "replay"
    }:
        raise ValueError("Missing symbolic plan, serial reference, TPG, or recorded replay evidence")
    if mode_dir.name not in {"mode_a", "mode_b"}:
        raise ValueError(f"Unexpected mode directory {mode_dir}")
    paths = {"validation": acceptance, **{key: Path(path).resolve() for key, path in evidence_paths.items()}}
    if not all(path.is_relative_to(mode_dir) for path in paths.values()):
        raise ValueError("Figure evidence must belong to this mode")
    hashes = {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}

    folder = mode_dir / "figures"
    base = f"{mode_dir.name}_apex_mr_diagram"
    output, partial, receipt = (folder / f"{base}.{suffix}" for suffix in ("png", "partial.png", "json"))
    for path in (output, partial, receipt):
        if path.exists():
            raise FileExistsError(path)
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(14, 7.7), gridspec_kw={"height_ratios": [1., 1., 1.4]},
                               layout="constrained")
    lanes = {"A": 1., "B": 0.}
    timing_lanes = {"TPG A": 3., "TPG B": 2., "Replay A": .65, "Replay B": -.35}
    _, plan_span = _bars(axes[0], plan, lanes, labels=True)
    _axis(axes[0], f"(a) Symbolic plan · {len(plan)} actions", lanes, plan_span)
    _, reference_span = _bars(axes[1], reference, lanes)
    _axis(axes[1], f"(b) Verified serial reference · {len(reference)} segments", lanes, reference_span)
    tpg_geometry, scheduled_span = _bars(axes[2], tpg, {"A": 3., "B": 2.}, alpha=.72)
    _, replay_span = _bars(axes[2], replay, {"A": .65, "B": -.35})
    _draw_edges(axes[2], tpg_geometry, edges)
    _axis(axes[2], f"(c) TPG and recorded replay · {len(tpg)} nodes / {len(edges)} edges",
          timing_lanes, max(scheduled_span, replay_span))
    axes[2].legend(handles=[Patch(facecolor=color, label=f"Arm {arm}") for arm, color in ARM_COLORS.items()],
                   loc="upper right", fontsize=8, frameon=False)
    figure.suptitle(f"LEGO bridge assembly  ·  {mode_dir.name.replace('_', ' ').title()}",
                    color="#172a36", fontsize=16, fontweight="semibold")
    try:
        figure.savefig(partial, format="png", dpi=300, facecolor="white")
    finally:
        plt.close(figure)
    with Image.open(partial) as image:
        dimensions = image.size
        dpi = image.info["dpi"]
        if not all(abs(value - 300) < .1 for value in dpi):
            raise RuntimeError(f"Figure DPI verification failed: {dpi}")
    partial.rename(output)
    result = {
        "passed": True, "path": str(output), "sha256": sha256(output), "width": dimensions[0],
        "height": dimensions[1], "dpi": list(dpi), "inputs": hashes, "symbolic_actions": len(plan),
        "serial_reference_segments": len(reference), "tpg_nodes": len(tpg), "tpg_edges": len(edges),
        "replay_timing_nodes": len(replay), "symbolic_timing_s": plan_span,
        "reference_timing_s": reference_span, "scheduled_timing_s": scheduled_span, "recorded_replay_timing_s": replay_span,
    }
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.rename(receipt)
    return result

