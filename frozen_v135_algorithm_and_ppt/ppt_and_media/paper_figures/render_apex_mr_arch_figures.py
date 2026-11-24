"""Generate publication-grade APEX-MR architecture figures (replicating the original paper's Fig. 2 style)
for all three verified dual-arm LEGO bridge scenarios:
  1. Mode A (Free-Reach Parallel Assembly, 18 steps, 1.161x speedup)
  2. Mode B (Strict-Zone + 4x Mid-Air Handover, 22 steps, 1.250x speedup)
  3. Online Recovery (Mode A Prefix -> Fault Barrier -> db77489 Graph Repair -> 4x Handover Suffix)

All embedded photographs are extracted directly from the verified CoppeliaSim MP4 videos.
"""

from __future__ import annotations

import math
import pathlib
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Polygon, Rectangle
import numpy as np
from PIL import Image, ImageOps


ROOT = pathlib.Path(r"d:\APEX_MR_Plus\dualarm_skillnet\results\sim_v1_3\motion_v135\showcases\lego_bridge_9blocks")
OUT_DIR = ROOT / "paper_figures"
FRAME_DIR = OUT_DIR / "frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
})

# Harmonized academic palette matching APEX-MR Figure 2
C_BG_BORDER = "#B0BAC5"
C_PANEL_DASH = "#64748B"
C_ARROW_BIG = "#B8C0C8"

C_ARM_A_BG = "#D8F0D8"
C_ARM_A_EDGE = "#78B878"
C_ARM_B_BG = "#D6E8FA"
C_ARM_B_EDGE = "#6FA4DC"

C_TRANSIT = "#848A92"
C_TRANSIT_EDGE = "#636870"
C_PICK = "#5AA9E6"
C_PICK_EDGE = "#2B7AB8"
C_PLACE = "#E56B63"
C_PLACE_EDGE = "#BC433C"
C_HANDOVER = "#F3C64F"
C_HANDOVER_EDGE = "#C9991F"
C_WAIT = "#8E3CB8"
C_TASK_DEP = "#3B556E"
C_MOTION_DEP = "#4EA860"
C_FAULT = "#C0392B"
C_FAULT_BG = "#FDEDEC"
C_TEXT = "#1E293B"
C_MUTED = "#475569"


def grab_frame(video_path: pathlib.Path, time_s: float, out_path: pathlib.Path) -> pathlib.Path:
    cmd = [
        "ffmpeg", "-y", "-ss", f"{time_s:.3f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2", str(out_path)
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return out_path


def crop_pil(img_path: pathlib.Path, box_frac: tuple[float, float, float, float], target_size: tuple[int, int]) -> np.ndarray:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = box_frac
    crop = im.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
    crop = ImageOps.fit(crop, target_size, method=Image.Resampling.LANCZOS)
    return np.asarray(crop)


def draw_robot_badge(ax, x, y, w, h, side: str):
    bg = C_ARM_A_BG if side == "A" else C_ARM_B_BG
    ec = C_ARM_A_EDGE if side == "A" else C_ARM_B_EDGE
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        facecolor=bg, edgecolor=ec, linewidth=1.1, zorder=5
    ))
    cx, cy = x + w * 0.48, y + h * 0.22
    ax.add_patch(Rectangle((cx - w * 0.26, cy - h * 0.08), w * 0.52, h * 0.14,
                           facecolor="#475569", edgecolor="#1E293B", lw=0.7, zorder=6))
    p0 = (cx, cy + h * 0.06)
    p1 = (cx - w * 0.14, cy + h * 0.42)
    p2 = (cx + w * 0.16, cy + h * 0.62)
    p3 = (cx + w * 0.26, cy + h * 0.45)
    for pa, pb in ((p0, p1), (p1, p2), (p2, p3)):
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#334155", lw=2.3, solid_capstyle="round", zorder=6)
    for pt in (p0, p1, p2):
        ax.add_patch(Circle(pt, min(w, h) * 0.085, facecolor="#94A3B8", edgecolor="#1E293B", lw=0.7, zorder=7))
    ax.plot([p3[0] - w * 0.04, p3[0] - w * 0.04], [p3[1], p3[1] - h * 0.12], color="#1E293B", lw=1.1, zorder=7)
    ax.plot([p3[0] + w * 0.04, p3[0] + w * 0.04], [p3[1], p3[1] - h * 0.12], color="#1E293B", lw=1.1, zorder=7)
    ax.text(x + w * 0.12, y + h * 0.85, side, fontsize=7.2, fontweight="bold",
            color="#1E293B", ha="left", va="center", zorder=8)


def draw_skills_cylinder(ax, x, y, w, h, has_handover: bool):
    rx, ry = w * 0.46, h * 0.14
    cx = x + w * 0.50
    top_y = y + h * 0.80
    bot_y = y + h * 0.22
    ax.add_patch(Rectangle((cx - rx, bot_y), 2 * rx, top_y - bot_y, facecolor="#F8FAFC", edgecolor="none", zorder=4))
    ax.add_patch(Ellipse((cx, bot_y), 2 * rx, 2 * ry, facecolor="#F8FAFC", edgecolor="#475569", lw=1.15, zorder=4))
    ax.plot([cx - rx, cx - rx], [bot_y, top_y], color="#475569", lw=1.15, zorder=5)
    ax.plot([cx + rx, cx + rx], [bot_y, top_y], color="#475569", lw=1.15, zorder=5)
    ax.add_patch(Ellipse((cx, top_y), 2 * rx, 2 * ry, facecolor="#FFFFFF", edgecolor="#475569", lw=1.15, zorder=6))
    if has_handover:
        rows = [[(C_PICK, "Pick"), (C_PLACE, "Place")], [(C_TRANSIT, "Park"), (C_HANDOVER, "Handover")]]
    else:
        rows = [[(C_PICK, "Pick"), (C_PLACE, "Place")], [(C_TRANSIT, "Transit"), ("#7FB3D5", "Park")]]
    for r_i, row in enumerate(rows):
        py = bot_y + (top_y - bot_y) * (0.58 - r_i * 0.34)
        for c_i, (col, lbl) in enumerate(row):
            px = cx - rx * 0.78 + c_i * (rx * 0.82)
            pw = rx * 0.74
            ph = (top_y - bot_y) * 0.22
            ax.add_patch(FancyBboxPatch(
                (px, py - ph * 0.5), pw, ph,
                boxstyle="round,pad=0.001,rounding_size=0.005",
                facecolor=col, edgecolor="none", zorder=6
            ))
            ax.text(px + pw * 0.5, py, lbl, fontsize=5.5, fontweight="bold",
                    color="#FFFFFF" if col != C_HANDOVER else "#1E293B",
                    ha="center", va="center", zorder=7)


def draw_big_block_arrow(ax, x0, y0, x1, y1, shaft_w=0.014, head_w=0.032, head_l=0.016):
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    sb = max(0.0, length - head_l)
    pts = [
        (x0 + nx * shaft_w * 0.5, y0 + ny * shaft_w * 0.5),
        (x0 + ux * sb + nx * shaft_w * 0.5, y0 + uy * sb + ny * shaft_w * 0.5),
        (x0 + ux * sb + nx * head_w * 0.5, y0 + uy * sb + ny * head_w * 0.5),
        (x1, y1),
        (x0 + ux * sb - nx * head_w * 0.5, y0 + uy * sb - ny * head_w * 0.5),
        (x0 + ux * sb - nx * shaft_w * 0.5, y0 + uy * sb - ny * shaft_w * 0.5),
        (x0 - nx * shaft_w * 0.5, y0 - ny * shaft_w * 0.5),
    ]
    ax.add_patch(Polygon(pts, closed=True, facecolor=C_ARROW_BIG, edgecolor="#9AA4AE", lw=0.6, zorder=4))


def place_image_box(ax, img_arr: np.ndarray, x: float, y: float, w: float, h: float, border_color="#94A3B8", lw=0.9):
    ax.imshow(img_arr, extent=(x, x + w, y, y + h), aspect="auto", zorder=5, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=border_color, lw=lw, zorder=6))


def draw_rect_skill(ax, x, y, w, h, label, facecolor, edgecolor, text_color="#FFFFFF", fontsize=7.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.001,rounding_size=0.009",
        facecolor=facecolor, edgecolor=edgecolor, lw=1.0, zorder=6
    ))
    ax.text(x + w * 0.5, y + h * 0.5, label, fontsize=fontsize, fontweight="bold",
            color=text_color, ha="center", va="center", zorder=7)
    return (x + w * 0.5, y + h * 0.5)


def draw_oval_skill(ax, cx, cy, w, h, label, facecolor, edgecolor, text_color="#FFFFFF", fontsize=7.0):
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor=facecolor, edgecolor=edgecolor, lw=1.15, zorder=7))
    ax.text(cx, cy, label, fontsize=fontsize, fontweight="bold",
            color=text_color, ha="center", va="center", zorder=8)


def draw_object_node(ax, cx, cy, w, h, label, edge_color, anchor_xy, text_color=None):
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor="#FFFFFF", edgecolor=edge_color,
                         lw=1.15, linestyle=(0, (4, 2.5)), zorder=7))
    ax.text(cx, cy, label, fontsize=6.9, fontweight="medium",
            color=text_color or edge_color, ha="center", va="center", zorder=8)
    ax.plot([cx, anchor_xy[0]], [cy - h * 0.5 if cy > anchor_xy[1] else cy + h * 0.5, anchor_xy[1]],
            color=edge_color, lw=0.95, zorder=5)


def draw_pose_chain(ax, x0, x1, y, n_dots=5, arrow_end=True):
    if x1 <= x0 + 0.004:
        return
    ax.plot([x0, x1 - (0.006 if arrow_end else 0.0)], [y, y], color="#334155", lw=1.25, zorder=5)
    if arrow_end:
        ax.annotate("", xy=(x1, y), xytext=(x1 - 0.008, y),
                    arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.1, mutation_scale=8), zorder=6)
    if n_dots > 0:
        margin = min(0.008, (x1 - x0) * 0.18)
        end_x = x1 - margin - (0.005 if arrow_end else 0.0)
        if end_x > x0 + margin:
            xs = np.linspace(x0 + margin, end_x, n_dots)
            for dx in xs:
                ax.add_patch(Ellipse((dx, y), 0.0070, 0.0170, facecolor="#FFFFFF", edgecolor="#334155", lw=0.95, zorder=6))


def draw_wait_line(ax, x0, x1, y, label=None, label_offset=0.022):
    if x1 <= x0 + 0.004:
        return
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=C_WAIT, lw=1.45, mutation_scale=8), zorder=5)
    if label:
        ax.text((x0 + x1) * 0.5, y + label_offset, label, fontsize=7.3, fontstyle="italic",
                color=C_TEXT, ha="center", va="center", zorder=7)


def draw_curved_dep(ax, p0, p1, color, rad=0.22, lw=1.25, dashed=True):
    arr = FancyArrowPatch(
        p0, p1,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=9.2,
        color=color,
        lw=lw,
        linestyle=(0, (4.5, 2.5)) if dashed else "solid",
        zorder=9,
    )
    ax.add_patch(arr)


def build_figure(spec: dict):
    fig = plt.figure(figsize=(16.2, 6.4), dpi=300, facecolor="#FFFFFF")
    ax = fig.add_axes([0.008, 0.015, 0.984, 0.970])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (0.004, 0.012), 0.992, 0.976,
        boxstyle="round,pad=0.002,rounding_size=0.014",
        facecolor="#FFFFFF", edgecolor=C_BG_BORDER, lw=1.2, zorder=1
    ))

    # =========================================================================
    # LEFT COLUMN
    # =========================================================================
    LX, LW = 0.014, 0.264

    ax.add_patch(FancyBboxPatch(
        (LX, 0.655), LW, 0.318,
        boxstyle="round,pad=0.001,rounding_size=0.008",
        facecolor="#FFFFFF", edgecolor=C_PANEL_DASH, lw=1.05, linestyle=(0, (4, 2.5)), zorder=2
    ))
    ax.text(LX + LW * 0.5, 0.946, "Input", fontsize=11.5, fontweight="bold", color=C_TEXT, ha="center", va="center", zorder=6)

    iw, ih, iy = 0.076, 0.175, 0.742
    ixs = [LX + 0.010, LX + 0.094, LX + 0.178]
    place_image_box(ax, spec["img_env"], ixs[0], iy, iw, ih, border_color="#94A3B8", lw=0.85)
    ax.text(ixs[0] + iw * 0.5, 0.698, "Environment\nSetup", fontsize=7.6, color=C_TEXT, ha="center", va="center", linespacing=1.15)
    place_image_box(ax, spec["img_design"], ixs[1], iy, iw, ih, border_color="#94A3B8", lw=0.85)
    ax.text(ixs[1] + iw * 0.5, 0.698, "Assembly\nDesign", fontsize=7.6, color=C_TEXT, ha="center", va="center", linespacing=1.15)
    draw_skills_cylinder(ax, ixs[2], iy, iw, ih, has_handover=spec["has_handover"])
    ax.text(ixs[2] + iw * 0.5, 0.698, "Manipulation\nSkills", fontsize=7.6, color=C_TEXT, ha="center", va="center", linespacing=1.15)

    draw_big_block_arrow(ax, LX + LW * 0.5, 0.648, LX + LW * 0.5, 0.600, shaft_w=0.014, head_w=0.030, head_l=0.018)

    ax.add_patch(FancyBboxPatch(
        (LX, 0.378), LW, 0.214,
        boxstyle="round,pad=0.001,rounding_size=0.008",
        facecolor="#FFFFFF", edgecolor="#475569", lw=1.0, zorder=2
    ))
    ax.text(LX + LW * 0.5, 0.568, spec["assembly_plan_title"], fontsize=9.0, fontweight="bold", color=C_TEXT, ha="center", va="center", zorder=6)

    tw, th, ty = 0.044, 0.118, 0.425
    txs = [LX + 0.008, LX + 0.056, LX + 0.104, LX + 0.166, LX + 0.214]
    for idx, (tx, im_arr, lbl) in enumerate(zip(txs, spec["img_steps"], spec["plan_step_labels"])):
        place_image_box(ax, im_arr, tx, ty, tw, th, border_color="#CBD5E1", lw=0.75)
        ax.text(tx + tw * 0.5, 0.400, lbl, fontsize=7.4, color=C_TEXT, ha="center", va="center")
        if idx in (0, 1, 3):
            ax.annotate("", xy=(txs[idx + 1] - 0.001, ty + th * 0.5),
                        xytext=(tx + tw + 0.001, ty + th * 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=0.85, mutation_scale=6), zorder=7)
    ax.text(LX + 0.156, ty + th * 0.5, "...", fontsize=10, fontweight="bold", color="#475569", ha="center", va="center")

    ax.add_patch(FancyBboxPatch(
        (LX, 0.028), LW, 0.332,
        boxstyle="round,pad=0.001,rounding_size=0.008",
        facecolor="#FFFFFF", edgecolor=C_PANEL_DASH, lw=1.05, linestyle=(0, (4, 2.5)), zorder=2
    ))
    ax.text(LX + LW * 0.5, 0.334, spec["bottom_title"], fontsize=10.5, fontweight="bold", color=C_TEXT, ha="center", va="center", zorder=6)

    cw, ch, cy = 0.080, 0.240, 0.062
    cxs = [LX + 0.009, LX + 0.092, LX + 0.175]
    for cx_i, im_arr, cap in zip(cxs, spec["img_collab"], spec["collab_captions"]):
        place_image_box(ax, im_arr, cx_i, cy, cw, ch, border_color="#64748B", lw=0.9)
        ax.text(cx_i + cw * 0.5, 0.044, cap, fontsize=6.4, fontweight="medium", color=C_MUTED, ha="center", va="center")

    draw_big_block_arrow(ax, LX + LW + 0.007, 0.665, 0.312, 0.665, shaft_w=0.028, head_w=0.062, head_l=0.015)
    draw_big_block_arrow(ax, 0.312, 0.205, LX + LW + 0.007, 0.205, shaft_w=0.028, head_w=0.062, head_l=0.015)

    # =========================================================================
    # RIGHT COLUMN: APEX-MR Architecture Box
    # =========================================================================
    RX, RW = 0.318, 0.670
    ax.add_patch(FancyBboxPatch(
        (RX, 0.028), RW, 0.945,
        boxstyle="round,pad=0.001,rounding_size=0.010",
        facecolor="#FFFFFF", edgecolor=C_PANEL_DASH, lw=1.15, linestyle=(0, (5, 3)), zorder=2
    ))

    ax.text(RX + RW - 0.012, 0.948, spec["framework_title"], fontsize=11.0, fontweight="bold", color=C_TEXT, ha="right", va="center", zorder=6)
    ax.text(RX + 0.012, 0.948, spec["panel_a_title"], fontsize=9.8, fontweight="bold", color=C_TEXT, ha="left", va="center", zorder=6)

    y_div_ab, y_div_bc = 0.558, 0.418
    ax.plot([RX + 0.004, RX + RW - 0.004], [y_div_ab, y_div_ab], color="#475569", lw=0.95, linestyle=(0, (3.5, 2.5)), zorder=3)
    ax.plot([RX + 0.004, RX + RW - 0.004], [y_div_bc, y_div_bc], color="#475569", lw=0.95, linestyle=(0, (3.5, 2.5)), zorder=3)

    # -------------------------------------------------------------------------
    # (a) Sequential Task Graph
    # -------------------------------------------------------------------------
    ya_top, ya_bot = 0.805, 0.660
    bw, bh, bx = 0.036, 0.088, RX + 0.022
    draw_robot_badge(ax, bx, ya_top - bh * 0.5, bw, bh, "A")
    draw_robot_badge(ax, bx, ya_bot - bh * 0.5, bw, bh, "B")

    sh = 0.056
    motion_drops = []
    a_anchors = {}

    for item in spec["panel_a_items"]:
        kind = item["kind"]
        row_y = ya_top if item["arm"] == "A" else ya_bot
        x0, x1 = item["x0"], item["x1"]
        if kind == "wait":
            draw_wait_line(ax, x0, x1, row_y, label=item.get("label"), label_offset=item.get("loff", 0.024))
        elif kind == "ellipsis":
            ax.add_patch(FancyBboxPatch((x0, row_y - sh * 0.45), x1 - x0, sh * 0.90,
                                        boxstyle="round,pad=0.001,rounding_size=0.007",
                                        facecolor="#F8FAFC", edgecolor="#64748B", lw=0.9, zorder=6))
            ax.text((x0 + x1) * 0.5, row_y, "...", fontsize=10, fontweight="bold", color="#475569", ha="center", va="center", zorder=7)
        else:
            pal = {
                "transit": (C_TRANSIT, C_TRANSIT_EDGE, "#FFFFFF"),
                "pick": (C_PICK, C_PICK_EDGE, "#FFFFFF"),
                "place": (C_PLACE, C_PLACE_EDGE, "#FFFFFF"),
                "handover": (C_HANDOVER, C_HANDOVER_EDGE, "#1E293B"),
            }[kind]
            cx, cy = draw_rect_skill(ax, x0, row_y - sh * 0.5, x1 - x0, sh, item["label"], pal[0], pal[1], pal[2],
                                     fontsize=item.get("fs", 7.1))
            if "id" in item:
                a_anchors[item["id"]] = (cx, cy, x0, x1, row_y)
            if item.get("drop_planner"):
                motion_drops.append((cx, row_y - sh * 0.5, item["arm"], item["drop_planner"], item.get("drop_box", True)))
            if item.get("obj"):
                obj_lbl, obj_pos = item["obj"]
                oy = row_y + (0.072 if obj_pos == "above" else -0.070)
                oc = C_PICK_EDGE if kind == "pick" else C_PLACE_EDGE if kind == "place" else C_HANDOVER_EDGE
                anchor = (cx, row_y + sh * 0.5 if obj_pos == "above" else row_y - sh * 0.5)
                draw_object_node(ax, cx + item.get("obj_dx", 0.0), oy, item.get("obj_w", 0.072), 0.046,
                                 obj_lbl, oc, anchor)
        if item.get("next_arrow_to") is not None:
            ax.annotate("", xy=(item["next_arrow_to"], row_y), xytext=(x1, row_y),
                        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.0, mutation_scale=7), zorder=5)

    for dep in spec["panel_a_deps"]:
        src = a_anchors[dep["src"]]
        dst = a_anchors[dep["dst"]]
        p0 = (src[0] + dep.get("dx0", 0.0), src[4] + dep.get("dy0", -sh * 0.5))
        p1 = (dst[0] + dep.get("dx1", 0.0), dst[4] + dep.get("dy1", sh * 0.5))
        draw_curved_dep(ax, p0, p1, C_TASK_DEP, rad=dep.get("rad", -0.18), lw=1.2)

    if spec.get("fault_x_a") is not None:
        fx = spec["fault_x_a"]
        bx_c = spec.get("fault_badge_x_a", fx)
        bw_a = 0.096
        ax.plot([fx, fx], [0.568, 0.878], color=C_FAULT, lw=1.55, linestyle=(0, (4, 2)), zorder=10)
        ax.add_patch(FancyBboxPatch((bx_c - bw_a * 0.5, 0.878), bw_a, 0.044,
                                    boxstyle="round,pad=0.001,rounding_size=0.008",
                                    facecolor=C_FAULT_BG, edgecolor=C_FAULT, lw=1.1, zorder=11))
        ax.text(bx_c, 0.900, spec["fault_label_a"], fontsize=6.1, fontweight="bold",
                color=C_FAULT, ha="center", va="center", zorder=12)

    # -------------------------------------------------------------------------
    # (b) Motion Plan (zero overlapping blocks!)
    # -------------------------------------------------------------------------
    ax.text(RX + 0.012, 0.534, spec["panel_b_title"], fontsize=9.8, fontweight="bold", color=C_TEXT, ha="left", va="center", zorder=6)
    ax.text(RX + RW - 0.012, 0.534, spec["panel_b_note"], fontsize=7.4, color=C_MUTED, ha="right", va="center", zorder=6)

    yb = 0.468
    mbw = spec.get("mbw", 0.044)
    mbh = 0.052
    motion_drops.sort(key=lambda r: r[0])
    # Draw vertical drop lines for all items, and render boxes only where drop_box=True
    box_xs = []
    for mx, y_from, arm, plbl, draw_box in motion_drops:
        ax.annotate("", xy=(mx, yb + mbh * 0.5 + 0.003), xytext=(mx, y_from - 0.004),
                    arrowprops=dict(arrowstyle="-|>", color="#B0BAC5", lw=0.95,
                                    linestyle=(0, (3, 2.5)), mutation_scale=7.5), zorder=3)
        if draw_box:
            bg = C_ARM_A_BG if arm == "A" else C_ARM_B_BG
            ec = C_ARM_A_EDGE if arm == "A" else C_ARM_B_EDGE
            ax.add_patch(FancyBboxPatch((mx - mbw * 0.5, yb - mbh * 0.5), mbw, mbh,
                                        boxstyle="round,pad=0.001,rounding_size=0.007",
                                        facecolor=bg, edgecolor=ec, lw=1.0, zorder=6))
            ax.text(mx, yb, plbl, fontsize=6.9, fontweight="bold", color=C_TEXT, ha="center", va="center", zorder=7)
            box_xs.append(mx)
    for idx in range(len(box_xs) - 1):
        mx, nx = box_xs[idx], box_xs[idx + 1]
        if nx - mx > mbw + 0.004:
            ax.annotate("", xy=(nx - mbw * 0.5 - 0.002, yb), xytext=(mx + mbw * 0.5 + 0.002, yb),
                        arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=0.95, mutation_scale=7), zorder=5)

    # -------------------------------------------------------------------------
    # (c) Parallel, Asynchronous Plan after TPG
    # -------------------------------------------------------------------------
    ax.text(RX + 0.012, 0.392, spec["panel_c_title"], fontsize=9.8, fontweight="bold", color=C_TEXT, ha="left", va="center", zorder=6)
    ax.text(RX + RW - 0.012, 0.392, spec["metrics_badge"], fontsize=7.6, fontweight="bold",
            color="#1E3A5F", ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.25", facecolor="#EFF6FF", edgecolor="#93C5FD", lw=0.9),
            zorder=8)

    yc_top, yc_bot = 0.296, 0.168
    draw_robot_badge(ax, bx, yc_top - bh * 0.5, bw, bh, "A")
    draw_robot_badge(ax, bx, yc_bot - bh * 0.5, bw, bh, "B")

    for seg in spec["panel_c_items"]:
        kind = seg["kind"]
        row_y = yc_top if seg["arm"] == "A" else yc_bot
        if kind == "poses":
            draw_pose_chain(ax, seg["x0"], seg["x1"], row_y, n_dots=seg.get("dots", 5), arrow_end=seg.get("arrow", True))
        elif kind == "wait":
            draw_wait_line(ax, seg["x0"], seg["x1"], row_y, label=seg.get("label"), label_offset=seg.get("loff", 0.022))
        elif kind in ("pick", "place", "handover"):
            pal = {
                "pick": (C_PICK, C_PICK_EDGE, "#FFFFFF"),
                "place": (C_PLACE, C_PLACE_EDGE, "#FFFFFF"),
                "handover": (C_HANDOVER, C_HANDOVER_EDGE, "#1E293B"),
            }[kind]
            draw_oval_skill(ax, seg["cx"], row_y, seg.get("w", 0.068), seg.get("h", 0.070),
                            seg["label"], pal[0], pal[1], pal[2], fontsize=seg.get("fs", 6.8))
        elif kind == "sync_handover":
            cx, w_ov, h_ov = seg["cx"], seg.get("w", 0.078), seg.get("h", 0.072)
            cy_mid = (yc_top + yc_bot) * 0.5
            ax.plot([cx, cx], [yc_bot, yc_top], color=C_HANDOVER_EDGE, lw=1.4, linestyle=(0, (3, 2)), zorder=5)
            draw_oval_skill(ax, cx, cy_mid, w_ov, h_ov, seg["label"], C_HANDOVER, C_HANDOVER_EDGE, "#1E293B",
                            fontsize=seg.get("fs", 6.5))

    for dep in spec["panel_c_deps"]:
        col = C_MOTION_DEP if dep["dtype"] == "motion" else C_TASK_DEP if dep["dtype"] == "task" else "#334155"
        draw_curved_dep(ax, dep["p0"], dep["p1"], col, rad=dep.get("rad", 0.20),
                        lw=dep.get("lw", 1.25), dashed=dep.get("dashed", True))

    for note in spec.get("panel_c_notes", []):
        ax.text(note["x"], note["y"], note["text"], fontsize=note.get("fs", 6.6),
                fontstyle=note.get("style", "italic"), fontweight=note.get("weight", "normal"),
                color=note.get("color", C_TEXT), ha="center", va="center", zorder=9)

    if spec.get("fault_x_c") is not None:
        fx = spec["fault_x_c"]
        ax.plot([fx, fx], [0.105, 0.362], color=C_FAULT, lw=1.55, linestyle=(0, (4, 2)), zorder=10)
        ax.add_patch(FancyBboxPatch((fx - 0.048, 0.336), 0.096, 0.032,
                                    boxstyle="round,pad=0.001,rounding_size=0.007",
                                    facecolor=C_FAULT_BG, edgecolor=C_FAULT, lw=1.0, zorder=11))
        ax.text(fx, 0.352, "Barrier t=14.81s", fontsize=6.2, fontweight="bold",
                color=C_FAULT, ha="center", va="center", zorder=12)

    # -------------------------------------------------------------------------
    # Bottom Legend Strip
    # -------------------------------------------------------------------------
    ly = 0.060
    lx = RX + 0.020
    ax.add_patch(Ellipse((lx, ly), 0.0075, 0.018, facecolor="#FFFFFF", edgecolor="#334155", lw=1.0, zorder=7))
    ax.text(lx + 0.009, ly, "Pose Node", fontsize=7.6, color=C_TEXT, ha="left", va="center")

    lx = RX + 0.105
    ax.add_patch(FancyBboxPatch((lx, ly - 0.014), 0.016, 0.028, boxstyle="round,pad=0.001,rounding_size=0.005",
                                facecolor=C_PLACE, edgecolor=C_PLACE_EDGE, lw=0.8, zorder=7))
    ax.add_patch(Ellipse((lx + 0.026, ly), 0.016, 0.028, facecolor=C_PICK, edgecolor=C_PICK_EDGE, lw=0.8, zorder=7))
    if spec["has_handover"]:
        ax.add_patch(Ellipse((lx + 0.044, ly), 0.016, 0.028, facecolor=C_HANDOVER, edgecolor=C_HANDOVER_EDGE, lw=0.8, zorder=7))
        ax.text(lx + 0.058, ly, "Skill Node", fontsize=7.6, color=C_TEXT, ha="left", va="center")
        lx_obj = RX + 0.240
    else:
        ax.text(lx + 0.040, ly, "Skill Node", fontsize=7.6, color=C_TEXT, ha="left", va="center")
        lx_obj = RX + 0.222

    ax.add_patch(Ellipse((lx_obj, ly), 0.024, 0.026, facecolor="#FFFFFF", edgecolor=C_PICK_EDGE,
                         lw=1.05, linestyle=(0, (3, 2)), zorder=7))
    ax.text(lx_obj, ly, "$b_k$", fontsize=6.4, color=C_PICK_EDGE, ha="center", va="center", zorder=8)
    ax.text(lx_obj + 0.018, ly, "Object Node", fontsize=7.6, color=C_TEXT, ha="left", va="center")

    lx_w = lx_obj + 0.105
    ax.plot([lx_w, lx_w + 0.022], [ly, ly], color=C_WAIT, lw=1.6, zorder=7)
    ax.text(lx_w + 0.028, ly, "Wait", fontsize=7.6, color=C_TEXT, ha="left", va="center")

    lx_td = lx_w + 0.075
    draw_curved_dep(ax, (lx_td, ly - 0.012), (lx_td + 0.022, ly + 0.014), C_TASK_DEP, rad=-0.25, lw=1.15)
    ax.text(lx_td + 0.028, ly, "Task Dependency", fontsize=7.6, color=C_TEXT, ha="left", va="center")

    lx_md = lx_td + 0.132
    draw_curved_dep(ax, (lx_md, ly - 0.012), (lx_md + 0.022, ly + 0.014), C_MOTION_DEP, rad=-0.25, lw=1.15)
    ax.text(lx_md + 0.028, ly, "Motion Dependency", fontsize=7.6, color=C_TEXT, ha="left", va="center")

    png_path = OUT_DIR / f"{spec['filename']}.png"
    pdf_path = OUT_DIR / f"{spec['filename']}.pdf"
    fig.savefig(png_path, dpi=300, facecolor="#FFFFFF")
    fig.savefig(pdf_path, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved: {png_path} ({png_path.stat().st_size // 1024} KB) and .pdf")


def prepare_all_frames():
    va = ROOT / "mode_a/videos/mode_a_free_reach.mp4"
    vb = ROOT / "mode_b/videos/mode_b_strict_zones_handover.mp4"
    vbc = ROOT / "mode_b/videos/mode_b_handovers_closeup.mp4"
    vr = ROOT / "recovery_demo/videos/recovery_mode_a_to_b.mp4"

    f_env = grab_frame(va, 0.2, FRAME_DIR / "env_setup.png")
    f_end = grab_frame(va, 58.8, FRAME_DIR / "bridge_design.png")

    BOX_ENV = (0.04, 0.11, 0.96, 0.98)
    BOX_DESIGN = (0.37, 0.62, 0.63, 0.87)
    BOX_PLATE = (0.28, 0.56, 0.72, 0.97)
    BOX_ACTION = (0.14, 0.13, 0.86, 0.96)
    BOX_CLOSEUP = (0.08, 0.11, 0.92, 0.96)

    img_env = crop_pil(f_env, BOX_ENV, (360, 290))
    img_design = crop_pil(f_end, BOX_DESIGN, (360, 290))

    # Mode A frames
    a_steps = [crop_pil(grab_frame(va, t, FRAME_DIR / f"a_step_{i}.png"), BOX_PLATE, (260, 220))
               for i, t in enumerate([4.5, 11.0, 20.5, 51.0, 58.8])]
    a_collab = [crop_pil(grab_frame(va, t, FRAME_DIR / f"a_col_{i}.png"), BOX_ACTION, (340, 340))
                for i, t in enumerate([4.0, 35.5, 52.8])]

    # Mode B frames
    b_steps = [crop_pil(grab_frame(vb, t, FRAME_DIR / f"b_step_{i}.png"), BOX_PLATE, (260, 220))
               for i, t in enumerate([4.5, 16.0, 28.0, 50.0, 58.8])]
    b_collab = [
        crop_pil(grab_frame(vbc, 1.3, FRAME_DIR / "b_col_0.png"), BOX_CLOSEUP, (340, 340)),
        crop_pil(grab_frame(vbc, 8.2, FRAME_DIR / "b_col_1.png"), BOX_CLOSEUP, (340, 340)),
        crop_pil(grab_frame(vb, 56.5, FRAME_DIR / "b_col_2.png"), BOX_ACTION, (340, 340)),
    ]

    # Recovery frames (clean scene crop without clipped text banner on col_0)
    r_steps = [crop_pil(grab_frame(vr, t, FRAME_DIR / f"r_step_{i}.png"), BOX_PLATE, (260, 220))
               for i, t in enumerate([3.5, 11.5, 27.0, 53.0, 60.8])]
    r_collab = [
        crop_pil(grab_frame(vr, 13.8, FRAME_DIR / "r_col_0_clean.png"), BOX_ACTION, (340, 340)),
        crop_pil(grab_frame(vr, 24.0, FRAME_DIR / "r_col_1.png"), BOX_ACTION, (340, 340)),
        crop_pil(grab_frame(vr, 58.5, FRAME_DIR / "r_col_2.png"), BOX_ACTION, (340, 340)),
    ]

    return {
        "env": img_env, "design": img_design,
        "a_steps": a_steps, "a_collab": a_collab,
        "b_steps": b_steps, "b_collab": b_collab,
        "r_steps": r_steps, "r_collab": r_collab,
    }


def main():
    imgs = prepare_all_frames()

    # =========================================================================
    # FIGURE 1: Mode A
    # =========================================================================
    spec_a = {
        "filename": "fig_apex_mr_mode_a",
        "has_handover": False,
        "framework_title": "APEX-MR · Mode A (Free-Reach)",
        "assembly_plan_title": "Assembly Plan (18 Steps)",
        "plan_step_labels": ["$a_1$", "$a_4$", "$a_8$", "$a_{17}$", "$a_{18}$"],
        "bottom_title": "Collaborative Assembly",
        "collab_captions": ["Base Piers (Cross)", "Upper Piers (Parallel)", "Deck Seating (0 Coll.)"],
        "img_env": imgs["env"], "img_design": imgs["design"],
        "img_steps": imgs["a_steps"], "img_collab": imgs["a_collab"],
        "panel_a_title": "(a) Sequential Task Graph (Free-Reach Reference)",
        "panel_b_title": "(b) Motion Plan",
        "panel_b_note": "Native Mesh Certified · 0.08 rad / 4 mm · Via + RRTConnect",
        "panel_c_title": "(c) Parallel, Asynchronous Plan after TPG",
        "metrics_badge": "44.60 s → 38.41 s · 1.161× Speedup · 6.20 s Dual-Arm · 0 Collisions",
        "panel_a_items": [
            {"arm": "B", "kind": "transit", "x0": 0.386, "x1": 0.440, "label": "Transit", "drop_planner": "Direct", "next_arrow_to": 0.444},
            {"arm": "B", "kind": "pick", "x0": 0.444, "x1": 0.494, "label": "Pick", "id": "b_pick1",
             "obj": ("Blue_02", "below"), "drop_planner": "IK", "next_arrow_to": 0.498},
            {"arm": "B", "kind": "transit", "x0": 0.498, "x1": 0.550, "label": "Transit", "drop_planner": "Via", "next_arrow_to": 0.554},
            {"arm": "B", "kind": "place", "x0": 0.554, "x1": 0.620, "label": "Place-Down", "id": "b_place1",
             "obj": ("Blue_02", "below"), "drop_planner": "IK", "next_arrow_to": 0.624},
            {"arm": "B", "kind": "transit", "x0": 0.624, "x1": 0.676, "label": "Park", "id": "b_park1", "drop_planner": "RRT"},
            {"arm": "B", "kind": "wait", "x0": 0.678, "x1": 0.946, "label": "Wait at Outboard Park (Corridor Yield)", "loff": 0.025},
            {"arm": "B", "kind": "ellipsis", "x0": 0.950, "x1": 0.978},

            {"arm": "A", "kind": "wait", "x0": 0.386, "x1": 0.678, "label": "Wait at HOME / Park", "loff": 0.025},
            {"arm": "A", "kind": "transit", "x0": 0.680, "x1": 0.730, "label": "Transit", "id": "a_tr1", "drop_planner": "Direct", "next_arrow_to": 0.734},
            {"arm": "A", "kind": "pick", "x0": 0.734, "x1": 0.780, "label": "Pick", "id": "a_pick1",
             "obj": ("Red_01", "above"), "drop_planner": "IK", "next_arrow_to": 0.784},
            {"arm": "A", "kind": "transit", "x0": 0.784, "x1": 0.834, "label": "Transit", "drop_planner": "Via", "next_arrow_to": 0.838},
            {"arm": "A", "kind": "place", "x0": 0.838, "x1": 0.902, "label": "Place-Down", "id": "a_place1",
             "obj": ("Red_01", "above"), "drop_planner": "IK", "next_arrow_to": 0.906},
            {"arm": "A", "kind": "transit", "x0": 0.906, "x1": 0.946, "label": "Park", "drop_planner": "RRT"},
            {"arm": "A", "kind": "ellipsis", "x0": 0.950, "x1": 0.978},
        ],
        "panel_a_deps": [
            {"src": "b_park1", "dst": "a_tr1", "dx0": 0.008, "dy0": 0.028, "dx1": -0.008, "dy1": -0.028, "rad": 0.25},
        ],
        "panel_c_items": [
            {"arm": "A", "kind": "wait", "x0": 0.386, "x1": 0.486, "label": "Wait", "loff": 0.022},
            {"arm": "A", "kind": "poses", "x0": 0.486, "x1": 0.542, "dots": 4},
            {"arm": "A", "kind": "pick", "cx": 0.576, "w": 0.064, "h": 0.068, "label": "Pick\nRed_01", "fs": 6.6},
            {"arm": "A", "kind": "poses", "x0": 0.610, "x1": 0.662, "dots": 4},
            {"arm": "A", "kind": "place", "cx": 0.698, "w": 0.070, "h": 0.070, "label": "Place\nGreen_01", "fs": 6.5},
            {"arm": "A", "kind": "poses", "x0": 0.735, "x1": 0.808, "dots": 5},
            {"arm": "A", "kind": "pick", "cx": 0.842, "w": 0.066, "h": 0.068, "label": "Pick\nOrange_01", "fs": 6.3},
            {"arm": "A", "kind": "poses", "x0": 0.877, "x1": 0.972, "dots": 6},

            {"arm": "B", "kind": "poses", "x0": 0.386, "x1": 0.434, "dots": 3},
            {"arm": "B", "kind": "pick", "cx": 0.466, "w": 0.062, "h": 0.068, "label": "Pick\nBlue_02", "fs": 6.6},
            {"arm": "B", "kind": "poses", "x0": 0.499, "x1": 0.532, "dots": 2},
            {"arm": "B", "kind": "place", "cx": 0.566, "w": 0.068, "h": 0.070, "label": "Place\nBlue_02", "fs": 6.5},
            {"arm": "B", "kind": "poses", "x0": 0.602, "x1": 0.654, "dots": 4},
            {"arm": "B", "kind": "wait", "x0": 0.654, "x1": 0.716},
            {"arm": "B", "kind": "poses", "x0": 0.716, "x1": 0.768, "dots": 4},
            {"arm": "B", "kind": "pick", "cx": 0.802, "w": 0.066, "h": 0.068, "label": "Pick\nGreen_02", "fs": 6.4},
            {"arm": "B", "kind": "poses", "x0": 0.837, "x1": 0.884, "dots": 3},
            {"arm": "B", "kind": "place", "cx": 0.922, "w": 0.074, "h": 0.070, "label": "Place\nDeck", "fs": 6.6},
            {"arm": "B", "kind": "poses", "x0": 0.960, "x1": 0.978, "dots": 1},
        ],
        "panel_c_deps": [
            {"dtype": "motion", "p0": (0.494, 0.198), "p1": (0.514, 0.286), "rad": -0.22},
            {"dtype": "motion", "p0": (0.595, 0.194), "p1": (0.630, 0.286), "rad": -0.18},
            {"dtype": "task",   "p0": (0.715, 0.264), "p1": (0.742, 0.178), "rad": 0.22},
            {"dtype": "motion", "p0": (0.822, 0.200), "p1": (0.848, 0.264), "rad": -0.22},
            {"dtype": "task",   "p0": (0.892, 0.286), "p1": (0.916, 0.202), "rad": 0.22},
        ],
        "panel_c_notes": [
            {"x": 0.628, "y": 0.122, "text": "Park Yield", "fs": 6.6, "color": C_MUTED},
            {"x": 0.928, "y": 0.344, "text": "Return Home || Deck", "fs": 6.6, "color": C_MUTED},
        ],
    }

    # =========================================================================
    # FIGURE 2: Mode B (refined spacing + continuous handover lane connectors)
    # =========================================================================
    spec_b = {
        "filename": "fig_apex_mr_mode_b",
        "has_handover": True,
        "framework_title": "APEX-MR · Mode B (4× Handover)",
        "assembly_plan_title": "Assembly Plan (22 Steps, 4 Handovers)",
        "plan_step_labels": ["$a_1$", "$a_6$", "$a_{14}$", "$a_{19}$", "$a_{22}$"],
        "bottom_title": "Collaborative Assembly",
        "collab_captions": ["A→B Handover (Orange)", "B→A Handover (Green)", "Deck Seating (0 Coll.)"],
        "img_env": imgs["env"], "img_design": imgs["design"],
        "img_steps": imgs["b_steps"], "img_collab": imgs["b_collab"],
        "panel_a_title": "(a) Sequential Task Graph (Strict-Zone + Handover)",
        "panel_b_title": "(b) Motion Plan",
        "panel_b_note": "Paired IK Branches · 180° Mirror Avoidance · Via + RRT",
        "panel_c_title": "(c) Parallel, Asynchronous Plan after TPG",
        "metrics_badge": "77.81 s → 62.25 s · 1.250× Speedup · 15.56 s (25.0%) Dual-Arm · 0 Coll.",
        "panel_a_items": [
            {"arm": "A", "kind": "wait", "x0": 0.386, "x1": 0.506, "label": "Wait (Zone A)", "loff": 0.025},
            {"arm": "A", "kind": "transit", "x0": 0.508, "x1": 0.556, "label": "Transit", "drop_planner": "Direct", "next_arrow_to": 0.560},
            {"arm": "A", "kind": "pick", "x0": 0.560, "x1": 0.610, "label": "Pick", "id": "ba_pick",
             "obj": ("Orange_01", "above"), "obj_w": 0.076, "obj_dx": -0.010, "drop_planner": "IK", "next_arrow_to": 0.614},
            {"arm": "A", "kind": "handover", "x0": 0.614, "x1": 0.684, "label": "Handover", "id": "ba_ho",
             "obj": ("hand_mid", "above"), "obj_w": 0.072, "obj_dx": 0.010, "drop_planner": "Via", "drop_box": False},
            {"arm": "A", "kind": "wait", "x0": 0.686, "x1": 0.808, "label": "Retreat / Next Pick", "loff": 0.025},
            {"arm": "A", "kind": "pick", "x0": 0.810, "x1": 0.864, "label": "Pick", "id": "ba_pick2",
             "obj": ("Red_02", "above"), "obj_w": 0.066, "obj_dx": -0.008, "drop_planner": "IK", "next_arrow_to": 0.868},
            {"arm": "A", "kind": "place", "x0": 0.868, "x1": 0.944, "label": "Place-Down", "id": "ba_pl2",
             "obj": ("Red_02", "above"), "obj_w": 0.066, "obj_dx": 0.008, "drop_planner": "IK"},
            {"arm": "A", "kind": "ellipsis", "x0": 0.950, "x1": 0.978},

            {"arm": "B", "kind": "pick", "x0": 0.386, "x1": 0.436, "label": "Pick", "id": "bb_pick",
             "obj": ("Blue_01", "below"), "obj_w": 0.064, "obj_dx": -0.008, "drop_planner": "IK", "next_arrow_to": 0.440},
            {"arm": "B", "kind": "place", "x0": 0.440, "x1": 0.506, "label": "Place-Down", "id": "bb_pl",
             "obj": ("Blue_01", "below"), "obj_w": 0.064, "obj_dx": 0.008, "drop_planner": "IK"},
            {"arm": "B", "kind": "wait", "x0": 0.508, "x1": 0.612, "label": "Wait", "loff": -0.025},
            {"arm": "B", "kind": "handover", "x0": 0.614, "x1": 0.684, "label": "Receive", "id": "bb_ho",
             "drop_planner": "Via", "drop_box": True, "next_arrow_to": 0.688},
            {"arm": "B", "kind": "transit", "x0": 0.688, "x1": 0.734, "label": "Transit", "drop_planner": "Direct", "next_arrow_to": 0.738},
            {"arm": "B", "kind": "place", "x0": 0.738, "x1": 0.808, "label": "Place-Down", "id": "bb_pl2",
             "obj": ("Orange_01", "below"), "obj_w": 0.078, "drop_planner": "IK"},
            {"arm": "B", "kind": "wait", "x0": 0.810, "x1": 0.946, "label": "Wait / Zone B Tasks", "loff": -0.025},
            {"arm": "B", "kind": "ellipsis", "x0": 0.950, "x1": 0.978},
        ],
        "panel_a_deps": [
            {"src": "bb_pl", "dst": "ba_ho", "dx0": 0.022, "dy0": 0.028, "dx1": -0.022, "dy1": -0.028, "rad": 0.16},
            {"src": "ba_ho", "dst": "bb_ho", "dx0": 0.000, "dy0": -0.028, "dx1": 0.000, "dy1": 0.028, "rad": 0.0},
        ],
        "panel_c_items": [
            {"arm": "A", "kind": "poses", "x0": 0.386, "x1": 0.428, "dots": 3},
            {"arm": "A", "kind": "pick", "cx": 0.464, "w": 0.068, "h": 0.068, "label": "Pick\nOrange_01", "fs": 6.2},
            {"arm": "A", "kind": "poses", "x0": 0.500, "x1": 0.530, "dots": 2, "arrow": False},
            {"arm": "A", "kind": "poses", "x0": 0.616, "x1": 0.646, "dots": 2},
            {"arm": "A", "kind": "pick", "cx": 0.682, "w": 0.066, "h": 0.068, "label": "Pick\nRed_02", "fs": 6.5},
            {"arm": "A", "kind": "poses", "x0": 0.717, "x1": 0.748, "dots": 2, "arrow": False},
            {"arm": "A", "kind": "poses", "x0": 0.834, "x1": 0.864, "dots": 2},
            {"arm": "A", "kind": "place", "cx": 0.900, "w": 0.070, "h": 0.070, "label": "Place\nGreen_01*", "fs": 6.2},
            {"arm": "A", "kind": "poses", "x0": 0.937, "x1": 0.976, "dots": 3},

            {"arm": "B", "kind": "poses", "x0": 0.386, "x1": 0.424, "dots": 2},
            {"arm": "B", "kind": "place", "cx": 0.460, "w": 0.068, "h": 0.070, "label": "Place\nBlue_01", "fs": 6.4},
            {"arm": "B", "kind": "poses", "x0": 0.496, "x1": 0.530, "dots": 2, "arrow": False},
            {"arm": "B", "kind": "poses", "x0": 0.616, "x1": 0.646, "dots": 2},
            {"arm": "B", "kind": "place", "cx": 0.682, "w": 0.070, "h": 0.070, "label": "Place\nOrange_01", "fs": 6.1},
            {"arm": "B", "kind": "poses", "x0": 0.719, "x1": 0.748, "dots": 2, "arrow": False},
            {"arm": "B", "kind": "poses", "x0": 0.834, "x1": 0.880, "dots": 3},
            {"arm": "B", "kind": "place", "cx": 0.920, "w": 0.070, "h": 0.070, "label": "Place\nDeck", "fs": 6.6},
            {"arm": "B", "kind": "poses", "x0": 0.957, "x1": 0.978, "dots": 1},

            {"arm": "A", "kind": "sync_handover", "cx": 0.573, "w": 0.080, "h": 0.072, "label": "Handover\nA→B (×2)", "fs": 6.4},
            {"arm": "A", "kind": "sync_handover", "cx": 0.791, "w": 0.080, "h": 0.072, "label": "Handover\nB→A (×2)", "fs": 6.4},
        ],
        "panel_c_deps": [
            # Smooth solid lane connectors into & out of both synchronized handover ovals
            {"dtype": "solid", "p0": (0.530, 0.296), "p1": (0.544, 0.256), "rad": -0.15, "dashed": False, "lw": 1.15},
            {"dtype": "solid", "p0": (0.530, 0.168), "p1": (0.544, 0.208), "rad":  0.15, "dashed": False, "lw": 1.15},
            {"dtype": "solid", "p0": (0.602, 0.256), "p1": (0.616, 0.296), "rad": -0.15, "dashed": False, "lw": 1.15},
            {"dtype": "solid", "p0": (0.602, 0.208), "p1": (0.616, 0.168), "rad":  0.15, "dashed": False, "lw": 1.15},

            {"dtype": "solid", "p0": (0.748, 0.296), "p1": (0.762, 0.256), "rad": -0.15, "dashed": False, "lw": 1.15},
            {"dtype": "solid", "p0": (0.748, 0.168), "p1": (0.762, 0.208), "rad":  0.15, "dashed": False, "lw": 1.15},
            {"dtype": "solid", "p0": (0.820, 0.256), "p1": (0.834, 0.296), "rad": -0.15, "dashed": False, "lw": 1.15},
            {"dtype": "solid", "p0": (0.820, 0.208), "p1": (0.834, 0.168), "rad":  0.15, "dashed": False, "lw": 1.15},

            # True TPG dependencies
            {"dtype": "motion", "p0": (0.708, 0.198), "p1": (0.728, 0.288), "rad": -0.20},
            {"dtype": "task",   "p0": (0.908, 0.262), "p1": (0.914, 0.202), "rad":  0.18},
        ],
        "panel_c_notes": [
            {"x": 0.464, "y": 0.354, "text": "Parallel Zone A || Zone B", "fs": 6.6, "color": C_MOTION_DEP, "weight": "bold"},
            {"x": 0.900, "y": 0.354, "text": "*180° Mirror Avoidance", "fs": 6.3, "color": C_MUTED},
        ],
    }

    # =========================================================================
    # FIGURE 3: Online Recovery (100% zero overlap in (a), (b), (c)!)
    # =========================================================================
    spec_r = {
        "filename": "fig_apex_mr_recovery",
        "has_handover": True,
        "mbw": 0.042,
        "framework_title": "APEX-MR+ · Online Recovery (db77489)",
        "assembly_plan_title": "Normal Prefix → Recovered Plan",
        "plan_step_labels": ["$a_1$", "$a_8$", "$r_2$", "$r_{12}$", "$r_{14}$"],
        "bottom_title": "Fault-Tolerant Execution",
        "collab_captions": ["Fault Barrier (4 Piers)", "Recovered Handover", "Bridge Completed (0 Coll.)"],
        "img_env": imgs["env"], "img_design": imgs["design"],
        "img_steps": imgs["r_steps"], "img_collab": imgs["r_collab"],
        "panel_a_title": "(a) Sequential Task Graph (Prefix + Recovered Suffix)",
        "panel_b_title": "(b) Motion Plan (Continuous Single-Session Compilation)",
        "panel_b_note": "L1 Backup <1 ms · L2 Graph Reuse 105/106 (99.1%) · Barrier Drift 0",
        "panel_c_title": "(c) Parallel, Asynchronous Plan after TPG (Full-Track Verified)",
        "metrics_badge": "75.42 s → 69.76 s · 1.081× · 4/4 Handovers · 6,019-Frame Replay 0 Coll.",
        "fault_x_a": 0.618,
        "fault_badge_x_a": 0.602,
        "fault_label_a": "Fault → db77489\n105/106 Reused",
        "fault_x_c": 0.594,
        "panel_a_items": [
            # Strictly staggered horizontal slots so every single block in (b) has its own clean column!
            # Slot 1 (0.386..0.436): B Pick Blue_01/02
            {"arm": "B", "kind": "pick", "x0": 0.386, "x1": 0.434, "label": "Pick", "id": "rb_p1",
             "obj": ("Blue_01/02", "below"), "obj_w": 0.074, "drop_planner": "IK", "next_arrow_to": 0.438},
            # Slot 2 (0.438..0.496): B Place+Park Blue
            {"arm": "B", "kind": "place", "x0": 0.438, "x1": 0.496, "label": "Place+Park", "id": "rb_pl1",
             "fs": 6.4, "drop_planner": "Via"},
            {"arm": "B", "kind": "wait", "x0": 0.498, "x1": 0.612, "label": "Wait at Barrier", "loff": -0.025},

            # Slot 3 (0.500..0.548): A Pick Red_02/01
            {"arm": "A", "kind": "wait", "x0": 0.386, "x1": 0.498, "label": "Wait", "loff": 0.024},
            {"arm": "A", "kind": "pick", "x0": 0.500, "x1": 0.548, "label": "Pick", "id": "ra_p1",
             "obj": ("Red_02/01", "above"), "obj_w": 0.072, "obj_dx": -0.029, "drop_planner": "IK", "next_arrow_to": 0.552},
            # Slot 4 (0.552..0.610): A Place+Park Red
            {"arm": "A", "kind": "place", "x0": 0.552, "x1": 0.610, "label": "Place+Park", "id": "ra_pl1",
             "fs": 6.4, "drop_planner": "Via"},

            # --- FAULT BARRIER at x = 0.618 ---

            # Slot 5 (0.630..0.680): A Pick Orange (Recovered)
            {"arm": "A", "kind": "pick", "x0": 0.630, "x1": 0.680, "label": "Pick", "id": "ra_r1",
             "obj": ("Orange_01", "above"), "obj_w": 0.068, "obj_dx": 0.041, "drop_planner": "IK", "next_arrow_to": 0.684},
            # Slot 6 (0.684..0.748): Synchronized Handover A->B (single shared box in (b))
            {"arm": "A", "kind": "handover", "x0": 0.684, "x1": 0.748, "label": "Handover", "id": "ra_ho",
             "obj": ("hand_mid", "above"), "obj_w": 0.066, "obj_dx": 0.057, "drop_planner": "Via", "drop_box": False},
            {"arm": "A", "kind": "wait", "x0": 0.750, "x1": 0.818, "label": "B→A", "loff": 0.024},
            # Slot 8 (0.820..0.880): A Place Green_01
            {"arm": "A", "kind": "place", "x0": 0.820, "x1": 0.880, "label": "Place-Down", "id": "ra_r2",
             "fs": 6.5, "obj": ("Green_01", "above"), "obj_w": 0.068, "obj_dx": 0.008, "drop_planner": "IK"},
            {"arm": "A", "kind": "wait", "x0": 0.882, "x1": 0.946, "label": "Park", "loff": 0.024},
            {"arm": "A", "kind": "ellipsis", "x0": 0.950, "x1": 0.978},

            {"arm": "B", "kind": "wait", "x0": 0.626, "x1": 0.682, "label": "Wait", "loff": -0.024},
            {"arm": "B", "kind": "handover", "x0": 0.684, "x1": 0.748, "label": "Receive", "id": "rb_ho",
             "drop_planner": "Via", "drop_box": True, "next_arrow_to": 0.752},
            # Slot 7 (0.752..0.814): B Place Orange_01
            {"arm": "B", "kind": "place", "x0": 0.752, "x1": 0.814, "label": "Place-Down", "id": "rb_r1",
             "fs": 6.5, "obj": ("Orange_01", "below"), "obj_w": 0.072, "drop_planner": "IK"},
            {"arm": "B", "kind": "wait", "x0": 0.816, "x1": 0.882},
            # Slot 9 (0.884..0.942): B Place Deck
            {"arm": "B", "kind": "place", "x0": 0.884, "x1": 0.942, "label": "Place Deck", "id": "rb_dk",
             "fs": 6.4, "obj": ("Deck_12x4", "below"), "obj_w": 0.074, "drop_planner": "Via"},
            {"arm": "B", "kind": "ellipsis", "x0": 0.950, "x1": 0.978},
        ],
        "panel_a_deps": [
            {"src": "rb_pl1", "dst": "ra_r1", "dx0": 0.020, "dy0": 0.028, "dx1": -0.014, "dy1": -0.028, "rad": 0.16},
            {"src": "ra_pl1", "dst": "ra_r1", "dx0": 0.026, "dy0": 0.000, "dx1": -0.022, "dy1":  0.000, "rad": 0.0},
            {"src": "ra_ho",  "dst": "rb_ho", "dx0": 0.000, "dy0": -0.028, "dx1": 0.000, "dy1":  0.028, "rad": 0.0},
            {"src": "ra_r2",  "dst": "rb_dk", "dx0": 0.010, "dy0": -0.028, "dx1": -0.010, "dy1": 0.028, "rad": 0.16},
        ],
        "panel_c_items": [
            # Prefix parallel base piers (x: 0.386..0.590) -> Fault Barrier (0.594) -> Recovered suffix (0.600..0.978)
            {"arm": "A", "kind": "poses", "x0": 0.386, "x1": 0.428, "dots": 3},
            {"arm": "A", "kind": "pick", "cx": 0.460, "w": 0.062, "h": 0.066, "label": "Pick\nRed_02/01", "fs": 6.0},
            {"arm": "A", "kind": "place", "cx": 0.526, "w": 0.066, "h": 0.068, "label": "Place\nRed_02/01", "fs": 5.9},
            {"arm": "A", "kind": "poses", "x0": 0.561, "x1": 0.590, "dots": 2},

            {"arm": "B", "kind": "poses", "x0": 0.386, "x1": 0.422, "dots": 2},
            {"arm": "B", "kind": "pick", "cx": 0.454, "w": 0.062, "h": 0.066, "label": "Pick\nBlue_01/02", "fs": 5.9},
            {"arm": "B", "kind": "place", "cx": 0.520, "w": 0.066, "h": 0.068, "label": "Place\nBlue_01/02", "fs": 5.8},
            {"arm": "B", "kind": "poses", "x0": 0.555, "x1": 0.590, "dots": 2},

            # Recovered Suffix
            {"arm": "A", "kind": "poses", "x0": 0.600, "x1": 0.628, "dots": 2},
            {"arm": "A", "kind": "pick", "cx": 0.660, "w": 0.062, "h": 0.066, "label": "Pick\nOrange", "fs": 6.3},
            {"arm": "A", "kind": "poses", "x0": 0.774, "x1": 0.808, "dots": 2, "arrow": False},
            {"arm": "A", "kind": "place", "cx": 0.896, "w": 0.066, "h": 0.068, "label": "Place\nGreen_01", "fs": 6.1},
            {"arm": "A", "kind": "poses", "x0": 0.931, "x1": 0.976, "dots": 3},

            {"arm": "B", "kind": "wait", "x0": 0.600, "x1": 0.692, "label": "Wait", "loff": -0.022},
            {"arm": "B", "kind": "place", "cx": 0.794, "w": 0.064, "h": 0.068, "label": "Place\nOrange", "fs": 6.2},
            {"arm": "B", "kind": "place", "cx": 0.922, "w": 0.066, "h": 0.068, "label": "Place\nDeck", "fs": 6.4},
            {"arm": "B", "kind": "poses", "x0": 0.957, "x1": 0.978, "dots": 1},

            {"arm": "A", "kind": "sync_handover", "cx": 0.732, "w": 0.074, "h": 0.068, "label": "Handover\nA→B (×2)", "fs": 6.2},
            {"arm": "A", "kind": "sync_handover", "cx": 0.848, "w": 0.074, "h": 0.068, "label": "Handover\nB→A (×2)", "fs": 6.2},
        ],
        "panel_c_deps": [
            # Clean solid flow connectors into/out of recovered handover ovals
            {"dtype": "solid", "p0": (0.691, 0.288), "p1": (0.706, 0.256), "rad": -0.15, "dashed": False, "lw": 1.1},
            {"dtype": "solid", "p0": (0.692, 0.168), "p1": (0.706, 0.208), "rad":  0.15, "dashed": False, "lw": 1.1},
            {"dtype": "solid", "p0": (0.760, 0.256), "p1": (0.774, 0.296), "rad": -0.15, "dashed": False, "lw": 1.1},
            {"dtype": "solid", "p0": (0.758, 0.206), "p1": (0.768, 0.184), "rad":  0.12, "dashed": False, "lw": 1.1},
            {"dtype": "solid", "p0": (0.808, 0.296), "p1": (0.822, 0.256), "rad": -0.15, "dashed": False, "lw": 1.1},
            {"dtype": "solid", "p0": (0.822, 0.184), "p1": (0.830, 0.202), "rad":  0.12, "dashed": False, "lw": 1.1},
            {"dtype": "solid", "p0": (0.872, 0.258), "p1": (0.876, 0.270), "rad": -0.10, "dashed": False, "lw": 1.1},
            {"dtype": "solid", "p0": (0.874, 0.204), "p1": (0.892, 0.180), "rad":  0.15, "dashed": False, "lw": 1.1},

            # True TPG barrier & completion dependencies
            {"dtype": "task",   "p0": (0.576, 0.178), "p1": (0.610, 0.286), "rad": -0.22},
            {"dtype": "motion", "p0": (0.794, 0.204), "p1": (0.794, 0.286), "rad":  0.22},
            {"dtype": "task",   "p0": (0.908, 0.262), "p1": (0.916, 0.202), "rad":  0.18},
        ],
        "panel_c_notes": [
            {"x": 0.476, "y": 0.354, "text": "Normal Prefix (4 Base Piers)", "fs": 6.4, "color": C_MUTED},
            {"x": 0.792, "y": 0.354, "text": "Recovered Mode B Suffix (4× Handover)", "fs": 6.4, "color": C_FAULT, "weight": "bold"},
        ],
    }

    for sp in (spec_a, spec_b, spec_r):
        build_figure(sp)


if __name__ == "__main__":
    main()
