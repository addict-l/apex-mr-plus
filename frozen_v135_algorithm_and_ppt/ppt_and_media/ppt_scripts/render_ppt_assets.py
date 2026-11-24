#!/usr/bin/env python3
"""Render bespoke 300-DPI academic figures and clean video poster frames for the PPT."""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
import numpy as np
from PIL import Image

# Use Microsoft YaHei for crisp Chinese + English academic typography
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"D:\APEX_MR_Plus\dualarm_skillnet")
SHOWCASE = ROOT / "results/sim_v1_3/motion_v135/showcases/lego_bridge_9blocks"
PRES_DIR = SHOWCASE / "presentation"
ASSET_DIR = PRES_DIR / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#0F2942"
DARK_SLATE = "#1E293B"
TEXT = "#334155"
MUTED = "#64748B"
BLUE = "#2563EB"
BLUE_LT = "#EFF6FF"
BLUE_BD = "#93C5FD"
TEAL = "#0D9488"
TEAL_LT = "#F0FDFA"
TEAL_BD = "#5EEAD4"
CORAL = "#DC2626"
CORAL_LT = "#FEF2F2"
CORAL_BD = "#FCA5A5"
AMBER = "#D97706"
AMBER_LT = "#FFFBEB"
AMBER_BD = "#FCD34D"
PURPLE = "#7C3AED"
PURPLE_LT = "#F5F3FF"
PURPLE_BD = "#C4B5FD"
BORDER = "#CBD5E1"
BG = "#FFFFFF"


def grab_frame(video_path: Path, t_sec: float, out_path: Path) -> Path:
    cmd = [
        "ffmpeg", "-y", "-ss", f"{t_sec:.2f}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2", str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out_path


def crop_save(src: Path, box: tuple[float, float, float, float], dst: Path, size: tuple[int, int] | None = None):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    cropped = im.crop((int(w * box[0]), int(h * box[1]), int(w * box[2]), int(h * box[3])))
    if size is not None:
        cropped = cropped.resize(size, Image.Resampling.LANCZOS)
    cropped.save(dst, quality=95)


def draw_rbox(ax, x, y, w, h, fc="#FFFFFF", ec=BORDER, lw=1.2, r=0.015, ls="-", z=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.002,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def draw_arrow(ax, p0, p1, color=NAVY, lw=1.8, rad=0.0, ls="-", z=6, ms=12):
    arr = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                          color=color, lw=lw, linestyle=ls,
                          connectionstyle=f"arc3,rad={rad}", zorder=z)
    ax.add_patch(arr)


def render_system_io():
    """Figure 1: Problem & Input-Output Architecture Diagram for Slide 3."""
    fig, ax = plt.subplots(figsize=(12.2, 4.35), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    # --- LEFT COLUMN: INPUT ---
    draw_rbox(ax, 0.012, 0.05, 0.255, 0.90, fc="#F8FAFC", ec="#94A3B8", lw=1.2, ls="--", r=0.02, z=1)
    ax.text(0.140, 0.905, "输入  INPUT", fontsize=11.5, fontweight="bold", color=NAVY, ha="center", va="center")

    in_cards = [
        (0.655, BLUE_LT, BLUE_BD, BLUE, "1. 终态结构与场景定义 (TOML)",
         "双臂初态位姿、料区/槽位几何连接\n仅需目标终态结构，无需预设安装序"),
        (0.375, TEAL_LT, TEAL_BD, TEAL, "2. 参数化原子技能库 (9 类技能)",
         "pick / place / handover / insert / detach\nsupport_on / support_off / co_pick / co_insert"),
        (0.095, AMBER_LT, AMBER_BD, AMBER, "3. 三值能力表 (Capability Table)",
         "显式编码 reach / liftable 可行性\n支持 VALID / INVALID / UNKNOWN 三态"),
    ]
    for y0, fc, ec, tc, title, body in in_cards:
        draw_rbox(ax, 0.028, y0, 0.223, 0.215, fc=fc, ec=ec, lw=1.1, r=0.016, z=3)
        ax.text(0.042, y0 + 0.165, title, fontsize=8.8, fontweight="bold", color=tc, ha="left", va="center", zorder=4)
        ax.text(0.042, y0 + 0.075, body, fontsize=7.6, color=DARK_SLATE, ha="left", va="center", linespacing=1.45, zorder=4)

    # --- CENTER COLUMN: CORE SOLVER ---
    draw_rbox(ax, 0.315, 0.05, 0.405, 0.90, fc="#FFFFFF", ec=NAVY, lw=1.6, r=0.022, z=1)
    draw_rbox(ax, 0.315, 0.845, 0.405, 0.105, fc=NAVY, ec=NAVY, lw=1.0, r=0.018, z=2)
    ax.text(0.5175, 0.897, "双臂技能网络与时空联合规划引擎 (SkillNet + APEX-MR)",
            fontsize=10.2, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=4)

    # Layer 1: Task Planning
    draw_rbox(ax, 0.335, 0.560, 0.365, 0.245, fc=BLUE_LT, ec=BLUE, lw=1.2, r=0.016, z=3)
    ax.text(0.350, 0.768, "① 上层：符号任务图搜索与剪枝 (SkillNet)", fontsize=9.2, fontweight="bold", color=BLUE, ha="left", va="center", zorder=4)
    ax.text(0.350, 0.655,
            "• 技能前后置条件自动构建技能网络 + 反向相关性剪枝\n"
            "• 事件状态 A* 搜索（可采纳启发：HMax + 抽象图 + 事件抽象图）\n"
            "• 自动联合求解：合法装配序 + 双臂分工 + 交接/托举策略",
            fontsize=7.8, color=DARK_SLATE, ha="left", va="center", linespacing=1.55, zorder=4)

    # Layer 2: Motion Compilation
    draw_rbox(ax, 0.335, 0.310, 0.365, 0.200, fc=TEAL_LT, ec=TEAL, lw=1.2, r=0.016, z=3)
    ax.text(0.350, 0.475, "② 中层：技能原语展开与物理运动编译", fontsize=9.2, fontweight="bold", color=TEAL, ha="left", va="center", zorder=4)
    ax.text(0.350, 0.382,
            "• 展开为 approach / grasp / insert 等底层运动原语\n"
            "• 原生网格碰撞认证：配对 IK 分支 + 180°镜像避障 + Via / RRTConnect",
            fontsize=7.8, color=DARK_SLATE, ha="left", va="center", linespacing=1.52, zorder=4)

    # Layer 3: TPG Scheduling
    draw_rbox(ax, 0.335, 0.085, 0.365, 0.175, fc=PURPLE_LT, ec=PURPLE, lw=1.2, r=0.016, z=3)
    ax.text(0.350, 0.225, "③ 下层：TPG 异步时空重排与动力学闭环", fontsize=9.2, fontweight="bold", color=PURPLE, ha="left", va="center", zorder=4)
    ax.text(0.350, 0.142,
            "• 自动提取 Task / Motion 依赖边，压缩冗余等待时间\n"
            "• 独立原生网格动力学回放验证：0 未授权碰撞 + 亚皮米落位精度",
            fontsize=7.8, color=DARK_SLATE, ha="left", va="center", linespacing=1.52, zorder=4)

    draw_arrow(ax, (0.5175, 0.560), (0.5175, 0.512), color=NAVY, lw=1.6, ms=11)
    draw_arrow(ax, (0.5175, 0.310), (0.5175, 0.262), color=NAVY, lw=1.6, ms=11)

    # --- RIGHT COLUMN: OUTPUT ---
    draw_rbox(ax, 0.768, 0.05, 0.220, 0.90, fc="#F8FAFC", ec="#94A3B8", lw=1.2, ls="--", r=0.02, z=1)
    ax.text(0.878, 0.905, "输出  OUTPUT", fontsize=11.5, fontweight="bold", color=NAVY, ha="center", va="center")

    out_cards = [
        (0.610, "#ECFDF5", "#6EE7B7", "#047857", "1. 符号任务图 & 时间表",
         "双臂分配序列与因果依赖\n（含交接/托举协同节点）\nplan.json / taskgraph.json"),
        (0.345, BLUE_LT, BLUE_BD, BLUE, "2. 异步并行执行图 (TPG)",
         "双臂解耦异步推进节点\n精确标注 Task & Motion\n两类避碰同步依赖边"),
        (0.095, CORAL_LT, CORAL_BD, CORAL, "3. 可执行零碰撞双臂轨迹",
         "CoppeliaSim 原生网格认证\n全程 0 违规 / 0 碰撞\n直接驱动物理仿真与实机"),
    ]
    for y0, fc, ec, tc, title, body in out_cards:
        draw_rbox(ax, 0.783, y0, 0.190, 0.225, fc=fc, ec=ec, lw=1.1, r=0.016, z=3)
        ax.text(0.796, y0 + 0.178, title, fontsize=8.6, fontweight="bold", color=tc, ha="left", va="center", zorder=4)
        ax.text(0.796, y0 + 0.078, body, fontsize=7.5, color=DARK_SLATE, ha="left", va="center", linespacing=1.42, zorder=4)

    # Big connector arrows between columns
    draw_arrow(ax, (0.270, 0.50), (0.312, 0.50), color=BLUE, lw=2.6, ms=15)
    draw_arrow(ax, (0.723, 0.50), (0.765, 0.50), color=TEAL, lw=2.6, ms=15)

    out = ASSET_DIR / "fig_ppt_system_io.png"
    fig.savefig(out, dpi=300, facecolor="#FFFFFF")
    plt.close(fig)
    print("Saved:", out)


def render_walkthrough():
    """Figure 2: Concrete Simple-Example End-to-End Walkthrough for Slide 4."""
    fig, ax = plt.subplots(figsize=(12.2, 4.65), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    # 4 vertical stage columns
    stages = [
        (0.010, 0.228, BLUE, BLUE_LT, "STEP 1 · 问题与三值能力建模"),
        (0.260, 0.232, TEAL, TEAL_LT, "STEP 2 · 反向剪枝与事件 A*"),
        (0.514, 0.226, AMBER, AMBER_LT, "STEP 3 · 原语展开与网格编译"),
        (0.762, 0.228, PURPLE, PURPLE_LT, "STEP 4 · TPG 异步并行重排"),
    ]

    for x0, w, col, bg, title in stages:
        draw_rbox(ax, x0, 0.04, w, 0.92, fc="#FFFFFF", ec=col, lw=1.35, r=0.018, z=1)
        draw_rbox(ax, x0, 0.855, w, 0.105, fc=col, ec=col, lw=1.0, r=0.015, z=2)
        ax.text(x0 + w * 0.5, 0.907, title, fontsize=8.8, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=4)

    # Inter-stage arrows
    for ax_x in (0.240, 0.494, 0.742):
        draw_arrow(ax, (ax_x, 0.50), (ax_x + 0.019, 0.50), color=NAVY, lw=2.2, ms=13)

    # --- STEP 1 Content: Concrete mini example (Block b1 at Left stock -> Right slot s1) ---
    ax.text(0.124, 0.800, "极简样例：跨区积木 b1 装入右槽 s1", fontsize=8.4, fontweight="bold", color=NAVY, ha="center", va="center")
    # Mini table diagram
    draw_rbox(ax, 0.026, 0.49, 0.196, 0.265, fc="#F8FAFC", ec=BORDER, lw=1.0, r=0.012, z=3)
    # Zone A & Zone B shading
    ax.add_patch(Rectangle((0.032, 0.505), 0.088, 0.235, facecolor="#DCFCE7", alpha=0.55, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((0.128, 0.505), 0.088, 0.235, facecolor="#DBEAFE", alpha=0.55, edgecolor="none", zorder=3))
    ax.plot([0.124, 0.124], [0.505, 0.740], color="#94A3B8", lw=1.1, ls="--", zorder=4)
    ax.text(0.076, 0.720, "Arm A 区域", fontsize=7.0, fontweight="bold", color="#15803D", ha="center", zorder=5)
    ax.text(0.172, 0.720, "Arm B 区域", fontsize=7.0, fontweight="bold", color="#1D4ED8", ha="center", zorder=5)
    # Block b1, Handover pt, Slot s1
    draw_rbox(ax, 0.044, 0.575, 0.048, 0.068, fc="#F97316", ec="#C2410C", lw=1.0, r=0.008, z=5)
    ax.text(0.068, 0.609, "b1", fontsize=8.5, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=6)
    ax.text(0.068, 0.542, "料区 L_A", fontsize=6.6, color=TEXT, ha="center", zorder=5)

    ax.add_patch(Ellipse((0.124, 0.609), 0.030, 0.062, facecolor="#FEF3C7", edgecolor=AMBER, lw=1.1, ls="--", zorder=5))
    ax.text(0.124, 0.609, "H", fontsize=7.5, fontweight="bold", color=AMBER, ha="center", va="center", zorder=6)
    ax.text(0.124, 0.542, "交接点", fontsize=6.4, color=AMBER, ha="center", zorder=5)

    draw_rbox(ax, 0.156, 0.575, 0.048, 0.068, fc="#FFFFFF", ec=BLUE, lw=1.2, ls="--", r=0.008, z=5)
    ax.text(0.180, 0.609, "s1", fontsize=8.5, fontweight="bold", color=BLUE, ha="center", va="center", zorder=6)
    ax.text(0.180, 0.542, "目标槽位", fontsize=6.6, color=TEXT, ha="center", zorder=5)

    # Capability mini table
    draw_rbox(ax, 0.026, 0.08, 0.196, 0.375, fc=BLUE_LT, ec=BLUE_BD, lw=1.0, r=0.012, z=3)
    ax.text(0.124, 0.422, "三值能力表 & 目标终态", fontsize=8.0, fontweight="bold", color=BLUE, ha="center", va="center", zorder=4)
    ax.text(0.036, 0.255,
            "Goal: installed(b1, s1) ∧ free(A,B)\n"
            "────────────────────\n"
            "reach(A, L_A) = VALID\n"
            "reach(A, s1)  = INVALID  ✕\n"
            "reach(B, L_A) = INVALID  ✕\n"
            "reach(B, s1)  = VALID\n"
            "reach(A/B, H) = VALID    ✓",
            fontsize=7.2, family="DejaVu Sans", color=DARK_SLATE, ha="left", va="center", linespacing=1.48, zorder=4)

    # --- STEP 2 Content: Backward Pruning + Event A* ---
    ax.text(0.376, 0.800, "从终态倒推：自动发现必须交接", fontsize=8.4, fontweight="bold", color=NAVY, ha="center", va="center")
    # Backward chain boxes
    b_nodes = [
        (0.695, "#DCFCE7", "#15803D", "目标事实：installed(b1, s1)"),
        (0.560, BLUE_LT, BLUE, "倒推 ①：insert(B, b1, s1)  [A不可达已剪枝]"),
        (0.425, AMBER_LT, AMBER, "倒推 ②：handover(A→B, b1, H)"),
        (0.290, TEAL_LT, TEAL, "倒推 ③：pick(A, b1, L_A)  [满足初态!]"),
    ]
    for y0, fc, ec, txt in b_nodes:
        draw_rbox(ax, 0.275, y0, 0.202, 0.082, fc=fc, ec=ec, lw=1.1, r=0.012, z=3)
        ax.text(0.376, y0 + 0.041, txt, fontsize=7.2, fontweight="bold", color=DARK_SLATE, ha="center", va="center", zorder=4)
    for y_top, y_bot in [(0.695, 0.644), (0.560, 0.509), (0.425, 0.374)]:
        draw_arrow(ax, (0.376, y_top), (0.376, y_bot), color=TEAL, lw=1.4, ms=9)

    draw_rbox(ax, 0.275, 0.08, 0.202, 0.175, fc="#F8FAFC", ec=TEAL_BD, lw=1.0, r=0.012, z=3)
    ax.text(0.376, 0.220, "事件状态 A* 最优搜索", fontsize=7.9, fontweight="bold", color=TEAL, ha="center", va="center", zorder=4)
    ax.text(0.285, 0.140,
            "• 排序键 f = g + max(HMax, PDB, EventPDB)\n"
            "• 219 样本检验 0 高估，保证时间最优\n"
            "• 无预设顺序，自动补齐交接/托举步骤",
            fontsize=6.9, color=DARK_SLATE, ha="left", va="center", linespacing=1.42, zorder=4)

    # --- STEP 3 Content: Primitive Expansion & Motion Compilation ---
    ax.text(0.627, 0.800, "动作展开与原生网格运动认证", fontsize=8.4, fontweight="bold", color=NAVY, ha="center", va="center")
    p_cards = [
        (0.630, BLUE_LT, BLUE, "pick(A, b1, L_A)", "approach(A) → grasp(b1) → lift\n[IK 分支 + Via 避障]"),
        (0.455, AMBER_LT, AMBER, "handover(A→B, b1, H)", "approach(A,H) || approach(B,H)\n180°镜像配对 IK → grasp(B) → rel(A)"),
        (0.280, CORAL_LT, CORAL, "insert(B, b1, s1)", "approach(B,s1) → align → insert\n[RRTConnect + 网格碰撞校验]"),
    ]
    for y0, fc, ec, hd, bd in p_cards:
        draw_rbox(ax, 0.528, y0, 0.198, 0.132, fc=fc, ec=ec, lw=1.1, r=0.012, z=3)
        ax.text(0.538, y0 + 0.098, hd, fontsize=7.6, fontweight="bold", color=ec, ha="left", va="center", zorder=4)
        ax.text(0.538, y0 + 0.040, bd, fontsize=6.8, color=DARK_SLATE, ha="left", va="center", linespacing=1.35, zorder=4)
    draw_arrow(ax, (0.627, 0.630), (0.627, 0.589), color=AMBER, lw=1.4, ms=9)
    draw_arrow(ax, (0.627, 0.455), (0.627, 0.414), color=AMBER, lw=1.4, ms=9)

    draw_rbox(ax, 0.528, 0.08, 0.198, 0.165, fc="#FFFBEB", ec=AMBER_BD, lw=1.0, r=0.012, z=3)
    ax.text(0.627, 0.210, "高保真物理检查门", fontsize=7.9, fontweight="bold", color=AMBER, ha="center", va="center", zorder=4)
    ax.text(0.538, 0.135,
            "• 0.08 rad / 4 mm 细粒度步进检测\n"
            "• 几何不可达反馈上层重规划",
            fontsize=7.0, color=DARK_SLATE, ha="left", va="center", linespacing=1.45, zorder=4)

    # --- STEP 4 Content: TPG Asynchronous Parallelism ---
    ax.text(0.876, 0.800, "TPG 压缩等待，释放双臂并发", fontsize=8.4, fontweight="bold", color=NAVY, ha="center", va="center")
    # Mini Before vs After diagram
    draw_rbox(ax, 0.776, 0.565, 0.200, 0.195, fc="#F8FAFC", ec=BORDER, lw=1.0, r=0.012, z=3)
    ax.text(0.876, 0.735, "串行编译轨迹（冗余等待）", fontsize=7.3, fontweight="bold", color=MUTED, ha="center", zorder=4)
    ax.text(0.784, 0.680, "A:", fontsize=7.2, fontweight="bold", color="#15803D", va="center", zorder=4)
    draw_rbox(ax, 0.802, 0.658, 0.046, 0.042, fc=BLUE, ec=BLUE, lw=0.8, r=0.006, z=4)
    ax.text(0.825, 0.679, "Pick", fontsize=6.4, color="#FFF", ha="center", va="center", zorder=5)
    draw_rbox(ax, 0.852, 0.658, 0.048, 0.042, fc=AMBER, ec=AMBER, lw=0.8, r=0.006, z=4)
    ax.text(0.876, 0.679, "HO", fontsize=6.4, color="#FFF", ha="center", va="center", zorder=5)
    ax.plot([0.903, 0.966], [0.679, 0.679], color=PURPLE, lw=1.4, ls="--", zorder=4)

    ax.text(0.784, 0.608, "B:", fontsize=7.2, fontweight="bold", color=BLUE, va="center", zorder=4)
    ax.plot([0.802, 0.850], [0.608, 0.608], color=PURPLE, lw=1.4, ls="--", zorder=4)
    draw_rbox(ax, 0.852, 0.587, 0.048, 0.042, fc=AMBER, ec=AMBER, lw=0.8, r=0.006, z=4)
    ax.text(0.876, 0.608, "HO", fontsize=6.4, color="#FFF", ha="center", va="center", zorder=5)
    draw_rbox(ax, 0.904, 0.587, 0.062, 0.042, fc=CORAL, ec=CORAL, lw=0.8, r=0.006, z=4)
    ax.text(0.935, 0.608, "Insert", fontsize=6.4, color="#FFF", ha="center", va="center", zorder=5)

    draw_arrow(ax, (0.876, 0.562), (0.876, 0.515), color=PURPLE, lw=1.8, ms=11)

    draw_rbox(ax, 0.776, 0.295, 0.200, 0.215, fc=PURPLE_LT, ec=PURPLE, lw=1.1, r=0.012, z=3)
    ax.text(0.876, 0.485, "TPG 异步重排（自动并行）", fontsize=7.4, fontweight="bold", color=PURPLE, ha="center", zorder=4)
    ax.text(0.784, 0.425, "A:", fontsize=7.2, fontweight="bold", color="#15803D", va="center", zorder=4)
    draw_rbox(ax, 0.802, 0.404, 0.044, 0.042, fc=BLUE, ec=BLUE, lw=0.8, r=0.006, z=4)
    ax.text(0.824, 0.425, "Pick", fontsize=6.2, color="#FFF", ha="center", va="center", zorder=5)
    draw_rbox(ax, 0.850, 0.404, 0.042, 0.042, fc=AMBER, ec=AMBER, lw=0.8, r=0.006, z=4)
    ax.text(0.871, 0.425, "HO", fontsize=6.2, color="#FFF", ha="center", va="center", zorder=5)
    draw_rbox(ax, 0.896, 0.404, 0.066, 0.042, fc=TEAL, ec=TEAL, lw=0.8, r=0.006, z=4)
    ax.text(0.929, 0.425, "Next b2", fontsize=6.2, color="#FFF", ha="center", va="center", zorder=5)

    ax.text(0.784, 0.345, "B:", fontsize=7.2, fontweight="bold", color=BLUE, va="center", zorder=4)
    draw_rbox(ax, 0.802, 0.324, 0.044, 0.042, fc="#64748B", ec="#475569", lw=0.8, r=0.006, z=4)
    ax.text(0.824, 0.345, "Pre-Mv", fontsize=5.8, color="#FFF", ha="center", va="center", zorder=5)
    draw_rbox(ax, 0.850, 0.324, 0.042, 0.042, fc=AMBER, ec=AMBER, lw=0.8, r=0.006, z=4)
    ax.text(0.871, 0.345, "HO", fontsize=6.2, color="#FFF", ha="center", va="center", zorder=5)
    draw_rbox(ax, 0.896, 0.324, 0.066, 0.042, fc=CORAL, ec=CORAL, lw=0.8, r=0.006, z=4)
    ax.text(0.929, 0.345, "Insert", fontsize=6.2, color="#FFF", ha="center", va="center", zorder=5)
    # Motion dep arrow between HO and next
    draw_arrow(ax, (0.871, 0.402), (0.871, 0.368), color=NAVY, lw=1.2, ls="--", ms=7)
    draw_arrow(ax, (0.910, 0.368), (0.920, 0.402), color="#16A34A", lw=1.2, ls="--", rad=-0.2, ms=7)

    draw_rbox(ax, 0.776, 0.08, 0.200, 0.180, fc="#ECFDF5", ec="#6EE7B7", lw=1.0, r=0.012, z=3)
    ax.text(0.876, 0.225, "全链路优势总结", fontsize=7.9, fontweight="bold", color="#047857", ha="center", va="center", zorder=4)
    ax.text(0.786, 0.145,
            "• 边交接边预备，双臂并发推进\n"
            "• 保留全部因果与防碰安全边\n"
            "• 物理回放 0 碰撞直接落地",
            fontsize=7.0, color=DARK_SLATE, ha="left", va="center", linespacing=1.42, zorder=4)

    out = ASSET_DIR / "fig_ppt_walkthrough.png"
    fig.savefig(out, dpi=300, facecolor="#FFFFFF")
    plt.close(fig)
    print("Saved:", out)


def render_ab_tradeoff():
    """Figure 3: Quantitative Scene A vs Scene B Makespan-Parallelism Trade-off Chart for Slide 8."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 4.35), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel 1: Serial vs Parallel Makespan
    labels = ["Scene A\n(自由跨区 · 18步)", "Scene B\n(分区交接 · 22步)"]
    serial = [44.60, 77.81]
    parallel = [38.41, 62.25]
    x = np.arange(len(labels))
    w = 0.33

    b1 = ax1.bar(x - w / 2, serial, w, label="串行总时长", color="#CBD5E1", edgecolor="#64748B", linewidth=1.0, zorder=3)
    b2 = ax1.bar(x + w / 2, parallel, w, label="TPG 并行总时长", color=[BLUE, TEAL], edgecolor=[NAVY, "#115E59"], linewidth=1.0, zorder=3)

    ax1.set_title("① 总完工时长对比 (越低越快)", fontsize=10.5, fontweight="bold", color=NAVY, pad=12)
    ax1.set_ylabel("执行时长 (s)", fontsize=9.2, color=DARK_SLATE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8, fontweight="bold", color=DARK_SLATE)
    ax1.set_ylim(0, 96)
    ax1.grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, fontsize=8.2, loc="upper left")

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.2f}s", ha="center", va="bottom", fontsize=8.3, color=MUTED, fontweight="bold")
    for bar, spd in zip(b2, ["1.161×", "1.250×"]):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.2f}s\n({spd})", ha="center", va="bottom", fontsize=8.5, color=NAVY, fontweight="bold")

    # Callout on Panel 1 (cleanly clear of bar label)
    ax1.annotate("总时长更短\n(-38.3%)", xy=(0 + w / 2 + 0.06, 51.0), xytext=(0.42, 65.5),
                 ha="center", fontsize=8.0, fontweight="bold", color=BLUE,
                 bbox=dict(boxstyle="round,pad=0.32", fc=BLUE_LT, ec=BLUE_BD, lw=1.0),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.25))

    # Panel 2: Dual-Arm Parallelism (Overlap Duration & Ratio)
    overlap_pct = [16.1, 25.0]
    overlap_sec = [6.20, 15.56]
    bars = ax2.bar(x, overlap_pct, width=0.46, color=["#93C5FD", AMBER], edgecolor=[BLUE, "#B45309"], linewidth=1.2, zorder=3)

    ax2.set_title("② 双臂并行占比与并发时长 (越高并行越好)", fontsize=10.5, fontweight="bold", color=NAVY, pad=12)
    ax2.set_ylabel("双臂并发时间占比 (%)", fontsize=9.2, color=DARK_SLATE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.8, fontweight="bold", color=DARK_SLATE)
    ax2.set_ylim(0, 36.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for bar, pct, sec in zip(bars, overlap_pct, overlap_sec):
        ax2.text(bar.get_x() + bar.get_width() / 2, pct + 0.9,
                 f"{pct:.1f}%\n({sec:.2f} s)", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=NAVY)

    ax2.annotate("分区解除空间挤压\n并发时长 +151%!", xy=(0.78, 25.8), xytext=(0.24, 31.0),
                 ha="center", fontsize=8.0, fontweight="bold", color="#B45309",
                 bbox=dict(boxstyle="round,pad=0.32", fc=AMBER_LT, ec=AMBER_BD, lw=1.0),
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.3))

    fig.tight_layout(pad=1.6)
    out = ASSET_DIR / "fig_ppt_ab_tradeoff.png"
    fig.savefig(out, dpi=300, facecolor="#FFFFFF")
    plt.close(fig)
    print("Saved:", out)


def prepare_posters():
    """Generate clean, crisp poster images for cover and all embedded videos."""
    # Cover hero image (clean workcell + finished 9-block bridge mid-action)
    va = SHOWCASE / "mode_a/videos/mode_a_free_reach.mp4"
    vb = SHOWCASE / "mode_b/videos/mode_b_strict_zones_handover.mp4"
    vbc = SHOWCASE / "mode_b/videos/mode_b_handovers_closeup.mp4"

    raw_cover = grab_frame(vb, 57.5, ASSET_DIR / "_raw_cover.png")
    crop_save(raw_cover, (0.04, 0.11, 0.96, 0.98), ASSET_DIR / "poster_cover.png", (960, 620))

    raw_a = grab_frame(va, 35.5, ASSET_DIR / "_raw_mode_a.png")
    crop_save(raw_a, (0.02, 0.00, 0.98, 1.00), ASSET_DIR / "poster_mode_a.png", (960, 540))

    raw_b = grab_frame(vb, 25.5, ASSET_DIR / "_raw_mode_b.png")
    crop_save(raw_b, (0.02, 0.00, 0.98, 1.00), ASSET_DIR / "poster_mode_b.png", (960, 540))

    raw_bc = grab_frame(vbc, 1.3, ASSET_DIR / "_raw_mode_bc.png")
    crop_save(raw_bc, (0.00, 0.00, 1.00, 1.00), ASSET_DIR / "poster_mode_b_closeup.png", (960, 540))

    # F1-F4 clean posters (from their verified last/active frames, keeping full 1280x880 aspect)
    f_dir = ROOT / "out/review_videos_20261121"
    for key, fname, t_sec in [
        ("f1", "F1_014_parallel_place.mp4", 184.0),
        ("f2", "F2_014_handover.mp4", 72.0),
        ("f3", "F3_021_tower.mp4", 301.0),
        ("f4", "F4_033_cantilever_support.mp4", 195.0),
    ]:
        raw = grab_frame(f_dir / fname, t_sec, ASSET_DIR / f"_raw_{key}.png")
        # Crop top black band slightly so robots and structure fill the poster cleanly
        crop_save(raw, (0.04, 0.12, 0.96, 0.98), ASSET_DIR / f"poster_{key}.png", (800, 500))

    # Remove temp raw files
    for p in ASSET_DIR.glob("_raw_*.png"):
        p.unlink(missing_ok=True)
    print("Prepared all video poster frames in", ASSET_DIR)


if __name__ == "__main__":
    render_system_io()
    render_walkthrough()
    render_ab_tradeoff()
    prepare_posters()
