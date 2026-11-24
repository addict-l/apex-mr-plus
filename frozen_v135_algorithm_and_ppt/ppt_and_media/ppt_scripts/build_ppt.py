#!/usr/bin/env python3
"""Build a publication-grade 11-slide widescreen PPTX for Dual-Arm SkillNet + APEX-MR."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(r"D:\APEX_MR_Plus\dualarm_skillnet")
SHOWCASE = ROOT / "results/sim_v1_3/motion_v135/showcases/lego_bridge_9blocks"
PAPER_FIG = SHOWCASE / "paper_figures"
PRES_DIR = SHOWCASE / "presentation"
ASSET_DIR = PRES_DIR / "assets"
F_DIR = ROOT / "out/review_videos_20261121"

OUT_PPTX = PRES_DIR / "DualArm_SkillNet_APEX_MR_Presentation.pptx"

# Design Palette (Deep Academic Navy + Clean Light Surface + Semantic Accents)
C_NAVY = RGBColor(0x0F, 0x29, 0x42)
C_DARK_BG = RGBColor(0x0B, 0x19, 0x2C)
C_DARK_CARD = RGBColor(0x16, 0x2C, 0x46)
C_BG = RGBColor(0xF8, 0xFA, 0xFC)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_TEXT = RGBColor(0x1E, 0x29, 0x3B)
C_SUBTEXT = RGBColor(0x47, 0x55, 0x69)
C_MUTED = RGBColor(0x64, 0x74, 0x8B)
C_BORDER = RGBColor(0xCB, 0xD5, 0xE1)

C_BLUE = RGBColor(0x25, 0x63, 0xEB)
C_BLUE_LT = RGBColor(0xEF, 0xF6, 0xFF)
C_BLUE_BD = RGBColor(0x93, 0xC5, 0xFD)

C_TEAL = RGBColor(0x0D, 0x94, 0x88)
C_TEAL_LT = RGBColor(0xF0, 0xFD, 0xFA)
C_TEAL_BD = RGBColor(0x5E, 0xEA, 0xD4)

C_AMBER = RGBColor(0xD9, 0x77, 0x06)
C_AMBER_LT = RGBColor(0xFF, 0xFB, 0xEB)
C_AMBER_BD = RGBColor(0xFC, 0xD3, 0x4D)

C_CORAL = RGBColor(0xDC, 0x26, 0x26)
C_CORAL_LT = RGBColor(0xFE, 0xF2, 0xF2)
C_CORAL_BD = RGBColor(0xFC, 0xA5, 0xA5)

C_PURPLE = RGBColor(0x7C, 0x3A, 0xED)
C_PURPLE_LT = RGBColor(0xF5, 0xF3, 0xFF)
C_PURPLE_BD = RGBColor(0xC4, 0xB5, 0xFD)

C_CYAN = RGBColor(0x38, 0xBD, 0xF8)
C_GREEN_LT = RGBColor(0xEC, 0xFD, 0xF5)

FONT_CN = "Microsoft YaHei"
TOTAL_SLIDES = 11


def set_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, fc=C_WHITE, ec=None, lw=1.0, rounded=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = fc
    if ec is not None:
        sh.line.color.rgb = ec
        sh.line.width = Pt(lw)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def add_textbox(slide, x, y, w, h, text="", font_size=13, bold=False, color=C_TEXT,
                align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, line_spacing=1.22):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    if text:
        p = tf.paragraphs[0]
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = text
        r.font.name = FONT_CN
        r.font.size = Pt(font_size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def add_bullets(slide, x, y, w, h, items: list[tuple[str, str]], font_size=12.5,
                bullet_color=C_BLUE, text_color=C_TEXT, space_after=6, line_spacing=1.26):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for idx, (lead, body) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        r0 = p.add_run()
        r0.text = "▸ "
        r0.font.name = FONT_CN
        r0.font.size = Pt(font_size)
        r0.font.bold = True
        r0.font.color.rgb = bullet_color

        if lead:
            r1 = p.add_run()
            r1.text = lead
            r1.font.name = FONT_CN
            r1.font.size = Pt(font_size)
            r1.font.bold = True
            r1.font.color.rgb = C_NAVY

        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_CN
        r2.font.size = Pt(font_size)
        r2.font.bold = False
        r2.font.color.rgb = text_color
    return tb


def add_picture_fit(slide, img_path: Path, x, y, max_w, max_h):
    """Insert image preserving exact aspect ratio, centered inside (x, y, max_w, max_h)."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w = iw * scale
    h = ih * scale
    px = x + (max_w - w) * 0.5
    py = y + (max_h - h) * 0.5
    return slide.shapes.add_picture(str(img_path), Inches(px), Inches(py), width=Inches(w), height=Inches(h))


def add_movie_fit(slide, video_path: Path, poster_path: Path, x, y, w, h):
    """Embed playable MP4 video with crisp poster frame."""
    return slide.shapes.add_movie(
        str(video_path), Inches(x), Inches(y), Inches(w), Inches(h),
        poster_frame_image=str(poster_path), mime_type="video/mp4",
    )


def add_chrome(slide, kicker: str, title: str, slide_num: int, subtitle: str | None = None):
    """Consistent academic header + subtle bottom footer bar."""
    set_bg(slide, C_BG)
    # Top slim accent bar
    add_rect(slide, 0, 0, 13.333, 0.07, fc=C_NAVY)
    add_rect(slide, 0, 0, 3.2, 0.07, fc=C_BLUE)

    # Kicker pill
    add_textbox(slide, 0.55, 0.22, 8.0, 0.26, kicker.upper(), font_size=10, bold=True, color=C_BLUE)
    # Main title
    add_textbox(slide, 0.55, 0.44, 11.8, 0.48, title, font_size=23, bold=True, color=C_NAVY)
    if subtitle:
        add_textbox(slide, 0.55, 0.90, 12.1, 0.30, subtitle, font_size=11.5, color=C_MUTED)
        line_y = 1.24
    else:
        line_y = 0.98
    add_rect(slide, 0.55, line_y, 12.23, 0.015, fc=C_BORDER)

    # Bottom footer
    add_rect(slide, 0.55, 7.06, 12.23, 0.012, fc=C_BORDER)
    add_textbox(slide, 0.55, 7.12, 8.8, 0.28,
                "双臂协同装配任务与运动联合规划  |  Dual-Arm SkillNet + APEX-MR",
                font_size=9.5, color=C_MUTED)
    add_textbox(slide, 11.6, 7.12, 1.18, 0.28, f"{slide_num:02d} / {TOTAL_SLIDES:02d}",
                font_size=9.5, bold=True, color=C_NAVY, align=PP_ALIGN.RIGHT)


def add_kpi_pill(slide, x, y, w, h, label: str, value: str, accent=C_BLUE, bg=C_WHITE):
    add_rect(slide, x, y, w, h, fc=bg, ec=C_BORDER, lw=0.9, rounded=True)
    add_rect(slide, x, y + 0.10, 0.07, h - 0.20, fc=accent)
    add_textbox(slide, x + 0.16, y + 0.06, w - 0.22, 0.24, label, font_size=9.5, bold=True, color=C_MUTED)
    add_textbox(slide, x + 0.16, y + 0.30, w - 0.22, 0.42, value, font_size=14.5, bold=True, color=C_NAVY)


def add_notes(slide, text: str):
    slide.notes_slide.notes_text_frame.text = text


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: COVER (Dark Navy Executive Academic Cover)
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    set_bg(s1, C_DARK_BG)
    # Subtle geometric accents
    add_rect(s1, 0, 0, 0.22, 7.5, fc=C_BLUE)
    add_rect(s1, 0.22, 0, 0.06, 7.5, fc=C_CYAN)
    add_rect(s1, 7.85, 0, 5.483, 7.5, fc=RGBColor(0x11, 0x24, 0x3A))

    # Top tag pill
    add_rect(s1, 0.72, 1.05, 3.65, 0.38, fc=C_DARK_CARD, ec=C_CYAN, lw=1.0, rounded=True)
    add_textbox(s1, 0.85, 1.10, 3.45, 0.28, "组会专题汇报 · 第三次进展",
                font_size=11.5, bold=True, color=C_CYAN)

    # Main Title
    add_textbox(s1, 0.72, 1.72, 6.95, 1.85,
                "双臂协同装配任务与\n运动联合规划算法",
                font_size=38, bold=True, color=C_WHITE, line_spacing=1.18)

    # Subtitle
    add_textbox(s1, 0.72, 3.68, 6.75, 0.95,
                "基于参数化技能网络（SkillNet）与异步时序图（APEX-MR TPG）的\n终态驱动规划、多任务泛化及高保真物理闭环验证",
                font_size=14.5, color=RGBColor(0xCB, 0xD5, 0xE1), line_spacing=1.45)

    # Highlight metric badges on cover
    badges = [
        ("9 类原子技能", "三值能力表 + 反向搜索"),
        ("F1–F4 泛化 100%", "4/4 动力学 0 违规通过"),
        ("Scene A / B 实证", "9 块积木桥 · 0 碰撞闭环"),
    ]
    for i, (b_top, b_bot) in enumerate(badges):
        bx = 0.72 + i * 2.28
        add_rect(s1, bx, 5.10, 2.12, 0.95, fc=C_DARK_CARD, ec=RGBColor(0x33, 0x4E, 0x68), lw=1.0, rounded=True)
        add_rect(s1, bx, 5.10, 2.12, 0.06, fc=[C_CYAN, C_TEAL, C_AMBER][i])
        add_textbox(s1, bx + 0.12, 5.24, 1.90, 0.32, b_top, font_size=12.5, bold=True, color=C_WHITE)
        add_textbox(s1, bx + 0.12, 5.58, 1.90, 0.36, b_bot, font_size=9.5, color=RGBColor(0x94, 0xA3, 0xB8))

    add_textbox(s1, 0.72, 6.58, 6.8, 0.36,
                "汇报范围：算法原理 · 全链路实现 · F1–F4 泛化 · 双场景实战对比 · 创新与反思",
                font_size=10.5, color=RGBColor(0x94, 0xA3, 0xB8))

    # Right hero visual card (real CoppeliaSim 9-block LEGO bridge)
    add_rect(s1, 8.20, 1.15, 4.65, 5.20, fc=C_DARK_CARD, ec=RGBColor(0x38, 0xBD, 0xF8), lw=1.3, rounded=True)
    add_picture_fit(s1, ASSET_DIR / "poster_cover.png", 8.35, 1.32, 4.35, 3.45)
    add_textbox(s1, 8.38, 4.92, 4.30, 0.34,
                "CoppeliaSim 原生网格双臂装配验证平台",
                font_size=12.5, bold=True, color=C_WHITE)
    add_textbox(s1, 8.38, 5.30, 4.30, 0.88,
                "• 双 UR5e 协同完成 9 块积木桥全流程装配\n"
                "• 覆盖自由跨区并行（Scene A）与严格分区交接（Scene B）\n"
                "• 独立动力学全轨回放：0 碰撞 · 亚皮米级落位精度",
                font_size=10.0, color=RGBColor(0xCB, 0xD5, 0xE1), line_spacing=1.42)
    add_notes(s1, "开场说明：本次第三次汇报聚焦双臂协同装配任务与运动联合规划算法本身，涵盖算法原理与全链路拆解、F1-F4泛化验证、Scene A/B双场景对比实战以及深度反思；在线故障恢复部分将由队友在下一部分衔接汇报。")

    # =========================================================================
    # SLIDE 2: ROADMAP / AGENDA
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    add_chrome(s2, "00 · Presentation Roadmap", "汇报目录与核心主线", 2,
               "从“解决什么问题”出发，经由全链路机制拆解、多构型泛化与双模式实证，提炼核心创新与未来改进方向")

    agenda = [
        ("01", "算法介绍", "解决什么问题？\n输入（终态/技能库/三值表）与输出（TPG/无碰轨迹）", C_BLUE, "P03"),
        ("02", "全链路实现", "用一个极简跨区交接样例\n讲清“反向剪枝 → 事件 A* → 网格编译 → TPG 重排”", C_TEAL, "P04"),
        ("03", "泛化性验证", "F1–F4 四类标准构型动力学基准\n并行 / 交接 / 堆叠 / 悬臂托举 4/4 零违规（附视频）", C_AMBER, "P05"),
        ("04", "双场景实战对比", "9 块积木桥 Scene A（自由跨区）vs Scene B（分区交接）\nAPEX-MR 架构图 + 实拍视频 + 剪刀差定量剖析", C_PURPLE, "P06–P08"),
        ("05", "创新点与深度反思", "终态驱动无须预设顺序 vs 传统 ILP\n两大局限：三值表几何解耦缺陷 + 小模型先验压缩搜索空间", C_CORAL, "P09–P11"),
    ]
    for idx, (num, title, desc, col, pg) in enumerate(agenda):
        y0 = 1.48 + idx * 1.07
        add_rect(s2, 0.60, y0, 12.13, 0.92, fc=C_WHITE, ec=C_BORDER, lw=1.0, rounded=True)
        add_rect(s2, 0.60, y0, 0.95, 0.92, fc=col, rounded=True)
        add_textbox(s2, 0.60, y0 + 0.18, 0.95, 0.56, num, font_size=22, bold=True, color=C_WHITE,
                    align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        add_textbox(s2, 1.78, y0 + 0.12, 2.55, 0.68, title, font_size=16, bold=True, color=C_NAVY,
                    valign=MSO_ANCHOR.MIDDLE)
        add_rect(s2, 4.42, y0 + 0.18, 0.02, 0.56, fc=C_BORDER)
        add_textbox(s2, 4.65, y0 + 0.12, 6.75, 0.70, desc, font_size=12.5, color=C_SUBTEXT,
                    valign=MSO_ANCHOR.MIDDLE, line_spacing=1.32)
        add_rect(s2, 11.65, y0 + 0.28, 0.88, 0.36, fc=C_BG, ec=col, lw=1.0, rounded=True)
        add_textbox(s2, 11.65, y0 + 0.30, 0.88, 0.32, pg, font_size=10.5, bold=True, color=col,
                    align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    add_notes(s2, "快速过一遍汇报的五个板块，强调后面会结合真实物理仿真视频和数据展示一个非常有意思的发现。")

    # =========================================================================
    # SLIDE 3: ALGORITHM OVERVIEW — PROBLEM & INPUT/OUTPUT
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    add_chrome(s3, "01 · Algorithm Overview", "算法介绍：核心问题与系统输入输出", 3,
               "打通“终态驱动的符号任务规划”与“物理网格认证的异步运动并行执行”之间的断层")

    # Top two crisp problem/goal cards
    add_rect(s3, 0.55, 1.38, 5.98, 1.24, fc=C_CORAL_LT, ec=C_CORAL_BD, lw=1.0, rounded=True)
    add_rect(s3, 0.55, 1.38, 0.10, 1.24, fc=C_CORAL)
    add_textbox(s3, 0.78, 1.45, 5.60, 0.30, "核心痛点：我们要解决什么问题？", font_size=13.2, bold=True, color=C_CORAL)
    add_textbox(s3, 0.78, 1.78, 5.62, 0.78,
                "• 传统双臂装配调度（如经典 ILP）高度依赖人工给定安装顺序与固定模板；\n"
                "• 任务层符号规划与运动层几何避碰割裂，易产生不可达、碰撞或串行等待。",
                font_size=10.6, color=C_TEXT, line_spacing=1.35)

    add_rect(s3, 6.80, 1.38, 5.98, 1.24, fc=C_TEAL_LT, ec=C_TEAL_BD, lw=1.0, rounded=True)
    add_rect(s3, 6.80, 1.38, 0.10, 1.24, fc=C_TEAL)
    add_textbox(s3, 7.03, 1.45, 5.60, 0.30, "设计目标：本框架如何破局？", font_size=13.2, bold=True, color=C_TEAL)
    add_textbox(s3, 7.03, 1.78, 5.62, 0.78,
                "• 仅需目标终态结构 + 9 类参数化技能库 + 三值能力表，自动推理装配序与交接；\n"
                "• 经原生网格碰撞认证与 APEX-MR TPG 重排，直接输出零碰撞并行轨迹。",
                font_size=10.6, color=C_TEXT, line_spacing=1.35)

    # Main bespoke 300-DPI System I/O Architecture Diagram
    add_rect(s3, 0.55, 2.74, 12.23, 4.20, fc=C_WHITE, ec=C_BORDER, lw=1.0, rounded=True)
    add_picture_fit(s3, ASSET_DIR / "fig_ppt_system_io.png", 0.65, 2.80, 12.03, 4.08)
    add_notes(s3, "重点讲清三点输入（终态结构、9类技能库、三值能力表 VALID/INVALID/UNKNOWN）如何经过三层架构转化为零碰撞异步并行 TPG 输出。")

    # =========================================================================
    # SLIDE 4: ALGORITHM WALKTHROUGH — SIMPLE EXAMPLE END-TO-END
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    add_chrome(s4, "02 · End-to-End Implementation", "算法如何实现：以极简样例贯通全链路", 4,
               "以“左侧料区积木 b1 跨区装入右侧目标槽位 s1”为例，完整演示从符号倒推到物理异步并行的四个关键阶段")

    add_rect(s4, 0.55, 1.38, 12.23, 4.78, fc=C_WHITE, ec=C_BORDER, lw=1.0, rounded=True)
    add_picture_fit(s4, ASSET_DIR / "fig_ppt_walkthrough.png", 0.63, 1.44, 12.07, 4.66)

    # Bottom concise takeaway strip
    add_rect(s4, 0.55, 6.28, 12.23, 0.64, fc=C_NAVY, rounded=True)
    add_textbox(s4, 0.75, 6.36, 11.83, 0.48,
                "核心机制闭环：三值能力表决定“谁能做什么” → 反向相关性剪枝压缩动作集 → 可采纳事件 A* 保证符号时间最优 → 原生网格编译与 TPG 释放真实并行度。",
                font_size=11.8, bold=True, color=C_WHITE, valign=MSO_ANCHOR.MIDDLE)
    add_notes(s4, "沿着 STEP 1 到 STEP 4 从左向右讲：不需要人为指定先拿还是先交接，系统从目标 installed(b1,s1) 反向推理，看到 A 够不到 s1 自动剪枝，自然涌现出 pick(A)->handover(A->B)->insert(B)，再通过网格校验与 TPG 把等待压掉。")

    # =========================================================================
    # SLIDE 5: GENERALIZATION VALIDATION F1–F4 (2×2 Grid + 4 Embedded MP4s)
    # =========================================================================
    s5 = prs.slides.add_slide(blank)
    add_chrome(s5, "03 · Generalization Validation", "算法泛化性验证：F1–F4 四类典型装配构型（附实拍视频）", 5,
               "同一套 9 技能库与求解器零改动泛化至四类差异化任务；全部通过 CoppeliaSim 原生网格独立动力学验收（4/4 · 0 违规）")

    f_specs = [
        ("F1 · 多点并行插装", "F1_014_parallel_place",
         "双臂协同分散插装 4 块积木，检验多工位并发避碰",
         "313 节点 · 184.1 s · 0 违规", C_BLUE,
         F_DIR / "F1_014_parallel_place.mp4", ASSET_DIR / "poster_f1.png"),
        ("F2 · 跨区双臂交接", "F2_014_handover",
         "包含 2 次实测空中交接接收与释放，检验跨区传递协同",
         "297 节点 · 144.2 s · 0 违规", C_AMBER,
         F_DIR / "F2_014_handover.mp4", ASSET_DIR / "poster_f2.png"),
        ("F3 · 多层塔式垂直堆叠", "F3_021_tower",
         "3 层立柱严格先后序约束，检验长链因果依赖求解",
         "583 节点 · 301.7 s · 0 违规", C_TEAL,
         F_DIR / "F3_021_tower.mp4", ASSET_DIR / "poster_f3.png"),
        ("F4 · 悬臂主动协同托举", "F4_033_cantilever_support",
         "一臂持续托举悬臂梁 + 另一臂装压重块，检验载荷稳定闭环",
         "524 节点 · 243.1 s · 0 违规", C_PURPLE,
         F_DIR / "F4_033_cantilever_support.mp4", ASSET_DIR / "poster_f4.png"),
    ]

    for idx, (title, code, desc, metric, col, vpath, ppath) in enumerate(f_specs):
        row, col_i = divmod(idx, 2)
        cx = 0.55 + col_i * 6.22
        cy = 1.38 + row * 2.82
        cw, ch = 6.00, 2.66

        add_rect(s5, cx, cy, cw, ch, fc=C_WHITE, ec=C_BORDER, lw=1.0, rounded=True)
        add_rect(s5, cx, cy, 0.09, ch, fc=col)

        # Left side of card: Embedded Video (16:10 clean aspect = 3.55 x 2.22)
        add_movie_fit(s5, vpath, ppath, cx + 0.18, cy + 0.22, 3.55, 2.22)

        # Right side of card: Info & Verified Badge
        tx = cx + 3.84
        tw = cw - 3.96
        add_textbox(s5, tx, cy + 0.18, tw, 0.34, title, font_size=13.0, bold=True, color=C_NAVY)
        add_textbox(s5, tx, cy + 0.52, tw, 0.24, code, font_size=9.2, bold=True, color=col)
        add_textbox(s5, tx, cy + 0.84, tw, 0.86, desc, font_size=10.3, color=C_SUBTEXT, line_spacing=1.35)

        add_rect(s5, tx, cy + 1.80, tw, 0.64, fc=C_GREEN_LT, ec=C_TEAL_BD, lw=0.9, rounded=True)
        add_textbox(s5, tx + 0.08, cy + 1.86, tw - 0.16, 0.24, "✓ 独立动力学验收 PASS", font_size=9.2, bold=True, color=C_TEAL)
        add_textbox(s5, tx + 0.08, cy + 2.12, tw - 0.16, 0.26, metric, font_size=9.2, bold=True, color=C_NAVY)

    add_notes(s5, "展示 F1 到 F4 四个泛化任务：同一套9技能库和算法内核无需改代码即可处理并行插装、双臂交接、多层高塔和需要主动受力托举的悬臂结构，全部 0 违规通过物理回放。放映时可点击任一视频直接播放。")

    # =========================================================================
    # SLIDE 6: SCENE A — FREE-REACH PARALLEL (APEX-MR Diagram + Video A)
    # =========================================================================
    s6 = prs.slides.add_slide(blank)
    add_chrome(s6, "04 · Dual-Scenario Practice · Scene A", "实战验证（一）：Scene A 自由跨区并行基线", 6,
               "9 块积木桥全流程装配 · 双臂可达全桌面，自由跨区取放并通过 TPG 自动调度通道让行")

    # Left: Paper-style APEX-MR Architecture Figure (wide aspect 16.2:6.4)
    add_rect(s6, 0.55, 1.38, 7.62, 4.46, fc=C_WHITE, ec=C_BORDER, lw=1.0, rounded=True)
    add_rect(s6, 0.55, 1.38, 7.62, 0.36, fc=C_BLUE_LT, ec=C_BLUE_BD, lw=0.8)
    add_textbox(s6, 0.70, 1.42, 7.30, 0.28,
                "APEX-MR 架构与 TPG 异步调度图（Mode A · 18 步自由跨区）",
                font_size=11.0, bold=True, color=C_BLUE)
    add_picture_fit(s6, PAPER_FIG / "fig_apex_mr_mode_a.png", 0.64, 1.80, 7.44, 3.96)

    # Right: Embedded Video A + Key Behavioral Notes
    add_rect(s6, 8.34, 1.38, 4.44, 4.46, fc=C_WHITE, ec=C_BORDER, lw=1.0, rounded=True)
    add_rect(s6, 8.34, 1.38, 4.44, 0.36, fc=C_NAVY)
    add_textbox(s6, 8.48, 1.42, 4.18, 0.28, "▶ Video A · 自由跨区实拍（点击播放）",
                font_size=11.0, bold=True, color=C_WHITE)
    add_movie_fit(s6, SHOWCASE / "mode_a/videos/mode_a_free_reach.mp4",
                  ASSET_DIR / "poster_mode_a.png", 8.48, 1.84, 4.16, 2.34)
    add_bullets(s6, 8.48, 4.28, 4.16, 1.46, [
        ("最短路径策略：", "双臂直接跨区抓取并插装，无需空中交接，动作步数仅 18 步。"),
        ("核心阻塞特征：", "跨区经过公共走廊时触发 TPG 通道让行（Corridor Yield），另一臂外侧停靠等待。"),
    ], font_size=10.5, bullet_color=C_BLUE, space_after=5, line_spacing=1.25)

    # Bottom 4 KPI Pills
    add_kpi_pill(s6, 0.55, 6.00, 2.90, 0.88, "装配规模 / 交接次数", "18 步 · 0 次交接", C_BLUE)
    add_kpi_pill(s6, 3.66, 6.00, 2.90, 0.88, "串行 → TPG 并行时长", "44.60 s → 38.41 s", C_TEAL)
    add_kpi_pill(s6, 6.77, 6.00, 2.90, 0.88, "TPG 加速比 / 双臂并发", "1.161× · 6.20 s (16.1%)", C_AMBER)
    add_kpi_pill(s6, 9.88, 6.00, 2.90, 0.88, "4,788 帧独立回放校验", "0 碰撞 · 1.13×10⁻¹³ m", C_PURPLE)
    add_notes(s6, "讲解 Scene A：左侧展示完整的 APEX-MR 三层架构图（输入->任务图->运动规划->TPG），右侧直接播放高清仿真视频。重点提醒老师注意：虽然总时间只有 38.41s，但双臂并发占比只有 16.1%。")

    # =========================================================================
    # SLIDE 7: SCENE B — STRICT-ZONE + 4× HANDOVER (APEX-MR + Video B + Closeup)
    # =========================================================================
    s7 = prs.slides.add_slide(blank)
    add_chrome(s7, "04 · Dual-Scenario Practice · Scene B", "实战验证（二）：Scene B 严格分区与 4 次空中交接", 7,
               "双臂严格限定各自工作区，跨区积木自动涌现 4 次双向空中交接（A→B ×2，B→A ×2）与 180° 镜像避障")

    # Left: Paper-style APEX-MR Architecture Figure for Mode B
    add_rect(s7, 0.55, 1.38, 7.62, 4.46, fc=C_WHITE, ec=C_BORDER, lw=1.0, rounded=True)
    add_rect(s7, 0.55, 1.38, 7.62, 0.36, fc=C_AMBER_LT, ec=C_AMBER_BD, lw=0.8)
    add_textbox(s7, 0.70, 1.42, 7.30, 0.28,
                "APEX-MR 架构与 TPG 异步调度图（Mode B · 22 步含 4 次双向交接）",
                font_size=11.0, bold=True, color=C_AMBER)
    add_picture_fit(s7, PAPER_FIG / "fig_apex_mr_mode_b.png", 0.64, 1.80, 7.44, 3.96)

    # Right Top: Video B Overview + Right Bottom: 15s Handover Closeup!
    add_rect(s7, 8.34, 1.38, 4.44, 4.46, fc=C_WHITE, ec=C_BORDER, lw=1.0, rounded=True)
    add_rect(s7, 8.34, 1.38, 4.44, 0.34, fc=C_NAVY)
    add_textbox(s7, 8.48, 1.41, 4.18, 0.26, "▶ Video B · 分区交接全景 + 15s 交接特写",
                font_size=10.8, bold=True, color=C_WHITE)

    # Overview video (top)
    add_movie_fit(s7, SHOWCASE / "mode_b/videos/mode_b_strict_zones_handover.mp4",
                  ASSET_DIR / "poster_mode_b.png", 8.50, 1.80, 3.30, 1.86)
    add_textbox(s7, 11.86, 1.86, 0.86, 1.72, "全景实拍\n60 s\n\n双臂分区\n并行作业", font_size=9.2, bold=True, color=C_NAVY, line_spacing=1.25)

    # Divider
    add_rect(s7, 8.50, 3.74, 4.12, 0.012, fc=C_BORDER)

    # Handover closeup video (bottom)
    add_movie_fit(s7, SHOWCASE / "mode_b/videos/mode_b_handovers_closeup.mp4",
                  ASSET_DIR / "poster_mode_b_closeup.png", 8.50, 3.84, 3.30, 1.86)
    add_textbox(s7, 11.86, 3.88, 0.86, 1.76, "交接特写\n15 s\n\nA→B 橙砖\nB→A 绿砖\n180°避障", font_size=9.2, bold=True, color=C_AMBER, line_spacing=1.25)

    # Bottom 4 KPI Pills
    add_kpi_pill(s7, 0.55, 6.00, 2.90, 0.88, "装配规模 / 双向交接", "22 步 · 4/4 空中交接", C_AMBER)
    add_kpi_pill(s7, 3.66, 6.00, 2.90, 0.88, "串行 → TPG 并行时长", "77.81 s → 62.25 s", C_TEAL)
    add_kpi_pill(s7, 6.77, 6.00, 2.90, 0.88, "TPG 加速比 / 双臂并发", "1.250× · 15.56 s (25.0%)", C_BLUE)
    add_kpi_pill(s7, 9.88, 6.00, 2.90, 0.88, "8,155 帧独立回放校验", "0 碰撞 · 3.31×10⁻¹³ m", C_PURPLE)
    add_notes(s7, "讲解 Scene B：在严格分区约束下，算法自动推理出 4 次双向空中交接并配对 180°镜像避障。右侧上面是全景视频，下面是 15 秒高清交接特写，可以专门点开让老师看清空中精准传递。")

    # =========================================================================
    # SLIDE 8: SCENE A vs SCENE B QUANTITATIVE COMPARISON & KEY DISCOVERY
    # =========================================================================
    s8 = prs.slides.add_slide(blank)
    add_chrome(s8, "04 · Quantitative Analysis & Insight", "Scene A vs Scene B 定量对比：一个反直觉的关键现象", 8,
               "自由跨区总时长更短，但双臂并发占比更低；引入分区交接后总时长增加，却使双臂并行时间暴增 151%")

    # Left: Bespoke 300-DPI Trade-off Chart
    add_rect(s8, 0.55, 1.38, 6.65, 5.48, fc=C_WHITE, ec=C_BORDER, lw=1.0, rounded=True)
    add_picture_fit(s8, ASSET_DIR / "fig_ppt_ab_tradeoff.png", 0.65, 1.48, 6.45, 4.35)
    add_rect(s8, 0.72, 5.92, 6.31, 0.78, fc=C_BG, ec=C_BORDER, lw=0.8, rounded=True)
    add_textbox(s8, 0.86, 6.00, 6.05, 0.62,
                "核心剪刀差：Scene A 完工快 38.3%（38.41s vs 62.25s），但 Scene B 的双臂并行时长达到 Scene A 的 2.51 倍（15.56s vs 6.20s），TPG 加速比更高（1.250× vs 1.161×）。",
                font_size=10.5, bold=True, color=C_NAVY, line_spacing=1.30)

    # Right Top: Clean Native Comparison Table
    tbl_shape = s8.shapes.add_table(6, 3, Inches(7.38), Inches(1.38), Inches(5.40), Inches(2.58))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(1.96)
    tbl.columns[1].width = Inches(1.72)
    tbl.columns[2].width = Inches(1.72)

    headers = ["核心指标", "Scene A (跨区)", "Scene B (交接)"]
    rows_data = [
        ["任务步数 / 交接", "18 步 / 0 次", "22 步 / 4 次"],
        ["串行 → TPG 并行", "44.60 → 38.41 s", "77.81 → 62.25 s"],
        ["TPG 并行加速比", "1.161×", "1.250×  ▲"],
        ["双臂并行时间 (占比)", "6.20 s (16.1%)", "15.56 s (25.0%) ▲"],
        ["独立回放碰撞 / 误差", "0 碰 / 1.13e-13 m", "0 碰 / 3.31e-13 m"],
    ]
    for c_i, h_txt in enumerate(headers):
        cell = tbl.cell(0, c_i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if c_i > 0 else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = h_txt
        r.font.name = FONT_CN
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.color.rgb = C_WHITE

    for r_i, row_vals in enumerate(rows_data, start=1):
        bg_c = C_WHITE if r_i % 2 == 1 else C_BG
        for c_i, val in enumerate(row_vals):
            cell = tbl.cell(r_i, c_i)
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_AMBER_LT if (r_i in (3, 4) and c_i == 2) else bg_c
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_i > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.name = FONT_CN
            r.font.size = Pt(10.2)
            r.font.bold = (c_i == 0) or (r_i in (2, 3, 4) and c_i > 0)
            r.font.color.rgb = C_AMBER if (r_i in (3, 4) and c_i == 2) else (C_BLUE if (r_i == 2 and c_i == 1) else C_TEXT)

    # Right Bottom: Deep Mechanism Explanation Card
    add_rect(s8, 7.38, 4.14, 5.40, 2.72, fc=C_AMBER_LT, ec=C_AMBER_BD, lw=1.2, rounded=True)
    add_rect(s8, 7.38, 4.14, 5.40, 0.42, fc=C_AMBER)
    add_textbox(s8, 7.54, 4.20, 5.10, 0.30, "现象机理剖析：空间挤压 vs 并行释放", font_size=12.5, bold=True, color=C_WHITE)
    add_bullets(s8, 7.52, 4.66, 5.12, 2.10, [
        ("Scene A（跨区挤压）：", "双臂跨大区域移动时严重侵占另一臂工作空间，TPG 被迫插入大量通道让行依赖，压缩了有效并行时间（仅 6.20 s）。"),
        ("Scene B（分区解耦）：", "严格分区虽带来 4 次空中交接的动作开销（总时长增至 62.25 s），但显著消除空间争抢，双臂并发时长跃升至 15.56 s。"),
        ("对规划算法的深层启示：", "这一矛盾直接暴露出上层三值能力表规划的局限（详见 P10）。"),
    ], font_size=10.8, bullet_color=C_AMBER, space_after=6, line_spacing=1.28)
    add_notes(s8, "这一页是整场汇报的高潮之一！用左图和右表直观指出：Scene A 虽然总时间更短，但由于双臂跨大区域移动会大幅挤压另一臂工作空间，导致有效并行时间更少；加入分区交接后虽然总时长变长，但双臂并行时间大幅提升。这自然过渡到我们对算法创新点与不足的剖析。")

    # =========================================================================
    # SLIDE 9: CORE INNOVATIONS (Especially End-State Driven vs Traditional ILP)
    # =========================================================================
    s9 = prs.slides.add_slide(blank)
    add_chrome(s9, "05 · Core Innovations", "算法核心创新点", 9,
               "以“终态结构驱动的符号自动搜索”替代人工序约束，并与高保真原生网格物理闭环深度融合")

    # Innovation Card 1 (Main Highlight: End-State Driven vs ILP)
    add_rect(s9, 0.55, 1.42, 6.00, 5.42, fc=C_WHITE, ec=C_BLUE, lw=1.4, rounded=True)
    add_rect(s9, 0.55, 1.42, 6.00, 0.62, fc=C_BLUE)
    add_textbox(s9, 0.78, 1.54, 5.56, 0.38, "创新点 ① · 终态驱动规划，摆脱人工预置安装顺序",
                font_size=14.5, bold=True, color=C_WHITE)

    # Mini Contrast Box inside Card 1
    add_rect(s9, 0.78, 2.22, 2.66, 1.52, fc=C_CORAL_LT, ec=C_CORAL_BD, lw=1.0, rounded=True)
    add_textbox(s9, 0.90, 2.30, 2.42, 0.28, "传统 ILP / 调度框架", font_size=11.5, bold=True, color=C_CORAL)
    add_textbox(s9, 0.90, 2.62, 2.42, 1.04,
                "• 必须预先给定合法安装顺序\n"
                "• 交接/托举节点需人工模板化\n"
                "• 换构型即需重写约束与工序",
                font_size=10.5, color=C_TEXT, line_spacing=1.35)

    add_rect(s9, 3.66, 2.22, 2.66, 1.52, fc=C_TEAL_LT, ec=C_TEAL_BD, lw=1.0, rounded=True)
    add_textbox(s9, 3.78, 2.30, 2.42, 0.28, "本方法（SkillNet + A*）", font_size=11.5, bold=True, color=C_TEAL)
    add_textbox(s9, 3.78, 2.62, 2.42, 1.04,
                "• 仅需输入目标终态结构\n"
                "• 自动涌现顺序、分工与交接\n"
                "• 9 类技能零改动泛化 F1–F4",
                font_size=10.5, color=C_TEXT, line_spacing=1.35)

    add_bullets(s9, 0.78, 3.94, 5.54, 2.72, [
        ("可采纳启发保证时间最优：", "融合 HMax 事实传播、工作量抽象图与事件抽象图三类下界（219 样本检验 0 高估），事件状态 A* 首次弹出即最优解。"),
        ("自动涌现复杂协同原语：", "无需人工提示，即可自动求解跨区交接（F2 / Scene B）、多层先后序（F3）及悬臂主动托举与释放（F4）。"),
        ("全链路标准库轻量实现：", "符号搜索层完全独立于仿真与大模型，单测 213 项全通，求解过程 100% 可解释、可复现。"),
    ], font_size=10.8, bullet_color=C_BLUE, space_after=8, line_spacing=1.30)

    # Innovation Card 2 (Three-Valued Capability + Native Mesh Closed Loop)
    add_rect(s9, 6.78, 1.42, 6.00, 5.42, fc=C_WHITE, ec=C_TEAL, lw=1.4, rounded=True)
    add_rect(s9, 6.78, 1.42, 6.00, 0.62, fc=C_TEAL)
    add_textbox(s9, 7.01, 1.54, 5.56, 0.38, "创新点 ② · 三值能力表与高保真网格物理闭环",
                font_size=14.5, bold=True, color=C_WHITE)

    add_bullets(s9, 7.01, 2.26, 5.54, 3.15, [
        ("显式三值能力建模（VALID / INVALID / UNKNOWN）：", "统一表达已知可达、禁止跨区与待验证边界，天然支持从自由跨区（Scene A）到严格分区交接（Scene B）的策略无缝切换。"),
        ("面向交接的配对 IK 与 180° 镜像避障：", "在底层运动编译中针对双臂对向交接姿态引入配对 IK 分支与 180° 镜像防碰机制，攻克狭窄中心区双向交接碰撞难题。"),
        ("严格原生网格动力学零碰撞验证：", "经 0.08 rad / 4 mm 细粒度检碰与 TPG 异步重排，所有场景独立回放均达 0 未授权碰撞、落位误差 < 10⁻¹² m。"),
    ], font_size=11.0, bullet_color=C_TEAL, space_after=10, line_spacing=1.34)

    # Bottom verification banner inside Card 2
    add_rect(s9, 7.01, 5.62, 5.54, 0.98, fc=C_GREEN_LT, ec=C_TEAL_BD, lw=1.0, rounded=True)
    add_textbox(s9, 7.18, 5.70, 5.20, 0.28, "已落地的实证规模", font_size=11.2, bold=True, color=C_TEAL)
    add_textbox(s9, 7.18, 6.00, 5.20, 0.52,
                "• 基准套件：11/11 标准任务最优求解 + F1–F4 4/4 动力学 0 违规\n"
                "• 复杂实战：9 块积木桥双模式累计 12,943 帧独立回放 0 碰撞",
                font_size=10.5, bold=True, color=C_NAVY, line_spacing=1.35)
    add_notes(s9, "重点强调创新点一：传统 ILP 需要给定装配顺序，而本算法只需要给出目标终态结构即可联合求解顺序、分工和交接；创新点二强调三值能力表到原生网格 0 碰撞执行的完整闭环。")

    # =========================================================================
    # SLIDE 10: LIMITATIONS & FUTURE IMPROVEMENTS (Direct from User's 2 Deep Insights!)
    # =========================================================================
    s10 = prs.slides.add_slide(blank)
    add_chrome(s10, "05 · Limitations & Future Work", "深度反思：当前算法不足与后续改进方向", 10,
               "直面实测中暴露的两类核心瓶颈——“上层三值表全局性缺陷”与“全量符号展开的搜索扩展性瓶颈”")

    # Limitation 1: Three-Valued Table / Global Optimality Gap (from Scene A vs B discovery)
    add_rect(s10, 0.55, 1.42, 6.00, 5.42, fc=C_WHITE, ec=C_CORAL, lw=1.4, rounded=True)
    add_rect(s10, 0.55, 1.42, 6.00, 0.62, fc=C_CORAL)
    add_textbox(s10, 0.78, 1.54, 5.56, 0.38, "不足 ① · 三值能力表规划缺乏全局几何耦合视角",
                font_size=14.2, bold=True, color=C_WHITE)

    add_rect(s10, 0.78, 2.22, 5.54, 2.02, fc=C_CORAL_LT, ec=C_CORAL_BD, lw=1.0, rounded=True)
    add_textbox(s10, 0.94, 2.30, 5.22, 0.30, "问题本质（由 Scene A vs B 实测揭示）", font_size=11.8, bold=True, color=C_CORAL)
    add_bullets(s10, 0.94, 2.64, 5.22, 1.52, [
        ("静态三值表盲区：", "当前上层仅依据三值可达性与静态欧氏耗时规划，未感知双臂跨区对工作空间的动态挤压及 TPG 通道让行代价。"),
        ("难以实现全局时空权衡：", "自由跨区虽步数少但并行度低（16.1%）；交接虽大幅提高并行度（25.0%），却因静态动作开销增加而难以被自动择优。"),
    ], font_size=10.6, bullet_color=C_CORAL, space_after=6, line_spacing=1.28)

    add_rect(s10, 0.78, 4.44, 5.54, 2.18, fc=C_BLUE_LT, ec=C_BLUE_BD, lw=1.0, rounded=True)
    add_textbox(s10, 0.94, 4.52, 5.22, 0.30, "改进方向：几何感知的时空代价闭环反馈", font_size=11.8, bold=True, color=C_BLUE)
    add_bullets(s10, 0.94, 4.86, 5.22, 1.66, [
        ("引入空间竞争惩罚项：", "将下游 TPG 统计的走廊碰撞频率、等待阻塞时长反馈回上层任务规划器代价函数。"),
        ("混合交接自适应决策：", "允许规划器在同一任务中局部保留高效跨区、高冲突段自动插入交接，逼近真正的全局时空最优。"),
    ], font_size=10.6, bullet_color=C_BLUE, space_after=6, line_spacing=1.28)

    # Limitation 2: Exhaustive Symbolic Grounding -> Small Model Prior Pruning
    add_rect(s10, 6.78, 1.42, 6.00, 5.42, fc=C_WHITE, ec=C_PURPLE, lw=1.4, rounded=True)
    add_rect(s10, 6.78, 1.42, 6.00, 0.62, fc=C_PURPLE)
    add_textbox(s10, 7.01, 1.54, 5.56, 0.38, "不足 ② · 全量符号填入再剪枝，大场景扩展性受限",
                font_size=14.2, bold=True, color=C_WHITE)

    add_rect(s10, 7.01, 2.22, 5.54, 2.02, fc=C_PURPLE_LT, ec=C_PURPLE_BD, lw=1.0, rounded=True)
    add_textbox(s10, 7.17, 2.30, 5.22, 0.30, "问题本质（符号组合爆炸瓶颈）", font_size=11.8, bold=True, color=C_PURPLE)
    add_bullets(s10, 7.17, 2.64, 5.22, 1.52, [
        ("先全量实例化再剪枝：", "目前需先将所有零件、槽位与技能的参数组合全量填入，再通过反向相关性与能力表逐项剪枝。"),
        ("扩展性压力：", "当积木规模从 9 块扩展至数十块复杂结构时，初始符号展开与抽象图构建开销将急剧攀升。"),
    ], font_size=10.6, bullet_color=C_PURPLE, space_after=6, line_spacing=1.28)

    add_rect(s10, 7.01, 4.44, 5.54, 2.18, fc=C_TEAL_LT, ec=C_TEAL_BD, lw=1.0, rounded=True)
    add_textbox(s10, 7.17, 4.52, 5.22, 0.30, "改进方向：轻量化小模型先验 + 反向符号精搜", font_size=11.8, bold=True, color=C_TEAL)
    add_bullets(s10, 7.17, 4.86, 5.22, 1.66, [
        ("小模型先验收束搜索空间：", "前置接入轻量化小模型（如本地 7B / 轻量化图先验网络），优先输出高置信度的紧凑候选动作与子目标集合。"),
        ("神经先验与符号完备性互补：", "在收束后的子空间内执行精确反向搜索与 A* 验证，若先验失效则安全回退全量搜索，兼顾速度与可靠性。"),
    ], font_size=10.6, bullet_color=C_TEAL, space_after=6, line_spacing=1.28)
    add_notes(s10, "把两位核心思考讲透：第一，三值能力表未建模下层几何空间竞争，无法自适应权衡跨区短时长与交接高并行；第二，目前是先全量符号填入再剪枝，下一步计划引入轻量化小模型先收束出合理搜索空间，再交由反向符号搜索精确求解。")

    # =========================================================================
    # SLIDE 11: SUMMARY & HANDOFF TO RECOVERY PRESENTATION (Dark Navy Closing)
    # =========================================================================
    s11 = prs.slides.add_slide(blank)
    set_bg(s11, C_DARK_BG)
    add_rect(s11, 0, 0, 13.333, 0.09, fc=C_BLUE)
    add_rect(s11, 0, 7.41, 13.333, 0.09, fc=C_CYAN)

    add_textbox(s11, 0.75, 0.65, 6.0, 0.32, "06 · SUMMARY & NEXT SECTION",
                font_size=11, bold=True, color=C_CYAN)
    add_textbox(s11, 0.75, 1.02, 11.8, 0.60, "工作总结与下阶段衔接",
                font_size=30, bold=True, color=C_WHITE)

    summary_cards = [
        ("01 · 理论与链路闭环", C_BLUE,
         "打通终态结构 → 三值能力表 → 反向剪枝 + 可采纳 A* → 原生网格编译 → TPG 异步执行全链路，摆脱人工预置安装序。"),
        ("02 · 多任务与双模式实证", C_TEAL,
         "F1–F4 四类基准 100% 零违规通过；9 块积木桥 Scene A（38.41 s）与 Scene B（4/4 交接，并行 25.0%）均实现零碰撞物理闭环。"),
        ("03 · 瓶颈洞察与演进路径", C_AMBER,
         "定量揭示跨区空间挤压与交接并行增益的权衡规律，明确“TPG 空间代价反馈 + 小模型先验剪枝”两大升级方向。"),
    ]
    for i, (st, sc, sb) in enumerate(summary_cards):
        cx = 0.75 + i * 4.02
        add_rect(s11, cx, 1.98, 3.78, 2.55, fc=C_DARK_CARD, ec=RGBColor(0x33, 0x4E, 0x68), lw=1.0, rounded=True)
        add_rect(s11, cx, 1.98, 3.78, 0.08, fc=sc)
        add_textbox(s11, cx + 0.22, 2.22, 3.34, 0.38, st, font_size=14.5, bold=True, color=C_WHITE)
        add_textbox(s11, cx + 0.22, 2.72, 3.34, 1.62, sb, font_size=11.2, color=RGBColor(0xCB, 0xD5, 0xE1), line_spacing=1.42)

    # Seamless Handoff Banner to Friend's Recovery Talk
    add_rect(s11, 0.75, 4.92, 11.83, 1.32, fc=RGBColor(0x13, 0x31, 0x52), ec=C_CYAN, lw=1.2, rounded=True)
    add_textbox(s11, 1.02, 5.06, 11.30, 0.34,
                "下一步衔接：从静态规划闭环迈向动态异常恢复  →",
                font_size=14.0, bold=True, color=C_CYAN)
    add_textbox(s11, 1.02, 5.46, 11.30, 0.66,
                "以上工作验证了静态确定环境下的双臂任务-运动联合规划闭环；当执行过程中突发单臂动作受阻或跨区能力退化时，\n"
                "系统如何利用备份计划与搜索图复用实现在线快速恢复，将由下一部分队友重点汇报。",
                font_size=11.8, color=C_WHITE, line_spacing=1.38)

    add_textbox(s11, 0.75, 6.56, 11.83, 0.45,
                "感谢各位老师聆听 · 欢迎批评指正！",
                font_size=18, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    add_notes(s11, "总结三点成果，并自然把话筒交给负责在线恢复算法部分的队友。")

    prs.save(str(OUT_PPTX))
    print(f"Saved PPTX: {OUT_PPTX} ({OUT_PPTX.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    build_presentation()
