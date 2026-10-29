"""Rebuild embedded text-board PNGs for the APEX MR Plus workcell."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def fit_font(text: str, maximum: int, available: int, bold: bool = True):
    path = FONT_BOLD if bold else FONT_REGULAR
    for size in range(maximum, 19, -1):
        font = ImageFont.truetype(str(path), size)
        if font.getlength(text) <= available:
            return font
    raise ValueError(f"Text cannot fit: {text}")


def text_center(draw, size, y, text, fill, maximum, bold=True):
    font = fit_font(text, maximum, size[0] - 144, bold)
    bounds = draw.textbbox((0, 0), text, font=font)
    x = (size[0] - (bounds[2] - bounds[0])) / 2 - bounds[0]
    draw.text((x, y - bounds[1]), text, font=font, fill=fill)


def board(name, text, background, foreground=(255, 255, 255)):
    size = (1024, 256)
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    font = fit_font(text, 90, size[0] - 144)
    b = draw.textbbox((0, 0), text, font=font)
    x = (size[0] - b[2] + b[0]) / 2 - b[0]
    y = (size[1] - b[3] + b[1]) / 2 - b[1]
    draw.text((x, y), text, font=font, fill=foreground)
    image.save(ASSETS / name, optimize=True)
    return {"file": name, "size": list(size), "text": [text]}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = []
    title = Image.new("RGB", (2048, 384), (20, 39, 66))
    draw = ImageDraw.Draw(title)
    text_center(draw, title.size, 58, "APEX MR PLUS", (250, 252, 255), 132)
    text_center(draw, title.size, 226, "双臂积木协作 · 能力变化与恢复", (190, 220, 244), 76)
    title.save(ASSETS / "title.png", optimize=True)
    manifest.append({"file": "title.png", "size": list(title.size), "text": ["APEX MR PLUS", "双臂积木协作 · 能力变化与恢复"]})

    labels = [
        ("label_A.png", "A 臂 · 供料 / 安装", (224, 111, 32), (255, 255, 255)),
        ("label_B.png", "B 臂 · 安装 / 支撑", (34, 112, 194), (255, 255, 255)),
        ("label_buffer.png", "共享暂存区", (112, 79, 173), (255, 255, 255)),
        ("label_supply_A.png", "A 侧料盘", (49, 62, 79), (255, 255, 255)),
        ("label_supply_B.png", "B 侧料盘", (49, 62, 79), (255, 255, 255)),
        ("label_prop.png", "临时桥墩", (247, 194, 48), (44, 48, 54)),
        ("label_goal.png", "目标结构 · 城堡连桥", (35, 135, 101), (255, 255, 255)),
        ("label_fault.png", "A 臂能力受限", (196, 54, 62), (255, 255, 255)),
    ]
    for args in labels:
        manifest.append(board(*args))

    legend = Image.new("RGB", (2048, 384), (242, 246, 250))
    draw = ImageDraw.Draw(legend)
    text_center(draw, legend.size, 74, "橙色 A 臂 ｜ 蓝色 B 臂 ｜ 黄色临时桥墩", (27, 45, 66), 82)
    text_center(draw, legend.size, 218, "环境预览：尚未接入自主恢复规划算法", (70, 89, 110), 74)
    legend.save(ASSETS / "legend.png", optimize=True)
    manifest.append({"file": "legend.png", "size": list(legend.size), "text": ["橙色 A 臂 ｜ 蓝色 B 臂 ｜ 黄色临时桥墩", "环境预览：尚未接入自主恢复规划算法"]})
    (ASSETS / "labels_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
