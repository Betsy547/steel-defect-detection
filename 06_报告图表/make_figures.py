#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成实践报告用的两张示意图：图 1 技术路线、图 6 系统架构。

用法：
    python make_figures.py
输出：
    图1-技术路线图.png
    图6-系统架构图.png
"""

from pathlib import Path

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent

INK = "#3A4353"
ARROW = "#6B7789"
STAGE = [
    ("#E8F0FB", "#4A6FA5"),
    ("#EAF4EC", "#4F7F5B"),
    ("#FDF3E7", "#B07A3B"),
    ("#F3EEF9", "#6E5A94"),
    ("#EAF3F4", "#3F7A82"),
    ("#FBEDEF", "#9A4E58"),
]


def box(ax, x, y, w, h, title, lines, fill, edge, title_size=12, body_size=9.5):
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top",
            fontsize=title_size, color=edge, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.78, "\n".join(lines), ha="center", va="top",
            fontsize=body_size, color=INK, linespacing=1.55)


def arrow(ax, p1, p2):
    ax.add_patch(mpatches.FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.5, color=ARROW, shrinkA=2, shrinkB=2,
    ))


def fig_route():
    fig, ax = plt.subplots(figsize=(11.2, 6.4), dpi=200)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    w, h = 3.1, 1.7
    xs = [0.55, 4.05, 7.55]
    y_top, y_bot = 4.05, 0.75

    stages = {
        (0, 0): (0, "① 数据准备", ["NEU-DET 1797 张 / 6 类", "转 YOLO 格式并抽样校验"]),
        (1, 0): (1, "② 云端训练", ["AutoDL RTX 4090", "YOLO11s，imgsz 320，150 epochs"]),
        (2, 0): (2, "③ 测试集评估", ["180 张 / 420 个标注框", "mAP50 0.739，逐类分析"]),
        (2, 1): (3, "④ 三模型对比", ["YOLOv8s / YOLO11s / YOLO26s", "同超参控制变量，选部署模型"]),
        (1, 1): (4, "⑤ 检测系统", ["Flask + SQLite + 原生前端", "单张 / 批量 / 历史 / 报告 / API"]),
        (0, 1): (5, "⑥ 本地部署", ["CPU 单张约 26.9 ms", "21 条系统测试全部通过"]),
    }

    for (col, row), (idx, title, lines) in stages.items():
        fill, edge = STAGE[idx]
        box(ax, xs[col], y_top if row == 0 else y_bot, w, h, title, lines, fill, edge)

    arrow(ax, (xs[0] + w + 0.05, y_top + h / 2), (xs[1] - 0.05, y_top + h / 2))
    arrow(ax, (xs[1] + w + 0.05, y_top + h / 2), (xs[2] - 0.05, y_top + h / 2))
    arrow(ax, (xs[2] + w / 2, y_top - 0.05), (xs[2] + w / 2, y_bot + h + 0.05))
    arrow(ax, (xs[2] - 0.05, y_bot + h / 2), (xs[1] + w + 0.05, y_bot + h / 2))
    arrow(ax, (xs[1] - 0.05, y_bot + h / 2), (xs[0] + w + 0.05, y_bot + h / 2))

    fig.savefig(OUT / "图1-技术路线图.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_arch():
    fig, ax = plt.subplots(figsize=(11.2, 6.6), dpi=200)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # 浏览器前端
    box(ax, 0.45, 1.05, 2.75, 4.55, "浏览器（单页前端）",
        ["static/app/", "", "单张检测", "批量检测", "检测历史", "统计看板", "模型信息"],
        "#E8F0FB", "#4A6FA5", title_size=11.5, body_size=10)

    # Flask 应用
    box(ax, 3.95, 3.35, 3.15, 2.25, "Flask 应用 app.py",
        ["页面路由 · /api/* 接口", "参数校验 · 三态判定规则", "调用 detector / db"],
        "#EAF4EC", "#4F7F5B", title_size=11.5, body_size=9.5)

    # 报告模板
    box(ax, 3.95, 1.05, 3.15, 1.75, "报告模板 templates/",
        ["report.html", "A4 版式，浏览器打印为 PDF"],
        "#FDF3E7", "#B07A3B", title_size=11.5, body_size=9.5)

    # 推理模块
    box(ax, 7.85, 3.35, 2.95, 1.7, "detector.py",
        ["Ultralytics YOLO11s 推理", "置信度过滤 · NMS · 画框", "输入 320×320"],
        "#F3EEF9", "#6E5A94", title_size=11.5, body_size=9.5)

    # 权重
    box(ax, 7.85, 5.5, 2.95, 0.95, "权重 02_模型权重/best.pt",
        ["9.43 M 参数，约 19 MB"],
        "#F3EEF9", "#6E5A94", title_size=10.5, body_size=9)

    # 数据库
    box(ax, 7.85, 1.05, 2.95, 1.75, "db.py + SQLite",
        ["records：检测记录", "defects：每个框的类别/坐标", "static/uploads：上传图"],
        "#EAF3F4", "#3F7A82", title_size=11.5, body_size=9.2)

    # 箭头
    arrow(ax, (3.25, 4.45), (3.9, 4.45))
    arrow(ax, (3.9, 4.05), (3.25, 4.05))
    arrow(ax, (7.15, 4.55), (7.8, 4.55))
    arrow(ax, (8.6, 5.46), (8.6, 5.08))
    arrow(ax, (7.8, 3.7), (7.15, 3.7))
    ax.add_patch(mpatches.FancyArrowPatch(
        (7.13, 3.45), (7.8, 2.62), arrowstyle="<|-|>", mutation_scale=14,
        linewidth=1.4, color=ARROW, shrinkA=2, shrinkB=2,
    ))
    arrow(ax, (5.5, 3.3), (5.5, 2.85))

    # 箭头文字
    label = dict(fontsize=8.5, color=ARROW,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.text(3.55, 4.86, "HTTP / JSON", ha="center", va="bottom", **label)
    ax.text(7.47, 4.86, "调用", ha="center", va="bottom", **label)
    ax.text(8.75, 5.27, "加载", ha="left", va="center", **label)
    ax.text(7.47, 3.82, "检测结果", ha="center", va="bottom", **label)
    ax.text(7.62, 3.06, "读写", ha="center", va="bottom", **label)
    ax.text(5.62, 3.05, "渲染", ha="left", va="center", **label)

    fig.savefig(OUT / "图6-系统架构图.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def combine(paths, out, cell_w=1000, gap=24):
    """把两张系统截图并排拼成一张，供报告双图排版。"""
    root = OUT.parent
    imgs = []
    for p in paths:
        im = Image.open(root / p).convert("RGB")
        h = round(im.height * cell_w / im.width)
        imgs.append(im.resize((cell_w, h), Image.LANCZOS))
    height = max(im.height for im in imgs)
    width = cell_w * len(imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += cell_w + gap
    canvas.save(OUT / out)


if __name__ == "__main__":
    fig_route()
    fig_arch()
    combine(["05_系统截图/04-history.jpg", "05_系统截图/05-stats.jpg"], "图8-历史与统计.png")
    combine(["05_系统截图/03-batch.jpg", "05_系统截图/07-report.jpg"], "图9-批量与报告.png")
    print("已生成：图1、图6、图8、图9")
