#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成适合插入 A4 报告（正文宽约 5.75 in）的 seed / 鲁棒性图。

与云端生成的宽图相比，这里把字号和画布比例按 Word 正文宽度重新排版。
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TYPE_LABELS = {
    "brightness": "brightness",
    "contrast": "contrast",
    "noise": "noise",
    "motion_blur": "motion blur",
    "jpeg": "JPEG",
}
TYPE_ORDER = ["brightness", "contrast", "noise", "motion_blur", "jpeg"]
MODEL_ORDER = ["yolov8s", "yolo11s", "yolo26s"]


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def seed_figure(csv_path, out):
    rows = [r for r in read_csv(csv_path) if r.get("status") == "OK"]
    for r in rows:
        r["seed"] = int(r["seed"])
        r["map50"] = float(r["map50"])
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    seeds = sorted(set(r["seed"] for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    ax = axes[0]
    for i, model in enumerate(models):
        vals = [r["map50"] for r in rows if r["model"] == model]
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ax.bar(i, mean, yerr=std, capsize=3, alpha=0.75, color=f"C{i}", width=0.62)
        jitter = [(j - (len(vals) - 1) / 2) * 0.07 for j in range(len(vals))]
        ax.scatter([i + j for j in jitter], vals, color="black", s=12, zorder=3)
        ax.text(i, mean + std + 0.002, f"{mean:.3f}\n±{std:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylim(0.70, 0.79)
    ax.set_ylabel("test mAP50", fontsize=8.5)
    ax.set_title("Mean ± std over 3 seeds", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for i, model in enumerate(models):
        vals = []
        for seed in seeds:
            r = next((x for x in rows if x["model"] == model and x["seed"] == seed), None)
            vals.append(r["map50"] if r else float("nan"))
        ax.plot(seeds, vals, marker="o", markersize=4, linewidth=1.4, label=model, color=f"C{i}")
    ax.set_xlabel("seed", fontsize=8.5)
    ax.set_ylabel("test mAP50", fontsize=8.5)
    ax.set_xticks(seeds)
    ax.tick_params(labelsize=8)
    ax.set_title("Per-seed performance", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def robustness_figures(csv_path, out_dir):
    rows = read_csv(csv_path)
    for r in rows:
        r["level"] = int(r["level"])
        r["map50"] = float(r["map50"])
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    conds = [("clean", 0)] + [(k, l) for k in TYPE_ORDER for l in (1, 2, 3)]
    labels = ["clean"] + [f"{TYPE_LABELS[k]} L{l}" for k in TYPE_ORDER for l in (1, 2, 3)]

    matrix = np.full((len(conds), len(models)), np.nan)
    for i, (kind, level) in enumerate(conds):
        for j, model in enumerate(models):
            r = next((x for x in rows if x["model"] == model
                      and x["perturbation"] == kind and x["level"] == level), None)
            if r:
                matrix[i, j] = r["map50"]

    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(conds)):
        for j in range(len(models)):
            v = matrix[i, j]
            if v == v:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="black" if v < 0.4 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("mAP50", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Robustness: mAP50 under test-time perturbations", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "图13-robustness-heatmap.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.6), sharey=True)
    axes = axes.ravel()
    for ax, model in zip(axes, models):
        base = next((r["map50"] for r in rows if r["model"] == model and r["perturbation"] == "clean"), np.nan)
        ax.axhline(base, color="gray", ls="--", lw=0.9, label="clean")
        for kind in TYPE_ORDER:
            vals = []
            for level in (1, 2, 3):
                hit = next((r["map50"] for r in rows
                            if r["model"] == model and r["perturbation"] == kind and r["level"] == level), np.nan)
                vals.append(hit)
            ax.plot([1, 2, 3], vals, marker="o", markersize=3.5, linewidth=1.2,
                    label=TYPE_LABELS[kind])
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("severity", fontsize=7.5)
        ax.set_title(model, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mAP50", fontsize=8)
    axes[2].set_ylabel("mAP50", fontsize=8)
    handles, labels_ = axes[0].get_legend_handles_labels()
    axes[3].axis("off")
    axes[3].legend(handles, labels_, fontsize=7.5, loc="center", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "图14-robustness-curves.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-csv", required=True)
    ap.add_argument("--robust-csv", required=True)
    ap.add_argument("--out", default="06_报告图表")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_figure(args.seed_csv, out / "图12-seed-sweep.png")
    robustness_figures(args.robust_csv, out)
    print("written", out / "图12-seed-sweep.png")
    print("written", out / "图13-robustness-heatmap.png")
    print("written", out / "图14-robustness-curves.png")


if __name__ == "__main__":
    main()
