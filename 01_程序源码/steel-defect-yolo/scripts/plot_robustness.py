#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 robustness_runs.csv 重新生成热力图和严重度曲线（不重新评估）。"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TYPE_ORDER = ["brightness", "contrast", "noise", "motion_blur", "jpeg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    with open(args.csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["level"] = int(r["level"])
            r["map50"] = float(r["map50"])
            rows.append(r)

    out = Path(args.out) if args.out else Path(args.csv).parent
    models = list(dict.fromkeys(r["model"] for r in rows))
    conds = ["clean"] + [f"{k}_L{l}" for k in TYPE_ORDER for l in (1, 2, 3)]
    matrix = []
    for model in models:
        vals = {}
        for r in rows:
            if r["model"] != model:
                continue
            key = "clean" if r["perturbation"] == "clean" else f"{r['perturbation']}_L{r['level']}"
            vals[key] = r["map50"]
        matrix.append([vals.get(c, float("nan")) for c in conds])

    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(np.array(matrix, dtype=float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_L", "-L") for c in conds], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    for i, model in enumerate(models):
        for j, cond in enumerate(conds):
            v = matrix[i][j]
            if v == v:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="black" if v < 0.4 else "white")
    fig.colorbar(im, ax=ax, label="mAP50")
    ax.set_title("Robustness stress test: mAP50 under test-time perturbations", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "robustness_heatmap.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        base = next((r["map50"] for r in rows if r["model"] == model and r["perturbation"] == "clean"), float("nan"))
        ax.axhline(base, color="gray", ls="--", lw=1, label="clean")
        for kind in TYPE_ORDER:
            vals = []
            for level in (1, 2, 3):
                hit = next((r["map50"] for r in rows
                            if r["model"] == model and r["perturbation"] == kind and r["level"] == level), np.nan)
                vals.append(hit)
            ax.plot([1, 2, 3], vals, marker="o", label=kind)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("severity")
        ax.set_title(model, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mAP50")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "robustness_curves.png", dpi=300)
    plt.close(fig)
    print(f"written {out / 'robustness_heatmap.png'}")
    print(f"written {out / 'robustness_curves.png'}")


if __name__ == "__main__":
    main()
