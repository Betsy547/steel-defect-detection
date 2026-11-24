#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 seed_sweep_runs.csv 生成报告用图（不重新训练/评估）。

用法：
    python scripts/plot_seed_sweep.py --csv 04_训练结果/seed_sweep/seed_sweep_runs.csv
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    with open(args.csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "OK":
                r["seed"] = int(r["seed"])
                r["map50"] = float(r["map50"])
                rows.append(r)

    models = list(dict.fromkeys(r["model"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
    out = Path(args.out) if args.out else Path(args.csv).with_name("seed_sweep.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for i, model in enumerate(models):
        vals = [r["map50"] for r in rows if r["model"] == model]
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ax.bar(i, mean, yerr=std, capsize=4, alpha=0.75, color=f"C{i}")
        jitter = [(j - (len(vals) - 1) / 2) * 0.06 for j in range(len(vals))]
        ax.scatter([i + j for j in jitter], vals, color="black", s=18, zorder=3)
        ax.text(i, mean + std + 0.003, f"{mean:.3f}\n±{std:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("test mAP50")
    ax.set_ylim(0.70, 0.79)
    ax.set_title("Mean ± std over 3 seeds (axis zoomed)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for i, model in enumerate(models):
        vals = []
        for seed in seeds:
            r = next((x for x in rows if x["model"] == model and x["seed"] == seed), None)
            vals.append(r["map50"] if r else float("nan"))
        ax.plot(seeds, vals, marker="o", label=model, color=f"C{i}")
    ax.set_xlabel("seed")
    ax.set_ylabel("test mAP50")
    ax.set_xticks(seeds)
    ax.set_title("Per-seed performance", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"written {out}")


if __name__ == "__main__":
    main()
