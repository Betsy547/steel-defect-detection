#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""3 个模型 × 3 个随机种子的重复实验。

目的不是刷高 mAP，而是回答一个报告里常被忽略的问题：
同一套训练协议下，模型之间的差距有多少是真实差异，有多少只是随机种子带来的波动。

用法：
    # 本地（权重在 02_模型权重/）
    python scripts/seed_sweep.py --data-root D:/yolo/datasets/NEU-DET \
        --weights-dir "C:/Users/29507/Desktop/学习/计算机视觉/工业钢板检测/02_模型权重" \
        --seeds 0,1,2 --cache disk

    # AutoDL
    python scripts/seed_sweep.py --data-root /root/autodl-tmp/steel/NEU-DET \
        --weights-dir weights --seeds 0,1,2 --cache disk --workers 8

输出（默认 runs/seed_sweep/）：
    seed_sweep_runs.csv        每次运行一行
    seed_sweep_summary.csv     每个模型一行（均值/标准差/平均排名）
    seed_sweep_per_class.csv   每个模型每类一行
    seed_sweep.md              可直接粘进报告的表格
    logs/                      训练日志
"""

import _compat  # noqa: F401  控制台中文输出兼容
import argparse
import csv
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

from train import build_data_cfg, dump_data_cfg

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ("map50", "mAP50"),
    ("map50_95", "mAP50-95"),
    ("precision", "Precision"),
    ("recall", "Recall"),
]


def parse_list(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def load_names(data_yaml: Path):
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return [str(x) for x in names]


def resolve_weights(spec: str, weights_dir: str):
    """把 yolov8s 之类的简称解析成实际 .pt 路径；找不到就交给 Ultralytics 自动下载。"""
    p = Path(spec)
    if p.suffix == ".pt" and p.is_file():
        return str(p)
    if "/" in spec or "\\" in spec:
        raise SystemExit(f"权重不存在：{spec}")

    candidates = []
    if weights_dir:
        candidates.append(Path(weights_dir).expanduser() / f"{spec}.pt")
    candidates += [
        ROOT / "weights" / f"{spec}.pt",
        ROOT / f"{spec}.pt",
        Path.cwd() / f"{spec}.pt",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    print(f"[warn] 本地找不到 {spec}.pt，交给 Ultralytics 自动下载")
    return spec


def run_name(tag: str, model: str, seed: int) -> str:
    return f"{tag}-{Path(model).stem}-s{seed}"


def best_epoch_from_csv(run_dir: Path):
    results = run_dir / "results.csv"
    if not results.is_file():
        return ""
    try:
        with results.open(encoding="utf-8") as f:
            rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        if not rows:
            return ""
        key = "metrics/mAP50-95(B)"
        best = max(rows, key=lambda r: float(r.get(key, 0) or 0))
        return int(float(best.get("epoch", 0)))
    except Exception:
        return ""


def train_one(args, model_spec: str, seed: int, project: Path, out: Path):
    name = run_name(args.tag, model_spec, seed)
    run_dir = project / name
    best = run_dir / "weights" / "best.pt"

    if best.is_file() and not args.force:
        print(f"[skip] 已有权重：{best}")
        return best, 0.0, run_dir

    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"[force] 已删除旧目录：{run_dir}")

    weights = resolve_weights(model_spec, args.weights_dir)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--data", str(args.data),
        "--model", weights,
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--workers", str(args.workers),
        "--patience", str(args.patience),
        "--seed", str(seed),
        "--cache", args.cache,
        "--project", str(project),
        "--name", name,
    ]
    if args.data_root:
        cmd += ["--data-root", str(args.data_root)]
    if args.device:
        cmd += ["--device", str(args.device)]
    if args.deterministic:
        cmd.append("--deterministic")

    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"[train] {name} -> {log_path}")
    print("        " + " ".join(cmd))
    if args.dry_run:
        return None, 0.0, run_dir

    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if proc.returncode != 0 or not best.is_file():
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        except Exception:
            pass
        print(f"[fail] {name} 退出码 {proc.returncode}")
        if tail:
            print(tail)
        raise RuntimeError(f"{name} 训练失败，详见 {log_path}")

    print(f"[done] {name}  {elapsed / 60:.1f} min  best={best}")
    return best, elapsed, run_dir


def evaluate(best: Path, data_file: Path, args, names):
    model = YOLO(str(best))
    metrics = model.val(
        data=str(data_file),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        device=args.device or None,
        plots=False,
        verbose=False,
        workers=0,
    )
    box = metrics.box
    ap50 = list(box.ap50) if len(box.ap50) else [None] * len(names)
    ap = list(box.ap) if len(box.ap) else [None] * len(names)
    idx = list(getattr(box, "ap_class_index", range(len(names))))
    by_class_ap50 = {names[int(c)]: float(v) for c, v in zip(idx, ap50) if v is not None}
    by_class_ap = {names[int(c)]: float(v) for c, v in zip(idx, ap) if v is not None}

    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    row = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "params_m": params_m,
        "weight_mb": best.stat().st_size / 1024**2,
        "infer_ms": float(metrics.speed.get("inference", 0.0)),
    }
    for cls in names:
        row[f"ap50_{cls}"] = by_class_ap50.get(cls)
        row[f"ap_{cls}"] = by_class_ap.get(cls)
    return row


def mean_std(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return "", ""
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def fmt(value, digits=4):
    return "" if value in ("", None) else f"{value:.{digits}f}"


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_figure(out: Path, rows, models, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("status") == "OK"]
    model_names = [Path(m).stem for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    for i, model in enumerate(model_names):
        vals = [r["map50"] for r in ok if r["model"] == model]
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ax.bar(i, mean, yerr=std, capsize=4, alpha=0.75, color=f"C{i}")
        jitter = [(j - (len(vals) - 1) / 2) * 0.06 for j in range(len(vals))]
        ax.scatter([i + j for j in jitter], vals, color="black", s=18, zorder=3)
        ax.text(i, mean + std + 0.004, f"{mean:.3f}\n±{std:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("test mAP50")
    ax.set_title("Mean ± std over 3 seeds", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for i, model in enumerate(model_names):
        vals = []
        for seed in seeds:
            r = next((x for x in ok if x["model"] == model and x["seed"] == seed), None)
            vals.append(r["map50"] if r else float("nan"))
        ax.plot(seeds, vals, marker="o", label=model, color=f"C{i}")
    ax.set_xlabel("seed")
    ax.set_ylabel("test mAP50")
    ax.set_xticks(seeds)
    ax.set_title("Per-seed performance", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "seed_sweep.png", dpi=300)
    plt.close(fig)


def build_markdown(args, rows, names, models, seeds):
    ok = [r for r in rows if r.get("status") == "OK"]
    models = [Path(m).stem for m in models]

    lines = []
    lines.append("# 3 模型 × 3 随机种子重复实验\n")
    lines.append("## 实验协议\n")
    lines.append(f"- 数据集：NEU-DET，官方 train/val/test 划分，测试集 {args.test_images} 张、{args.test_boxes} 个标注框")
    lines.append(f"- 训练：{args.epochs} epochs，imgsz={args.imgsz}，batch={args.batch}，patience={args.patience}，cache={args.cache}")
    lines.append(f"- 随机种子：{', '.join(str(s) for s in seeds)}；每个模型独立训练 {len(seeds)} 次")
    lines.append(f"- 测试：conf=0.001，IoU=0.5（mAP50）/0.5:0.95（mAP50-95），测试集不参与早停")
    lines.append(f"- 确定性算法：{'开启' if args.deterministic else '关闭（Ultralytics 默认增强随机性）'}")
    lines.append("")

    lines.append("## 单次运行结果\n")
    lines.append("| 模型 | seed | mAP50 | mAP50-95 | P | R | 参数量/M | 权重/MB | best epoch | 训练/min | 状态 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['seed']} | {fmt(r.get('map50'))} | {fmt(r.get('map50_95'))} | "
            f"{fmt(r.get('precision'))} | {fmt(r.get('recall'))} | {fmt(r.get('params_m'), 2)} | "
            f"{fmt(r.get('weight_mb'), 1)} | {r.get('best_epoch', '')} | "
            f"{fmt(r.get('train_min'), 1)} | {r.get('status', '')} |"
        )
    lines.append("")

    lines.append("## 均值与标准差\n")
    lines.append("| 模型 | n | mAP50 | mAP50-95 | P | R | mAP50 变异系数 | 平均排名 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for model in models:
        subset = [r for r in ok if r["model"] == model]
        cells = []
        for key, _ in METRICS:
            m, s = mean_std([r.get(key) for r in subset])
            cells.append(f"{m:.4f} ± {s:.4f}" if m != "" else "")
        m50, s50 = mean_std([r.get("map50") for r in subset])
        cv = (s50 / m50 * 100) if m50 else 0
        ranks = [r.get("_rank") for r in subset if r.get("_rank")]
        mean_rank = statistics.fmean(ranks) if ranks else ""
        summary[model] = {"n": len(subset), "map50": (m50, s50), "mean_rank": mean_rank, "cells": cells, "cv": cv}
        rank_text = f"{mean_rank:.2f}" if mean_rank != "" else ""
        lines.append(
            f"| {model} | {len(subset)} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
            f"{cv:.2f}% | {rank_text} |"
        )
    lines.append("")

    lines.append("## 每个 seed 下的排名（按 mAP50）\n")
    header = "| seed | " + " | ".join(models) + " |"
    lines.append(header)
    lines.append("|---:|" + "---:|" * len(models))
    for seed in seeds:
        subset = [r for r in ok if r["seed"] == seed]
        ranked = sorted(subset, key=lambda r: r.get("map50", 0), reverse=True)
        rank_map = {r["model"]: i + 1 for i, r in enumerate(ranked)}
        lines.append(f"| {seed} | " + " | ".join(str(rank_map.get(m, "")) for m in models) + " |")
    lines.append("")

    lines.append("## 每类 AP50（均值 ± 标准差）\n")
    lines.append("| 模型 | " + " | ".join(names) + " |")
    lines.append("|---|" + "---:|" * len(names))
    for model in models:
        subset = [r for r in ok if r["model"] == model]
        cells = []
        for cls in names:
            m, s = mean_std([r.get(f"ap50_{cls}") for r in subset])
            cells.append(f"{m:.4f} ± {s:.4f}" if m != "" else "")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 自动摘要（写报告时请用自己的话重写）\n")
    if summary:
        best_model = max(summary, key=lambda m: summary[m]["map50"][0])
        m, s = summary[best_model]["map50"]
        lines.append(f"- mAP50 均值最高：**{best_model}**，{m:.4f} ± {s:.4f}。")
        spread = max(v["map50"][0] for v in summary.values()) - min(v["map50"][0] for v in summary.values())
        max_std = max(v["map50"][1] for v in summary.values())
        lines.append(f"- 三个模型 mAP50 均值的最大差距为 {spread:.4f}；单个模型跨 seed 的标准差最大为 {max_std:.4f}。")
        lines.append(f"- 若最大差距小于约 2 倍最大标准差，应表述为“性能相当”，不要宣称某个模型显著更优。")
        stable = min(summary, key=lambda m: summary[m]["cv"])
        lines.append(f"- mAP50 相对波动最小：**{stable}**（变异系数 {summary[stable]['cv']:.2f}%）。")
    else:
        lines.append("- 没有成功完成的运行。")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="3 模型 × 3 seed 重复实验")
    ap.add_argument("--data", default=str(ROOT / "data.yaml"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--weights-dir", default=None, help="存放 yolov8s.pt / yolo11s.pt / yolo26s.pt 的目录")
    ap.add_argument("--models", default="yolov8s,yolo11s,yolo26s")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="seed")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="")
    ap.add_argument("--cache", default="disk", choices=["none", "ram", "disk"])
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--project", default=str(ROOT / "runs" / "seed_sweep"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "seed_sweep"))
    ap.add_argument("--force", action="store_true", help="删除同名 run 目录后重训")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml 不存在：{data_yaml}")
    args.test_images, args.test_boxes = 0, 0
    try:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        test_dir = Path(args.data_root or cfg.get("path", "")) / str(cfg.get("test", "images/test"))
        args.test_images = len(list(test_dir.glob("*.jpg"))) + len(list(test_dir.glob("*.png")))
        label_dir = Path(str(test_dir).replace("images", "labels"))
        for p in label_dir.glob("*.txt"):
            args.test_boxes += sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        pass

    names = load_names(data_yaml)
    eval_data_file = dump_data_cfg(build_data_cfg(data_yaml, args.data_root))
    project = Path(args.project).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    project.mkdir(parents=True, exist_ok=True)

    models = parse_list(args.models)
    seeds = [int(s) for s in parse_list(args.seeds)]
    print(f"模型: {models}")
    print(f"seeds: {seeds}")
    print(f"数据: {args.data_root or data_yaml}")
    print(f"输出: {out}")

    rows = []
    for model_spec in models:
        for seed in seeds:
            name = run_name(args.tag, model_spec, seed)
            row = {
                "model": Path(model_spec).stem,
                "seed": seed,
                "run": name,
                "status": "PENDING",
            }
            try:
                best, train_seconds, run_dir = train_one(args, model_spec, seed, project, out)
                if args.dry_run:
                    row["status"] = "DRY-RUN"
                    rows.append(row)
                    continue
                row.update(evaluate(best, eval_data_file, args, names))
                row["best_epoch"] = best_epoch_from_csv(run_dir)
                row["train_min"] = train_seconds / 60
                row["run_dir"] = str(run_dir)
                row["status"] = "OK"
                print(
                    f"[test] {name}  mAP50={row['map50']:.4f}  "
                    f"mAP50-95={row['map50_95']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}"
                )
            except Exception as exc:
                row["status"] = "FAILED"
                row["error"] = str(exc)
                print(f"[error] {name}: {exc}")
            rows.append(row)

    # 每个 seed 内按 mAP50 排名
    for seed in seeds:
        subset = [r for r in rows if r["seed"] == seed and r.get("status") == "OK"]
        for rank, r in enumerate(sorted(subset, key=lambda x: x.get("map50", 0), reverse=True), 1):
            r["_rank"] = rank

    run_fields = [
        "model", "seed", "run", "status", "map50", "map50_95", "precision", "recall",
        "params_m", "weight_mb", "infer_ms", "best_epoch", "train_min", "run_dir", "error",
    ] + [f"ap50_{c}" for c in names] + [f"ap_{c}" for c in names]
    write_csv(out / "seed_sweep_runs.csv", rows, run_fields)

    ok = [r for r in rows if r.get("status") == "OK"]
    models_done = list(dict.fromkeys(r["model"] for r in ok))
    summary_rows = []
    for model in models_done:
        subset = [r for r in ok if r["model"] == model]
        entry = {"model": model, "n": len(subset)}
        for key, label in METRICS:
            m, s = mean_std([r.get(key) for r in subset])
            entry[f"{key}_mean"], entry[f"{key}_std"] = m, s
        ranks = [r["_rank"] for r in subset if r.get("_rank")]
        entry["mean_rank"] = statistics.fmean(ranks) if ranks else ""
        summary_rows.append(entry)
    write_csv(
        out / "seed_sweep_summary.csv",
        summary_rows,
        ["model", "n"] + [f"{k}_{s}" for k, _ in METRICS for s in ("mean", "std")] + ["mean_rank"],
    )

    per_class_rows = []
    for model in models_done:
        subset = [r for r in ok if r["model"] == model]
        for cls in names:
            m, s = mean_std([r.get(f"ap50_{cls}") for r in subset])
            per_class_rows.append({"model": model, "class": cls, "ap50_mean": m, "ap50_std": s, "n": len(subset)})
    write_csv(out / "seed_sweep_per_class.csv", per_class_rows, ["model", "class", "ap50_mean", "ap50_std", "n"])

    md = build_markdown(args, rows, names, models, seeds)
    (out / "seed_sweep.md").write_text(md, encoding="utf-8")
    try:
        make_figure(out, rows, models, seeds)
    except Exception as exc:
        print(f"[warn] 图生成失败：{exc}")
    print("\n" + md)
    print(f"\n结果已写入：{out}")


if __name__ == "__main__":
    main()
