#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理延迟与批量吞吐基准（CPU 或 GPU）。

测量内容：
1. 权重加载时间（YOLO(weights)）。
2. 首张冷启动时间（含 predictor 初始化）。
3. 单张端到端延迟：预热后抽 N 张测试图，逐张 `model.predict`，
   记录总耗时、Ultralytics 自报的 preprocess / inference / postprocess。
4. 批量吞吐：按 1 / 4 / 8 / 16 / 20 张一组调用 `predict`，统计 images/s。

说明：
- device 默认 'cpu'；有 NVIDIA 显卡时加 --device 0 测 GPU。conf=0.25、iou=0.45、imgsz=320，与系统默认一致。
- 图片事先读入内存，计时不含磁盘 I/O；Web 请求的实际耗时会再多几毫秒。
- 批量吞吐是“按 N 张一组调用 predict”的吞吐，包含 Python 前后处理开销，
  不是严格意义的单次 batch tensor 推理。

输出（默认写到 04_训练结果/）：
- cpu_latency.csv
- cpu_latency.md
- cpu_latency.png

用法：
    python scripts/cpu_latency.py
    python scripts/cpu_latency.py --n 100 --device cpu
"""

import argparse
import csv
import os
import platform
import random
import statistics
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


def list_images(root, split):
    img_dir = root / "images" / split
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def pct(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def main():
    here = Path(__file__).resolve()
    pkg_root = here.parents[3]
    ap = argparse.ArgumentParser(description="推理延迟与批量吞吐基准（CPU / GPU）")
    ap.add_argument("--weights", default=str(pkg_root / "02_模型权重" / "best.pt"))
    ap.add_argument("--data-root", default=str(pkg_root / "03_数据集" / "NEU-DET"))
    ap.add_argument("--out-dir", default=str(pkg_root / "04_训练结果"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=100, help="计时图片数")
    ap.add_argument("--warmup", type=int, default=10, help="预热图片数")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu", help="'cpu' 或 '0'（第一块 GPU）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-sizes", default="1,4,8,16,20")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(data_root, args.split)
    if len(images) < args.n + args.warmup:
        raise SystemExit(f"图片不足：{len(images)} 张，需要 {args.n + args.warmup} 张")
    random.Random(args.seed).shuffle(images)
    warm_paths = images[: args.warmup]
    test_paths = images[args.warmup : args.warmup + args.n]

    print(f"模型      : {args.weights}")
    print(f"数据      : {data_root} ({args.split})")
    print(f"设备      : {args.device}  imgsz={args.imgsz}  conf={args.conf}  iou={args.iou}")
    print(f"计时      : 预热 {len(warm_paths)} 张，计时 {len(test_paths)} 张")
    on_gpu = args.device not in ("cpu", "")
    if on_gpu:
        idx = int(args.device) if str(args.device).isdigit() else 0
        dev_name = torch.cuda.get_device_name(idx)
        dev_label = "GPU"
        print(f"GPU       : {dev_name}")
    else:
        dev_name = platform.processor() or platform.machine()
        dev_label = "CPU"
        print(f"CPU       : {dev_name}  "
              f"逻辑核心={os.cpu_count()}  torch线程={torch.get_num_threads()}")

    # 预读图片，计时不含磁盘 I/O
    def read(p):
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit(f"图像读取失败：{p}")
        return img

    warm_imgs = [read(p) for p in warm_paths]
    test_imgs = [read(p) for p in test_paths]

    # 加载权重
    t0 = time.perf_counter()
    model = YOLO(args.weights)
    load_ms = (time.perf_counter() - t0) * 1000

    # 首张冷启动
    t0 = time.perf_counter()
    model.predict(warm_imgs[0], conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                  device=args.device, verbose=False)
    first_ms = (time.perf_counter() - t0) * 1000

    # 预热
    for img in warm_imgs[1:]:
        model.predict(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                      device=args.device, verbose=False)

    # 单张计时
    rows = []
    for i, img in enumerate(test_imgs):
        t0 = time.perf_counter()
        r = model.predict(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]
        total_ms = (time.perf_counter() - t0) * 1000
        speed = getattr(r, "speed", {}) or {}
        n_boxes = 0 if r.boxes is None else len(r.boxes)
        rows.append({
            "idx": i,
            "total_ms": total_ms,
            "preprocess_ms": float(speed.get("preprocess", 0.0)),
            "inference_ms": float(speed.get("inference", 0.0)),
            "postprocess_ms": float(speed.get("postprocess", 0.0)),
            "n_boxes": n_boxes,
        })

    totals = [r["total_ms"] for r in rows]
    infers = [r["inference_ms"] for r in rows]
    stats = {
        "mean": statistics.mean(totals),
        "std": statistics.pstdev(totals),
        "min": min(totals),
        "p50": pct(totals, 50),
        "p90": pct(totals, 90),
        "p95": pct(totals, 95),
        "max": max(totals),
        "infer_mean": statistics.mean(infers) if infers else 0.0,
        "infer_std": statistics.pstdev(infers) if infers else 0.0,
    }

    # 批量吞吐
    batch_sizes = [int(v) for v in args.batch_sizes.split(",")]
    batch_rows = []
    for bs in batch_sizes:
        # 预热一组，避免首次 batch 形状变化带来的初始化开销
        model.predict(test_imgs[:bs], conf=args.conf, iou=args.iou,
                      imgsz=args.imgsz, device=args.device, verbose=False)
        t0 = time.perf_counter()
        for s in range(0, len(test_imgs), bs):
            model.predict(test_imgs[s:s + bs], conf=args.conf, iou=args.iou,
                          imgsz=args.imgsz, device=args.device, verbose=False)
        elapsed = time.perf_counter() - t0
        ips = len(test_imgs) / elapsed
        batch_rows.append({
            "batch": bs,
            "total_s": elapsed,
            "images_per_s": ips,
            "ms_per_image": elapsed * 1000 / len(test_imgs),
        })
        print(f"batch={bs:2d}  总耗时={elapsed:.2f}s  {ips:.1f} images/s  "
              f"{elapsed * 1000 / len(test_imgs):.1f} ms/张")

    # CSV：单张明细
    with (out_dir / "cpu_latency.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.hist(totals, bins=20, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.axvline(stats["mean"], color="#E45756", linestyle="--", linewidth=1.5,
               label=f"Mean {stats['mean']:.1f} ms")
    ax.axvline(stats["p95"], color="#F58518", linestyle=":", linewidth=1.5,
               label=f"P95 {stats['p95']:.1f} ms")
    ax.set_xlabel("End-to-end latency per image (ms)")
    ax.set_ylabel("Images")
    ax.set_title(f"(a) Single-image {dev_label} latency")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    xs = [str(r["batch"]) for r in batch_rows]
    ys = [r["images_per_s"] for r in batch_rows]
    bars = ax.bar(xs, ys, color="#54A24B", alpha=0.9)
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_xlabel("Images per predict call")
    ax.set_ylabel("Throughput (images/s)")
    ax.set_title(f"(b) Batch throughput ({dev_label})")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "cpu_latency.png", dpi=200)
    plt.close(fig)

    # Markdown
    lines = []
    lines.append(f"# {dev_label} 推理延迟与批量吞吐\n")
    lines.append(f"- 模型：`{Path(args.weights).name}`，imgsz={args.imgsz}，"
                 f"conf={args.conf}，NMS IoU={args.iou}，device=`{args.device}`")
    lines.append(f"- 数据：NEU-DET `{args.split}`，预热 {len(warm_paths)} 张，计时 {len(test_paths)} 张")
    if on_gpu:
        lines.append(f"- 环境：{dev_name}，PyTorch {torch.__version__}，CUDA {torch.version.cuda}")
    else:
        lines.append(f"- 环境：{dev_name}，"
                     f"逻辑核心 {os.cpu_count()}，PyTorch {torch.__version__}，"
                     f"torch 线程 {torch.get_num_threads()}，Ultralytics 见运行日志")
    lines.append(f"- 权重加载：{load_ms:.1f} ms；首张冷启动：{first_ms:.1f} ms\n")
    lines.append("## 单张端到端延迟\n")
    lines.append("| 指标 | 数值 (ms) |")
    lines.append("|---|---:|")
    for key, label in [("mean", "均值"), ("std", "标准差"), ("min", "最小值"),
                       ("p50", "P50"), ("p90", "P90"), ("p95", "P95"), ("max", "最大值")]:
        lines.append(f"| {label} | {stats[key]:.2f} |")
    lines.append(f"\nUltralytics 自报纯推理耗时：均值 {stats['infer_mean']:.2f} ms，"
                 f"标准差 {stats['infer_std']:.2f} ms。\n")
    lines.append("## 批量吞吐\n")
    lines.append("| 每组张数 | 总耗时 (s) | 吞吐 (images/s) | 平均每张 (ms) |")
    lines.append("|---:|---:|---:|---:|")
    for r in batch_rows:
        lines.append(f"| {r['batch']} | {r['total_s']:.2f} | {r['images_per_s']:.1f} | "
                     f"{r['ms_per_image']:.1f} |")
    lines.append("\n## 说明\n")
    lines.append("- 计时为端到端 `model.predict`，含预处理、推理、NMS 和后处理；不含磁盘读取。")
    lines.append("- 批量吞吐按“N 张一组调用 predict”统计，包含 Python 前后处理开销。")
    lines.append("- 笔记本的 CPU 和 GPU 都受功耗与温度影响，重复运行会有几个百分点的波动。")
    lines.append("- 计时为端到端墙钟时间。GPU 上 Ultralytics 在推理后调用 `torch.cuda.synchronize()`，"
                 "所以每张图的耗时已包含等待 GPU 算完的时间。")
    (out_dir / "cpu_latency.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n单张均值 {stats['mean']:.1f} ms，P95 {stats['p95']:.1f} ms，"
          f"纯推理均值 {stats['infer_mean']:.1f} ms")
    print(f"结果已写入：{out_dir}")


if __name__ == "__main__":
    main()
