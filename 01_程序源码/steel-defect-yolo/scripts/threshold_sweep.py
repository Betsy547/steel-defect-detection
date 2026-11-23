#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""置信度阈值扫描实验。

做两件事：
1. 检测阈值 conf 从 0.05 到 0.60 扫描，统计框级 TP/FP/FN、Precision、Recall、
   F1、漏检率。
2. 固定检测阈值（默认 0.15 和 0.25），扫描三态判定阈值 conf_high，
   统计测试集中“合格 / 建议复检 / 不合格”的图像数。

匹配规则：
- NMS IoU 默认 0.45，与系统 detector.py 一致。
- TP 判定：同类、IoU >= 0.5、每张图每个 GT 最多匹配一次（按置信度贪心）。

输出（默认写到 04_训练结果/）：
- threshold_sweep.csv
- image_rule_sweep.csv
- threshold_sweep.png
- threshold_sweep.md

用法：
    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --weights ../../02_模型权重/best.pt --device 0
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def iou_xyxy(a, b):
    """两个 (x1, y1, x2, y2) 框的 IoU。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path, width, height):
    """读取 YOLO 标签，转成 [(cls, (x1, y1, x2, y2)), ...]。"""
    out = []
    if not label_path.is_file():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        out.append((
            cls,
            (
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            ),
        ))
    return out


def match_image(preds, gts, match_iou):
    """一张图的贪心匹配。preds 已按置信度降序。

    preds: [(cls, conf, (x1, y1, x2, y2)), ...]
    gts:   [(cls, (x1, y1, x2, y2)), ...]
    返回 (tp, fp, fn)。
    """
    used = [False] * len(gts)
    tp = fp = 0
    for cls, _conf, box in preds:
        best_iou, best_j = 0.0, -1
        for j, (gcls, gbox) in enumerate(gts):
            if used[j] or gcls != cls:
                continue
            v = iou_xyxy(box, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= match_iou:
            used[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = used.count(False)
    return tp, fp, fn


def list_images(root, split):
    img_dir = root / "images" / split
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def main():
    here = Path(__file__).resolve()
    pkg_root = here.parents[3]  # 工业钢板检测/
    ap = argparse.ArgumentParser(description="置信度阈值扫描实验")
    ap.add_argument("--weights", default=str(pkg_root / "02_模型权重" / "best.pt"))
    ap.add_argument("--data-root", default=str(pkg_root / "03_数据集" / "NEU-DET"))
    ap.add_argument("--out-dir", default=str(pkg_root / "04_训练结果"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="0", help="'0' 为第一块 GPU，'cpu' 为 CPU")
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--confs", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60")
    ap.add_argument("--conf-highs", default="0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--rule-confs", default="0.15,0.25")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 为全部")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(data_root, args.split)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"没有找到图像：{data_root / 'images' / args.split}")

    print(f"模型      : {args.weights}")
    print(f"数据      : {data_root} ({args.split}, {len(images)} 张)")
    print(f"设备      : {args.device}  imgsz={args.imgsz}  NMS IoU={args.nms_iou}  匹配 IoU={args.match_iou}")

    model = YOLO(args.weights)
    confs = [float(v) for v in args.confs.split(",")]
    conf_highs = [float(v) for v in args.conf_highs.split(",")]
    rule_confs = [float(v) for v in args.rule_confs.split(",")]

    # 预读 GT 的尺寸和框
    gts = {}
    sizes = {}
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit(f"图像读取失败：{p}")
        h, w = img.shape[:2]
        sizes[p] = (w, h)
        gts[p] = load_gt(data_root / "labels" / args.split / (p.stem + ".txt"), w, h)

    n_gt = sum(len(v) for v in gts.values())
    print(f"GT 框总数 : {n_gt}")

    sweep_rows = []
    image_rows = []

    for conf in confs:
        results = model.predict(
            [str(p) for p in images],
            conf=conf, iou=args.nms_iou, imgsz=args.imgsz,
            device=args.device, verbose=False,
        )
        tp = fp = fn = 0
        max_confs = {}
        for p, r in zip(images, results):
            if r.boxes is None or len(r.boxes) == 0:
                preds = []
            else:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                order = np.argsort(-scores)
                preds = [
                    (int(classes[i]), float(scores[i]),
                     (float(boxes[i][0]), float(boxes[i][1]),
                      float(boxes[i][2]), float(boxes[i][3])))
                    for i in order
                ]
            t, f, n = match_image(preds, gts[p], args.match_iou)
            tp += t
            fp += f
            fn += n
            max_confs[p] = max([s for _, s, _ in preds], default=0.0)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        miss_rate = fn / n_gt if n_gt else 0.0
        sweep_rows.append({
            "conf": conf, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "miss_rate": miss_rate,
        })
        print(f"conf={conf:.2f}  TP={tp:4d} FP={fp:4d} FN={fn:4d}  "
              f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} 漏检率={miss_rate:.2%}")

        # 三态判定：固定该 conf，扫描 conf_high
        for rc in rule_confs:
            if abs(conf - rc) > 1e-9:
                continue
            for ch in conf_highs:
                pass_n = review_n = fail_n = 0
                for p in images:
                    top = max_confs[p]
                    if top < conf:
                        pass_n += 1
                    elif top >= ch:
                        fail_n += 1
                    else:
                        review_n += 1
                image_rows.append({
                    "conf": conf, "conf_high": ch,
                    "pass": pass_n, "review": review_n, "fail": fail_n,
                    "pass_rate": pass_n / len(images),
                })

    # CSV
    with (out_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    if image_rows:
        with (out_dir / "image_rule_sweep.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
            w.writeheader()
            w.writerows(image_rows)

    # 图
    xs = [r["conf"] for r in sweep_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(xs, [r["precision"] for r in sweep_rows], "o-", label="Precision")
    ax.plot(xs, [r["recall"] for r in sweep_rows], "s-", label="Recall")
    ax.plot(xs, [r["f1"] for r in sweep_rows], "^-", label="F1")
    ax.plot(xs, [r["miss_rate"] for r in sweep_rows], "v--", label="Miss rate (FN/GT)")
    ax.set_xlabel("Detection confidence threshold (conf)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("(a) Box-level P / R / F1 / miss rate")

    ax = axes[1]
    if image_rows:
        labels = [f"{r['conf_high']:.2f}" for r in image_rows if abs(r["conf"] - rule_confs[-1]) < 1e-9]
        rows = [r for r in image_rows if abs(r["conf"] - rule_confs[-1]) < 1e-9]
        pass_v = [r["pass"] for r in rows]
        review_v = [r["review"] for r in rows]
        fail_v = [r["fail"] for r in rows]
        x = np.arange(len(rows))
        ax.bar(x, pass_v, label="Pass (missed image)")
        ax.bar(x, review_v, bottom=pass_v, label="Review")
        ax.bar(x, fail_v, bottom=np.array(pass_v) + np.array(review_v), label="Fail")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel(f"Verdict threshold (conf_high), detection conf={rule_confs[-1]:.2f}")
        ax.set_ylabel("Images")
        ax.set_ylim(0, len(images) * 1.18)
        ax.set_title("(b) Image-level verdict distribution")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No rule-confs selected", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_dir / "threshold_sweep.png", dpi=200)
    plt.close(fig)

    # Markdown 摘要
    lines = []
    lines.append("# 置信度阈值扫描结果\n")
    lines.append(f"- 数据：NEU-DET `{args.split}`，{len(images)} 张图，{n_gt} 个 GT 框")
    lines.append(f"- 模型：`{Path(args.weights).name}`，imgsz={args.imgsz}，NMS IoU={args.nms_iou}，匹配 IoU={args.match_iou}")
    lines.append(f"- 设备：{args.device}\n")
    lines.append("## 1. 检测阈值扫描（框级）\n")
    lines.append("| conf | TP | FP | FN | Precision | Recall | F1 | 漏检率 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sweep_rows:
        lines.append(
            f"| {r['conf']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['miss_rate']:.2%} |"
        )
    if image_rows:
        lines.append("\n## 2. 三态判定阈值扫描（图像级）\n")
        lines.append("| 检测 conf | conf_high | 合格（漏判） | 建议复检 | 不合格 | 合格率 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in image_rows:
            lines.append(
                f"| {r['conf']:.2f} | {r['conf_high']:.2f} | {r['pass']} | "
                f"{r['review']} | {r['fail']} | {r['pass_rate']:.1%} |"
            )
    lines.append("\n## 3. 文件\n")
    lines.append("- `threshold_sweep.csv`：框级扫描原始数据")
    lines.append("- `image_rule_sweep.csv`：图像级判定扫描原始数据")
    lines.append("- `threshold_sweep.png`：左图 P/R/F1/漏检率，右图三态判定分布")
    (out_dir / "threshold_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已写入：{out_dir}")


if __name__ == "__main__":
    main()
