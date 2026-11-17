#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集校验：抽样把标签框画回原图，确认标注没有错位。

这一步很重要。VOC 转 YOLO 时常见的错误（坐标没归一化、宽高顺序反了、
类别索引错位）在训练时不会报错，只会让 mAP 莫名其妙上不去。

用法：
    python scripts/check_dataset.py --root D:/yolo/datasets/NEU-DET --samples 20
"""

import _compat  # noqa: F401  控制台中文输出兼容
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
COLORS = [
    (0, 0, 255),
    (0, 200, 0),
    (255, 128, 0),
    (255, 0, 255),
    (0, 200, 255),
    (128, 0, 255),
]


def draw_one(img_path: Path, lbl_path: Path, names: list) -> tuple:
    """在原图上画框，返回 (图像, 框数量, 类别计数)。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, 0, Counter()

    h, w = img.shape[:2]
    counter = Counter()
    n = 0

    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = COLORS[cid % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = names[cid] if cid < len(names) else str(cid)
            cv2.putText(
                img, label, (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )
            counter[label] += 1
            n += 1

    return img, n, counter


def main():
    ap = argparse.ArgumentParser(description="数据集抽样校验")
    ap.add_argument("--root", required=True, help="数据集根目录")
    ap.add_argument("--samples", type=int, default=20, help="每个 split 抽样张数")
    ap.add_argument("--out", default="outputs/check", help="预览图输出目录")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"目录不存在：{root}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    overall = Counter()

    print(f"数据集根目录：{root}\n")

    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            print(f"[{split}] 跳过（不存在 {img_dir}）")
            continue

        images = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not images:
            print(f"[{split}] 跳过（没有图片）")
            continue

        missing = [p for p in images if not (lbl_dir / f"{p.stem}.txt").exists()]
        empty = [
            p for p in images
            if (lbl_dir / f"{p.stem}.txt").exists()
            and not (lbl_dir / f"{p.stem}.txt").read_text(encoding="utf-8").strip()
        ]

        sampled = rng.sample(images, min(args.samples, len(images)))
        split_counter = Counter()
        bad = []

        for img_path in sampled:
            img, n, counter = draw_one(
                img_path, lbl_dir / f"{img_path.stem}.txt", CLASS_NAMES
            )
            if img is None:
                bad.append(img_path.name)
                continue
            split_counter.update(counter)
            cv2.imwrite(str(out_dir / f"{split}_{img_path.name}"), img)

        overall.update(split_counter)

        print(f"[{split}] 图片 {len(images)} 张")
        print(f"        缺标签文件 {len(missing)} 张，空标签 {len(empty)} 张")
        if bad:
            print(f"        无法读取的图片：{bad[:5]}")
        if split_counter:
            items = ", ".join(f"{k}={v}" for k, v in split_counter.most_common())
            print(f"        抽样目标：{items}")
        print()

    print("全类别实例总数（仅统计抽样部分）：")
    for name in CLASS_NAMES:
        print(f"  {name:<16} {overall.get(name, 0)}")

    print(f"\n预览图已保存到：{out_dir}")
    print("请打开这些图确认：框是否贴合缺陷、类别文字是否正确、有没有框跑出图像。")


if __name__ == "__main__":
    main()
