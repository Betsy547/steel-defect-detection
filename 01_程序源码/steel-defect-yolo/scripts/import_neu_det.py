#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把已划分好的 NEU-DET（images/{train,val,test} + annotations/*.xml）
转成 Ultralytics 需要的 images/ labels/ 平行结构，并保留原始划分。

适用于 Hugging Face 上 KeenForgeAI/NEU-DET-corrected 这类
"按 split 分好目录、标注是 VOC XML" 的版本。

用法：
    python scripts/import_neu_det.py --src D:/yolo/datasets/NEU-DET-raw \
                                     --dst D:/yolo/datasets/NEU-DET
"""

import _compat  # noqa: F401  控制台中文输出兼容
import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_voc(xml_path: Path):
    """返回 (w, h, [yolo 行...])。"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = float(size.find("width").text), float(size.find("height").text)

    rows = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASS_TO_ID:
            continue
        b = obj.find("bndbox")
        if b is None:
            continue
        x1 = max(0.0, min(float(b.findtext("xmin")), w))
        y1 = max(0.0, min(float(b.findtext("ymin")), h))
        x2 = max(0.0, min(float(b.findtext("xmax")), w))
        y2 = max(0.0, min(float(b.findtext("ymax")), h))
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        rows.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return w, h, rows


def main():
    ap = argparse.ArgumentParser(description="导入已划分的 NEU-DET")
    ap.add_argument("--src", required=True, help="原始目录（含 images/ 和 annotations/）")
    ap.add_argument("--dst", default="D:/yolo/datasets/NEU-DET", help="输出目录")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    ann_dir = src / "annotations"
    if not ann_dir.is_dir():
        raise SystemExit(f"找不到 annotations 目录：{ann_dir}")

    if dst.exists() and any(dst.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"输出目录非空：{dst}\n如需覆盖请加 --overwrite")
        shutil.rmtree(dst)

    total = Counter()
    print(f"{'split':<8}{'图片':>8}{'缺失标注':>10}{'无目标':>8}{'目标数':>10}")
    print("-" * 46)

    for split in ("train", "val", "test"):
        img_src = src / "images" / split
        if not img_src.is_dir():
            continue

        img_dst = dst / "images" / split
        lbl_dst = dst / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        n_img = n_missing = n_empty = n_box = 0

        for img in sorted(img_src.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            n_img += 1
            shutil.copy2(img, img_dst / img.name)

            xml = ann_dir / f"{img.stem}.xml"
            if not xml.exists():
                n_missing += 1
                (lbl_dst / f"{img.stem}.txt").write_text("", encoding="utf-8")
                continue

            _, _, rows = parse_voc(xml)
            if not rows:
                n_empty += 1
            n_box += len(rows)
            (lbl_dst / f"{img.stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
            )

        total[split] = (n_img, n_missing, n_empty, n_box)
        print(f"{split:<8}{n_img:>8}{n_missing:>10}{n_empty:>8}{n_box:>10}")

    print("-" * 46)
    print(f"合计目标框：{sum(v[3] for v in total.values())}")
    print(f"\n输出目录：{dst}")
    print("下一步：python scripts/check_dataset.py --root", dst)


if __name__ == "__main__":
    main()
