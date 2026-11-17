#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 NEU-DET 原始数据（VOC XML）转换为 Ultralytics YOLO 格式并按比例划分。

原始结构（官方下载包，路径大小写可能不同）：
    NEU-DET/
    ├── IMAGES/                 # 或 train/images、validation/images
    │   ├── crazing/
    │   ├── inclusion/
    │   └── ...
    └── ANNOTATIONS/            # 或 annotations/
        ├── crazing_1.xml
        └── ...

输出结构：
    <dst>/
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/

用法：
    python scripts/prepare_dataset.py --src "C:/data/NEU-DET" --dst datasets/NEU-DET
"""

import _compat  # noqa: F401  控制台中文输出兼容
import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# 类别顺序必须与 data.yaml 中 names 的索引严格一致
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_files(root: Path, exts: set) -> list:
    """递归收集指定扩展名的文件，忽略大小写。"""
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def parse_voc(xml_path: Path) -> tuple:
    """解析 VOC XML，返回 (width, height, [yolo 行...])。"""
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError("XML 缺少 <size> 节点")
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    if w <= 0 or h <= 0:
        raise ValueError("图像尺寸非法")

    rows = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None:
            continue
        name = name_node.text.strip().lower()
        if name not in CLASS_TO_ID:
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = float(bnd.find("xmin").text)
        y1 = float(bnd.find("ymin").text)
        x2 = float(bnd.find("xmax").text)
        y2 = float(bnd.find("ymax").text)

        # 裁剪到图像范围内，避免越界框
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))

        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue

        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        rows.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return w, h, rows


def main():
    ap = argparse.ArgumentParser(description="NEU-DET -> YOLO 格式转换与划分")
    ap.add_argument("--src", required=True, help="原始 NEU-DET 根目录")
    ap.add_argument("--dst", default="datasets/NEU-DET", help="输出目录")
    ap.add_argument("--val", type=float, default=0.1, help="验证集比例")
    ap.add_argument("--test", type=float, default=0.1, help="测试集比例")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--overwrite", action="store_true", help="清空已有的输出目录")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        sys.exit(f"源目录不存在：{src}")

    if dst.exists() and any(dst.iterdir()):
        if not args.overwrite:
            sys.exit(f"输出目录非空：{dst}\n如需覆盖请加 --overwrite")
        shutil.rmtree(dst)

    # 1. 收集图片与标注，按文件名主干（stem）配对
    images = find_files(src, IMAGE_EXTS)
    xmls = find_files(src, {".xml"})
    if not images:
        sys.exit(f"在 {src} 下没有找到图片")
    if not xmls:
        sys.exit(f"在 {src} 下没有找到 XML 标注")

    xml_by_stem = defaultdict(list)
    for x in xmls:
        xml_by_stem[x.stem].append(x)

    dup = {k: v for k, v in xml_by_stem.items() if len(v) > 1}
    if dup:
        print(f"[警告] {len(dup)} 个文件名存在多个 XML，取第一个：{list(dup)[:3]}")

    # 2. 转换
    pairs = []
    skipped_no_xml = 0
    skipped_empty = 0
    class_counter = Counter()

    for img in images:
        if img.stem not in xml_by_stem:
            skipped_no_xml += 1
            continue
        try:
            w, h, rows = parse_voc(xml_by_stem[img.stem][0])
        except Exception as e:
            print(f"[跳过] {img.name}: {e}")
            continue
        if not rows:
            skipped_empty += 1
            continue
        pairs.append((img, rows))
        for line in rows:
            class_counter[CLASS_NAMES[int(line.split()[0])]] += 1

    if not pairs:
        sys.exit("没有成功转换任何样本，请检查 XML 与图片是否对应")

    # 3. 划分（分层：按类别打散，保证每类分布均匀）
    by_class = defaultdict(list)
    for item in pairs:
        # NEU-DET 的文件名形如 crazing_12，主干前缀即类别
        prefix = item[0].stem.rsplit("_", 1)[0].lower()
        key = prefix if prefix in CLASS_TO_ID else "_other"
        by_class[key].append(item)

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for key, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_val = int(round(n * args.val))
        n_test = int(round(n * args.test))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise SystemExit(f"类别 {key} 样本太少（{n}），请调小 --val/--test")
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]

    # 4. 写出
    for split, items in splits.items():
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, rows in items:
            shutil.copy2(img_path, img_dir / img_path.name)
            (lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # 5. 汇总
    print("\n转换完成")
    print(f"  输出目录 : {dst}")
    print(f"  跳过（无 XML）: {skipped_no_xml}")
    print(f"  跳过（无目标）: {skipped_empty}")
    print(f"  train/val/test: {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
    print("\n各类别实例数：")
    for name in CLASS_NAMES:
        print(f"  {name:<16} {class_counter.get(name, 0)}")
    print("\n下一步：python scripts/check_dataset.py --root", dst)


if __name__ == "__main__":
    main()
