#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大图切片推理：把高分辨率钢板图切成交叠的小块分别检测，再合并结果。

为什么需要：产线相机拍出来常常是 2000x1000 以上，直接缩到 320/640
会让小缺陷缩到几个像素，直接消失。切片后每块保持原始分辨率。

用法：
    python scripts/slice_predict.py --weights D:/yolo/weights/best.pt --source big.jpg \
        --slice 640 --overlap 0.2
"""

import _compat  # noqa: F401  控制台中文输出兼容
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def tile_starts(total: int, size: int, step: int) -> list:
    """计算切片起点，保证最后一块贴住边界。"""
    if total <= size:
        return [0]
    starts = list(range(0, total - size + 1, step))
    if starts[-1] + size < total:
        starts.append(total - size)
    return starts


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float) -> list:
    """按类别做 NMS，返回保留的下标。"""
    keep_all = []
    for cid in np.unique(classes):
        idx = np.where(classes == cid)[0]
        b, s = boxes[idx], scores[idx]
        order = s.argsort()[::-1]
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        while order.size > 0:
            i = order[0]
            keep_all.append(idx[i])
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_thr]
    return keep_all


def slice_predict(model, image, slice_size=640, overlap=0.2, conf=0.25, iou=0.45, imgsz=None):
    """返回 (boxes, scores, classes) 三个 numpy 数组。"""
    H, W = image.shape[:2]
    step = max(int(slice_size * (1 - overlap)), 32)
    imgsz = imgsz or slice_size

    all_boxes, all_scores, all_classes = [], [], []

    for y in tile_starts(H, slice_size, step):
        for x in tile_starts(W, slice_size, step):
            tile = image[y:y + slice_size, x:x + slice_size]
            if tile.size == 0:
                continue
            r = model.predict(tile, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().copy()
            boxes[:, [0, 2]] += x
            boxes[:, [1, 3]] += y
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)

            all_boxes.append(boxes)
            all_scores.append(r.boxes.conf.cpu().numpy())
            all_classes.append(r.boxes.cls.cpu().numpy().astype(int))

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)

    keep = nms_per_class(boxes, scores, classes, iou)
    return boxes[keep], scores[keep], classes[keep]


def main():
    ap = argparse.ArgumentParser(description="大图切片推理")
    ap.add_argument("--weights", default="D:/yolo/weights/best.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--slice", type=int, default=640, help="切片边长")
    ap.add_argument("--overlap", type=float, default=0.2, help="相邻切片重叠比例")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45, help="合并时的 NMS IoU 阈值")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/slice_result.jpg")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    image = cv2.imread(str(src))
    if image is None:
        raise SystemExit(f"无法读取图片：{src}")

    print(f"原图尺寸：{image.shape[1]}x{image.shape[0]}，切片 {args.slice}，重叠 {args.overlap}")

    model = YOLO(args.weights)
    boxes, scores, classes = slice_predict(
        model, image, args.slice, args.overlap, args.conf, args.iou
    )

    print(f"合并后共检出 {len(boxes)} 处缺陷")

    for i, (b, s, c) in enumerate(zip(boxes, scores, classes)):
        color = (0, 0, 255)
        cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
        cv2.putText(
            image, f"{model.names[c]} {s:.2f}",
            (int(b[0]), max(int(b[1]) - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image)
    print(f"结果已保存：{out.resolve()}")


if __name__ == "__main__":
    main()
