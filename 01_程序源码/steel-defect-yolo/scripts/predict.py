#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理脚本：输出结构化 JSON，而不只是画好框的图片。

工业检测的价值在于"数据"：缺陷有几处、什么类别、多大面积、置信度多少。
有了 JSON 才能做统计、分级、报表导出。

用法：
    # 单张
    python scripts/predict.py --weights D:/yolo/weights/best.pt --source a.jpg

    # 整个文件夹
    python scripts/predict.py --weights D:/yolo/weights/best.pt --source test_images/ --save-img
"""

import _compat  # noqa: F401  控制台中文输出兼容
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def collect_images(source: Path) -> list:
    if source.is_file():
        return [source]
    return sorted(
        p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )


def main():
    ap = argparse.ArgumentParser(description="YOLO 推理")
    ap.add_argument("--weights", default="D:/yolo/weights/best.pt")
    ap.add_argument("--source", required=True, help="图片/文件夹/视频路径")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="cpu", help="本地演示用 cpu")
    ap.add_argument("--out", default="outputs", help="输出目录")
    ap.add_argument("--save-img", action="store_true", help="同时保存画框后的图片")
    args = ap.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        raise SystemExit(f"路径不存在：{source}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    model = YOLO(args.weights)

    # 视频单独处理：逐帧结果不写 JSON，直接存带框视频
    if source.suffix.lower() in VIDEO_EXTS:
        model.predict(
            str(source),
            conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            device=args.device, save=True, project=str(out_dir), name="video",
            exist_ok=True, verbose=False,
        )
        print(f"视频结果已保存到：{out_dir / 'video'}")
        return

    images = collect_images(source)
    if not images:
        raise SystemExit(f"在 {source} 下没有找到图片")

    records = []
    t_total = 0.0

    for img_path in images:
        t0 = time.perf_counter()
        result = model.predict(
            str(img_path), conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        t_total += elapsed_ms

        defects = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            defects.append({
                "class": model.names[int(box.cls)],
                "conf": round(float(box.conf), 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "area_px": round((x2 - x1) * (y2 - y1), 1),
            })

        records.append({
            "image": img_path.name,
            "defects": defects,
            "total": len(defects),
            "infer_ms": round(elapsed_ms, 1),
            "pass": len(defects) == 0,
        })

        if args.save_img:
            annotated = result.plot()
            cv2.imwrite(str(out_dir / "images" / img_path.name), annotated)

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_defects = sum(r["total"] for r in records)
    print(f"\n处理完成：{len(records)} 张图片，共检出 {total_defects} 处缺陷")
    print(f"平均单张耗时：{t_total / len(records):.1f} ms")
    print(f"结构化结果：{json_path}")
    if args.save_img:
        print(f"可视化结果：{out_dir / 'images'}")


if __name__ == "__main__":
    main()
