#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出模型为部署格式，用于做精度-速度对比实验。

    ONNX          跨平台，CPU/GPU 都能跑
    TensorRT      NVIDIA GPU 上最快，必须在有 GPU 的机器上导出
    OpenVINO      Intel CPU 加速

用法：
    python scripts/export.py --weights D:/yolo/weights/best.pt --format onnx
    python scripts/export.py --weights runs/detect/train/weights/best.pt --format engine --half
"""

import _compat  # noqa: F401  控制台中文输出兼容
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="模型导出")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="onnx",
                    choices=["onnx", "engine", "openvino", "torchscript"])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--half", action="store_true", help="FP16（仅 GPU 有效）")
    ap.add_argument("--dynamic", action="store_true", help="动态输入尺寸")
    args = ap.parse_args()

    model = YOLO(args.weights)
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
    )

    print(f"\n导出完成：{out}")
    print("建议在报告里做一张对比表：")
    print("  | 格式 | mAP50 | 单帧耗时 | 文件大小 |")
    print("  |------|-------|----------|----------|")
    print("  | .pt  |       |          |          |")
    print(f"  | .{args.format} |   |          |          |")


if __name__ == "__main__":
    main()
