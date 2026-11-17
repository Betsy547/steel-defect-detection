#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在测试集上评估模型，输出 mAP、每类指标和漏检情况。

工业场景下漏检比误检代价高，所以除了 mAP，这里还会单独打印
每类的 Recall，方便在报告里讨论"选哪个工作点"。

用法：
    python scripts/val.py --weights D:/yolo/weights/best.pt --data-root D:/yolo/datasets/NEU-DET
"""

import _compat  # noqa: F401  控制台中文输出兼容
import argparse
from pathlib import Path

from ultralytics import YOLO

from train import build_data_cfg, dump_data_cfg


def main():
    ap = argparse.ArgumentParser(description="模型评估")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.001, help="评估时用低阈值，算完整 PR 曲线")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    data_cfg = build_data_cfg(Path(args.data), args.data_root)
    data_file = dump_data_cfg(data_cfg)
    model = YOLO(args.weights)

    metrics = model.val(
        data=str(data_file),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=args.device,
        plots=True,
    )

    names = data_cfg.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    print("\n" + "=" * 64)
    print(f"整体指标（{args.split}）")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")
    print("\n每类指标：")
    print(f"  {'类别':<16}{'mAP50':>8}{'Precision':>12}{'Recall':>10}")
    for i, name in enumerate(names):
        try:
            print(
                f"  {name:<16}{metrics.box.ap50[i]:>8.4f}"
                f"{metrics.box.p[i]:>12.4f}{metrics.box.r[i]:>10.4f}"
            )
        except (IndexError, AttributeError):
            break
    print("=" * 64)
    print("\nPR 曲线、混淆矩阵等图表已保存在 runs/detect/val*/ 下，可直接放进报告。")


if __name__ == "__main__":
    main()
