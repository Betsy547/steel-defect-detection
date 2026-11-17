#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练脚本。本地和云端通用。

关键设计：用 --data-root 覆盖 data.yaml 里的 path，
这样同一份代码在 Windows（D:/yolo/...）和 Linux 服务器（/root/autodl-tmp/...）上都能跑。

用法：
    # 云端
    python scripts/train.py --data-root /root/autodl-tmp/NEU-DET --epochs 150

    # 本地小规模试跑
    python scripts/train.py --data-root D:/yolo/datasets/NEU-DET --epochs 10 --batch 8
"""

import _compat  # noqa: F401  控制台中文输出兼容
import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def build_data_cfg(yaml_path: Path, data_root: str = None) -> dict:
    """读取 data.yaml，按需覆盖数据集根目录。"""
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if data_root:
        root = Path(data_root).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"数据目录不存在：{root}")
        cfg["path"] = str(root)
    elif "path" in cfg:
        root = Path(cfg["path"]).expanduser()
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
        cfg["path"] = str(root)

    return cfg


def dump_data_cfg(cfg: dict) -> Path:
    """把配置写成临时 yaml。

    Ultralytics 的 train/val 的 data 参数只接受文件路径，不接受 dict，
    所以覆盖了 path 之后要落盘一次。
    """
    path = Path(tempfile.gettempdir()) / "ultralytics_dataset_resolved.yaml"
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def main():
    ap = argparse.ArgumentParser(description="YOLO 训练")
    ap.add_argument("--data", default="data.yaml", help="数据集 yaml")
    ap.add_argument("--data-root", default=None, help="覆盖 yaml 中的数据集根目录")
    ap.add_argument("--model", default="yolo11s.pt", help="预训练权重")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=320, help="NEU-DET 原图 200x200，320 足够")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="", help="留空自动选择，'0' 为第一块 GPU，'cpu' 为 CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30, help="多少轮无提升后早停")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="train")
    ap.add_argument("--resume", action="store_true", help="从 last.pt 继续训练")
    ap.add_argument("--cache", action="store_true", help="缓存图片到内存/磁盘，小数据集可开")
    args = ap.parse_args()

    data_cfg = build_data_cfg(Path(args.data), args.data_root)
    data_file = dump_data_cfg(data_cfg)

    # 转成绝对路径，否则 ultralytics 会把相对 project 拼到 settings 里的 runs_dir 下面，
    # 产物位置和命令里写的不一致。
    project = Path(args.project).expanduser().resolve()

    print("=" * 60)
    print(f"模型      : {args.model}")
    print(f"数据根目录: {data_cfg['path']}")
    print(f"类别      : {data_cfg.get('names')}")
    print(f"imgsz={args.imgsz}  batch={args.batch}  epochs={args.epochs}")
    print("=" * 60)

    model = YOLO(args.model)

    model.train(
        data=str(data_file),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=str(project),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        cache=args.cache,
        plots=True,
        val=True,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = Path(trainer.save_dir) if trainer is not None else project / args.name
    best = save_dir / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"训练结果目录：{save_dir}")
    print(f"训练结束，最佳权重：{best}")
    print("下一步：把 best.pt 下载到本地 D:/yolo/weights/ 做演示")
    print("=" * 60)


if __name__ == "__main__":
    main()
