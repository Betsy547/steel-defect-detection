#!/bin/bash
# 对比实验：yolov8s / yolo26s，与 yolo11s 用完全相同的超参
set -u
cd /root/autodl-tmp/steel
export PATH=/root/miniconda3/bin:$PATH

for M in yolov8s yolo26s; do
  echo "===== TRAIN $M ====="
  python scripts/train.py --data data.yaml --data-root /root/autodl-tmp/steel/NEU-DET \
    --model weights/$M.pt --epochs 150 --batch 32 --imgsz 320 --workers 8 --patience 30 --cache \
    --project /root/autodl-tmp/steel/runs --name neu-det-$M
  echo "===== TEST $M ====="
  python scripts/val.py --weights runs/neu-det-$M/weights/best.pt --data data.yaml \
    --data-root /root/autodl-tmp/steel/NEU-DET --split test --imgsz 320 --batch 32
done

echo "===== ALL DONE ====="
