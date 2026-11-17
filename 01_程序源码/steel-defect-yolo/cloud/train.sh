#!/usr/bin/env bash
# 一键训练。数据集路径通过环境变量覆盖，不写死在代码里。
#
# 用法：
#   bash cloud/train.sh
#   DATA_ROOT=/root/autodl-tmp/NEU-DET EPOCHS=200 bash cloud/train.sh
#   MODEL=yolo26s.pt bash cloud/train.sh
set -e

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/NEU-DET}"
MODEL="${MODEL:-yolo11s.pt}"
EPOCHS="${EPOCHS:-150}"
IMGSZ="${IMGSZ:-320}"
BATCH="${BATCH:-32}"
NAME="${NAME:-train}"

echo "数据: $DATA_ROOT"
echo "模型: $MODEL  epochs=$EPOCHS  imgsz=$IMGSZ  batch=$BATCH"
echo

python scripts/train.py \
  --data-root "$DATA_ROOT" \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --name "$NAME"

echo
echo "权重位置：runs/detect/$NAME/weights/best.pt"
