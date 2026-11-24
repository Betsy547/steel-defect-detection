#!/usr/bin/env bash
# AutoDL 上跑 3 模型 × 3 seed 重复实验。
#
# 用法：
#   bash cloud/seed_sweep.sh
#   CODE_DIR=/root/autodl-tmp/steel DATA_ROOT=/root/autodl-tmp/steel/NEU-DET bash cloud/seed_sweep.sh
#   MODELS=yolov8s,yolo11s,yolo26s SEEDS=0,1,2 bash cloud/seed_sweep.sh
#
# 后台跑（SSH 断开也不停）：
#   nohup bash cloud/seed_sweep.sh > /root/autodl-tmp/steel/seed_sweep.out 2>&1 &
#   tail -f /root/autodl-tmp/steel/seed_sweep.out
set -euo pipefail

CODE_DIR="${CODE_DIR:-/root/steel-defect-yolo}"
cd "$CODE_DIR"

# AutoDL 镜像的 python 通常在 /root/miniconda3/bin
if [ -d /root/miniconda3/bin ]; then
  export PATH=/root/miniconda3/bin:$PATH
fi

# 数据目录自动探测，兼容两种常见的 AutoDL 布局
if [ -z "${DATA_ROOT:-}" ]; then
  for d in /root/autodl-tmp/steel/NEU-DET /root/autodl-tmp/NEU-DET; do
    if [ -d "$d" ]; then DATA_ROOT="$d"; break; fi
  done
fi
if [ -z "${DATA_ROOT:-}" ] || [ ! -d "$DATA_ROOT" ]; then
  echo "找不到 NEU-DET，请用 DATA_ROOT=... 指定" >&2
  exit 1
fi

# 权重目录自动探测；找不到就让 Ultralytics 自动下载
if [ -z "${WEIGHTS_DIR:-}" ]; then
  for d in "$CODE_DIR/weights" /root/autodl-tmp/steel/weights; do
    if [ -d "$d" ]; then WEIGHTS_DIR="$d"; break; fi
  done
fi
WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"

OUT="${OUT:-/root/autodl-tmp/steel/runs/seed_sweep}"
MODELS="${MODELS:-yolov8s,yolo11s,yolo26s}"
SEEDS="${SEEDS:-0,1,2}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"

echo "CODE_DIR    = $CODE_DIR"
echo "DATA_ROOT   = $DATA_ROOT"
echo "WEIGHTS_DIR = $WEIGHTS_DIR"
echo "OUT         = $OUT"
echo "MODELS      = $MODELS"
echo "SEEDS       = $SEEDS"
echo

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python scripts/seed_sweep.py \
  --data "$CODE_DIR/data.yaml" \
  --data-root "$DATA_ROOT" \
  --weights-dir "$WEIGHTS_DIR" \
  --models "$MODELS" \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS" \
  --imgsz 320 \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --patience 30 \
  --cache disk \
  --deterministic \
  --project "$OUT" \
  --out "$OUT"

echo
echo "全部完成：$OUT/seed_sweep.md"
