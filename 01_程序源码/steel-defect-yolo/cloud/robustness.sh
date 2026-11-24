#!/usr/bin/env bash
# AutoDL 上跑鲁棒性压力测试（依赖 seed_sweep 产出的 seed-0 权重）。
#
# 用法：
#   bash cloud/robustness.sh
#   RUNS_DIR=/root/autodl-tmp/steel/runs/seed_sweep bash cloud/robustness.sh
#
# 后台跑：
#   nohup bash cloud/robustness.sh > /root/autodl-tmp/steel/robustness.out 2>&1 &
#   tail -f /root/autodl-tmp/steel/robustness.out
set -euo pipefail

CODE_DIR="${CODE_DIR:-/root/steel-defect-yolo}"
cd "$CODE_DIR"

if [ -d /root/miniconda3/bin ]; then
  export PATH=/root/miniconda3/bin:$PATH
fi

if [ -z "${DATA_ROOT:-}" ]; then
  for d in /root/autodl-tmp/steel/NEU-DET /root/autodl-tmp/NEU-DET; do
    if [ -d "$d" ]; then DATA_ROOT="$d"; break; fi
  done
fi
if [ -z "${DATA_ROOT:-}" ] || [ ! -d "$DATA_ROOT" ]; then
  echo "找不到 NEU-DET，请用 DATA_ROOT=... 指定" >&2
  exit 1
fi

RUNS_DIR="${RUNS_DIR:-/root/autodl-tmp/steel/runs/seed_sweep}"
OUT="${OUT:-/root/autodl-tmp/steel/runs/robustness}"
MODELS="${MODELS:-yolov8s,yolo11s,yolo26s}"
SEED="${SEED:-0}"

echo "CODE_DIR  = $CODE_DIR"
echo "DATA_ROOT = $DATA_ROOT"
echo "RUNS_DIR  = $RUNS_DIR"
echo "OUT       = $OUT"
echo "MODELS    = $MODELS  seed=$SEED"
echo

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python scripts/robustness.py \
  --data "$CODE_DIR/data.yaml" \
  --data-root "$DATA_ROOT" \
  --runs-dir "$RUNS_DIR" \
  --models "$MODELS" \
  --seed "$SEED" \
  --types brightness,contrast,noise,motion_blur,jpeg \
  --levels 1,2,3 \
  --imgsz 320 \
  --batch 32 \
  --out "$OUT"

echo
echo "全部完成：$OUT/robustness.md"
