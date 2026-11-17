#!/usr/bin/env bash
# 云端训练环境初始化（AutoDL / 阿里云 / 腾讯云 GPU 实例）
# 用法：bash cloud/setup.sh
set -e

echo "== 1/3 检查 GPU 和 PyTorch =="
python -c "
import torch
print('torch      :', torch.__version__)
print('cuda 可用  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU        :', torch.cuda.get_device_name(0))
    print('显存       :', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
"

echo
echo "== 2/3 安装依赖 =="
pip install -r requirements-server.txt

echo
echo "== 3/3 验证 ultralytics =="
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

echo
echo "完成。下一步：bash cloud/train.sh"
