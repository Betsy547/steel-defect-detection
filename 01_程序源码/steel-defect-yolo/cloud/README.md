# 云端训练流程

以 AutoDL 为例，其他云厂商大同小异。

## 1. 上传代码

在**本地**项目根目录执行（把 `user@host` 和端口换成你的实例信息）：

```bash
scp -P 12345 -r ./* user@region.autodl.com:/root/steel-defect-yolo/
```

或者用 AutoDL 自带的 JupyterLab 拖拽上传。

## 2. 上传数据集

NEU-DET 转换好之后只有 ~50MB，直接传：

```bash
scp -P 12345 -r D:/yolo/datasets/NEU-DET user@region.autodl.com:/root/autodl-tmp/NEU-DET
```

> **注意**：AutoDL 的数据盘 `/root/autodl-tmp` 不计入系统盘容量，
> 且关机后保留。数据集和 `runs/` 都放这里。

## 3. 训练

```bash
cd /root/steel-defect-yolo
bash cloud/setup.sh
bash cloud/train.sh
```

常用变体：

```bash
# 换模型
MODEL=yolo26s.pt bash cloud/train.sh

# 跑对照实验，换个名字即可，不会覆盖
NAME=exp_yolo26 MODEL=yolo26s.pt bash cloud/train.sh

# 缩短训练做快速验证
EPOCHS=20 bash cloud/train.sh
```

## 4. 下载权重

```bash
# 在本地执行
scp -P 12345 user@region.autodl.com:/root/steel-defect-yolo/runs/detect/train/weights/best.pt D:/yolo/weights/
```

顺便把图表一起下回来，报告里要用：

```bash
scp -P 12345 -r user@region.autodl.com:/root/steel-defect-yolo/runs/detect/train/ D:/yolo/runs/
```

## 5. 本地演示

```bash
streamlit run app/streamlit_app.py
```

模型路径填 `D:/yolo/weights/best.pt`。

## 6. 3 模型 × 3 随机种子重复实验

`cloud/seed_sweep.sh` 会依次训练 yolov8s / yolo11s / yolo26s，
每个模型用 seed 0/1/2 各训一次，然后统一在测试集上评估，
输出 `seed_sweep.md`（可直接改成报告表格）。

```bash
cd /root/steel-defect-yolo
bash cloud/seed_sweep.sh
```

脚本会自动探测两种常见的数据布局：

- `/root/autodl-tmp/steel/NEU-DET`
- `/root/autodl-tmp/NEU-DET`

也可以手动指定，并放到后台跑：

```bash
CODE_DIR=/root/steel-defect-yolo \
DATA_ROOT=/root/autodl-tmp/steel/NEU-DET \
WEIGHTS_DIR=/root/steel-defect-yolo/weights \
OUT=/root/autodl-tmp/steel/runs/seed_sweep \
nohup bash cloud/seed_sweep.sh > /root/autodl-tmp/steel/seed_sweep.out 2>&1 &

tail -f /root/autodl-tmp/steel/seed_sweep.out
```

结果文件：

- `seed_sweep_runs.csv`：9 次运行逐行记录
- `seed_sweep_summary.csv`：每个模型的均值/标准差/平均排名
- `seed_sweep_per_class.csv`：每类 AP50
- `seed_sweep.md`：汇总表

## 7. 鲁棒性压力测试

依赖第 6 步产出的 `seed-<model>-s0/weights/best.pt`。只改测试图像，不改标注：

```bash
cd /root/steel-defect-yolo
bash cloud/robustness.sh
```

默认条件：clean 基线 + 5 种扰动 × 3 个严重度（变暗、低对比度、高斯噪声、水平运动模糊、JPEG 压缩），
三个模型都会评估。后台跑：

```bash
RUNS_DIR=/root/autodl-tmp/steel/runs/seed_sweep \
OUT=/root/autodl-tmp/steel/runs/robustness \
nohup bash cloud/robustness.sh > /root/autodl-tmp/steel/robustness.out 2>&1 &
```

结果文件：

- `robustness_runs.csv`：模型 × 条件逐行记录
- `robustness_summary.csv`：模型 × 扰动类型汇总
- `robustness.md`：报告用表格
- `robustness_heatmap.png`、`robustness_curves.png`、`robustness_examples.png`

---

## 跑多组对照实验的建议

改 `--name` 即可，每组产物独立存放：

```bash
NAME=baseline_v8s  MODEL=yolov8s.pt  bash cloud/train.sh
NAME=baseline_v11s MODEL=yolo11s.pt  bash cloud/train.sh
NAME=exp_p2        MODEL=yolo11s.pt  EPOCHS=150 bash cloud/train.sh
```

最后把每组的 `results.csv` 最后一行取出来，就是报告里的对比表。

## 断点续训

实例被关机或训练中断后：

```bash
python scripts/train.py --data-root /root/autodl-tmp/NEU-DET --resume
```

前提是 `runs/detect/<name>/weights/last.pt` 还在。
