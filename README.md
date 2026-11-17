# 工业钢板表面缺陷检测系统

计算机视觉课程设计 —— 基于 YOLO 的热轧钢带表面缺陷检测。

上传一张钢板图像，系统自动检测 6 类表面缺陷并给出「合格 / 建议复检 / 不合格」判定，
同时提供批量检测、历史记录、统计看板、检测报告和开放接口。

| 项目 | 内容 |
|---|---|
| 数据集 | NEU-DET（东北大学热轧钢带表面缺陷数据集） |
| 模型 | YOLO11s，NEU-DET 微调，输入 320×320 |
| 测试集 mAP50 | **0.739**（mAP50-95 0.427） |
| 训练环境 | AutoDL RTX 4090，150 epochs，约 7 分钟 |
| 部署环境 | 本地 CPU，单张推理约 30–50 ms |
| 系统形态 | Flask + 原生 HTML/CSS/JS 单页前端 + SQLite + JSON 接口 |

## 目录结构

```
工业钢板检测/
├── app.py                      Web 系统入口（Flask）
├── detector.py                 推理 + 判定 + 画框
├── db.py                       SQLite 检测记录
├── model_meta.py               模型与指标展示数据
├── static/app/                 前端单页：index.html / app.css / app.js
├── templates/report.html       A4 检测报告（可打印成 PDF）
├── 系统说明.md                 系统功能、接口、判定规则说明
├── 01_程序源码/
│   └── steel-defect-yolo/      训练脚本与云端配置
├── 02_模型权重/                4 个 .pt 权重，见该目录下 说明.md
├── 03_数据集/                  NEU-DET 原始图像与 YOLO 标签
├── 04_训练结果/                训练曲线、PR 曲线、混淆矩阵、三模型对比
├── 05_系统截图/                检测首页 / 检测结果 / 批量 / 历史 / 统计 / 模型 / 报告
└── README.md                   本文件
```

## 快速开始（只看系统）

需要 Python 3.10+，先装依赖：

```bash
pip install flask ultralytics opencv-python
```

在本目录启动：

```bash
python app.py
```

浏览器打开 http://localhost:5000 。

> 不指定 `--weights` 时会自动找权重：环境变量 `STEEL_WEIGHTS` → 本目录 `02_模型权重/`
> → `D:/yolo/weights/`。换机器或换目录后如果提示找不到，
> 用 `python app.py --weights 路径/best.pt` 显式指定即可。详见 [系统说明.md](系统说明.md)。

## 系统功能

界面是原生 HTML + CSS + JS 的单页应用，数据全部由 `/api/*` 提供。

| 页面 | 路径 | 说明 |
|---|---|---|
| 单张检测 | `/` | 上传图像，出判定 + 标注图 + 缺陷明细表 |
| 批量检测 | `/` 送检面板切到「批量」 | 一次最多 20 张，汇总合格率 |
| 检测历史 | `/#/history` | 所有记录入库，按判定/文件名筛选，可回看 |
| 统计看板 | `/#/stats` | 合格率、近 7 天趋势、缺陷类型分布、判定占比 |
| 模型信息 | `/#/model` | 训练指标、每类 P/R、模型对比 |
| 检测报告 | `/report/<id>` | A4 版式，浏览器直接打印成 PDF |
| 接口 | `POST /api/predict` | 返回 JSON，可被产线程序调用 |

判定规则：

| 条件 | 结论 |
|---|---|
| 没有检出缺陷 | 合格 |
| 最高置信度 ≥ 不合格阈值（默认 0.60） | 不合格 |
| 有检出但低于阈值 | 建议复检 |

## 复现训练

数据集放在 `03_数据集/NEU-DET`，类别配置是工程根目录的 `data.yaml`：

```bash
cd 01_程序源码/steel-defect-yolo
python scripts/train.py --data data.yaml --data-root ../../03_数据集/NEU-DET \
  --model ../../02_模型权重/yolo11s.pt --epochs 150 --batch 32 --imgsz 320 --cache
```

测试集评估：

```bash
python scripts/val.py --weights ../../02_模型权重/best.pt \
  --data-root ../../03_数据集/NEU-DET --split test
```

## 实验结果

三个模型用同一套超参对比，完整数据见 `04_训练结果/comparison.md`。

| 模型 | 参数量 | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| YOLOv8s | 11.14 M | 0.768 | 0.429 | 0.718 | 0.735 |
| **YOLO11s** | **9.43 M** | 0.739 | 0.427 | 0.709 | 0.718 |
| YOLO26s | 9.95 M | 0.758 | 0.436 | 0.755 | 0.703 |

结论：三者差距在 1~3 个百分点，属于单次训练的随机波动范围。YOLO11s 参数量最少，
精度损失可忽略，因此选作部署模型。

已知短板：龟裂（crazing）各类 mAP50 只有 0.26~0.41。这类缺陷是细密网状微裂纹，
在 200×200 灰度图上与轧制纹理高度相似，是 NEU-DET 公认最难的一类。
