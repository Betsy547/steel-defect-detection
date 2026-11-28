# -*- coding: utf-8 -*-
"""模型与训练结果的展示数据。

重新训练后只需要改这一个文件，前端各页面自动跟着变。
部署权重（best.pt）：YOLO26s，2026-11-28 在本机 RTX 4060 Laptop 上训练
（150 轮上限，第 83 轮最佳，第 113 轮早停）；指标为该权重在测试集（180 张 / 420 框）上的实测值。
对比实验的三行来自云端 RTX 4090 的第一批对比实验（单次训练），
和 SEED_SWEEP 里的 seed 0、以及部署权重都不是同一次训练。
"""

MODEL = {
    "name": "YOLO26s",
    "framework": "Ultralytics 8.4.202",
    "weights": "best.pt",
    "params_m": 9.95,
    "size_mb": 20.3,
    "imgsz": 320,
    "device": "GPU（RTX 4060 Laptop，本地演示）",
}

DATASET = {
    "name": "NEU-DET",
    "full_name": "NEU Surface Defect Database",
    "origin": "东北大学热轧钢带表面缺陷数据集",
    "classes": 6,
    "resolution": "200 × 200 灰度图",
    "train": 1437,
    "val": 180,
    "test": 180,
    "boxes": 4177,
}

TRAIN = {
    "epochs": 113,
    "best_epoch": 83,
    "batch": 32,
    "patience": 30,
    "optimizer": "Ultralytics 默认（SGD + 自动学习率）",
    "gpu": "RTX 4060 Laptop 8GB",
    "duration": "约 18 分钟（113 轮，第 83 轮后早停）",
}

# 测试集（180 张 / 420 个框）最终指标；441 是验证集的框数
TEST = {
    "mAP50": 0.751,
    "mAP50_95": 0.429,
    "precision": 0.669,
    "recall": 0.725,
}

# 每类在测试集上的表现
PER_CLASS = [
    {"name": "crazing", "zh": "龟裂", "map50": 0.336, "p": 0.379, "r": 0.372},
    {"name": "inclusion", "zh": "夹杂", "map50": 0.851, "p": 0.742, "r": 0.798},
    {"name": "patches", "zh": "斑块", "map50": 0.899, "p": 0.775, "r": 0.873},
    {"name": "pitted_surface", "zh": "麻点", "map50": 0.810, "p": 0.776, "r": 0.727},
    {"name": "rolled-in_scale", "zh": "氧化铁皮压入", "map50": 0.658, "p": 0.599, "r": 0.639},
    {"name": "scratches", "zh": "划痕", "map50": 0.951, "p": 0.743, "r": 0.942},
]

# 对比实验（同一套超参，只换骨干；云端 RTX 4090，第一批单次训练，见上方说明）
COMPARE = [
    {"model": "YOLOv8s", "params_m": 11.14, "size_mb": 22.5,
     "map50": 0.768, "map50_95": 0.429, "p": 0.718, "r": 0.735},
    {"model": "YOLO11s", "params_m": 9.43, "size_mb": 19.2,
     "map50": 0.739, "map50_95": 0.427, "p": 0.709, "r": 0.718},
    {"model": "YOLO26s", "params_m": 9.95, "size_mb": 20.3,
     "map50": 0.758, "map50_95": 0.436, "p": 0.755, "r": 0.703},
]

# 3 个随机种子的重复实验（均值 ± 标准差，云端 RTX 4090）
SEED_SWEEP = [
    {"model": "YOLOv8s", "map50": "0.7559 ± 0.0137", "map50_95": "0.4215 ± 0.0087",
     "p": "0.6821 ± 0.0532", "r": "0.7349 ± 0.0219"},
    {"model": "YOLO11s", "map50": "0.7441 ± 0.0039", "map50_95": "0.4302 ± 0.0033",
     "p": "0.7145 ± 0.0160", "r": "0.7096 ± 0.0140"},
    {"model": "YOLO26s", "map50": "0.7646 ± 0.0094", "map50_95": "0.4411 ± 0.0029",
     "p": "0.7295 ± 0.0138", "r": "0.7177 ± 0.0018"},
]

RULES = [
    {"cond": "没有检出任何缺陷", "verdict": "合格", "status": "pass"},
    {"cond": "最高置信度 ≥ 不合格阈值", "verdict": "不合格", "status": "fail"},
    {"cond": "有检出但低于不合格阈值", "verdict": "建议复检", "status": "review"},
]
