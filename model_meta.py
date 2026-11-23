# -*- coding: utf-8 -*-
"""模型与训练结果的展示数据。

重新训练后只需要改这一个文件，前端各页面自动跟着变。
数字来自 2026-11-04 的云端训练（RTX 4090，150 epochs）。
"""

MODEL = {
    "name": "YOLO11s",
    "framework": "Ultralytics 8.4.202",
    "weights": "best.pt",
    "params_m": 9.43,
    "size_mb": 19.2,
    "imgsz": 320,
    "device": "CPU（本地演示）",
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
    "epochs": 150,
    "best_epoch": 130,
    "batch": 32,
    "patience": 30,
    "optimizer": "Ultralytics 默认（SGD + 自动学习率）",
    "gpu": "RTX 4090 24GB",
    "duration": "约 7 分钟（开启 --cache）",
}

# 测试集（180 张 / 420 个框）最终指标；441 是验证集的框数
TEST = {
    "mAP50": 0.739,
    "mAP50_95": 0.427,
    "precision": 0.709,
    "recall": 0.718,
}

# 每类在测试集上的表现
PER_CLASS = [
    {"name": "crazing", "zh": "龟裂", "map50": 0.263, "p": 0.398, "r": 0.297},
    {"name": "inclusion", "zh": "夹杂", "map50": 0.870, "p": 0.803, "r": 0.789},
    {"name": "patches", "zh": "斑块", "map50": 0.899, "p": 0.844, "r": 0.886},
    {"name": "pitted_surface", "zh": "麻点", "map50": 0.784, "p": 0.826, "r": 0.727},
    {"name": "rolled-in_scale", "zh": "氧化铁皮压入", "map50": 0.671, "p": 0.659, "r": 0.644},
    {"name": "scratches", "zh": "划痕", "map50": 0.949, "p": 0.722, "r": 0.962},
]

# 对比实验（同一套超参，只换骨干）
COMPARE = [
    {"model": "YOLOv8s", "params_m": 11.14, "size_mb": 22.5,
     "map50": 0.768, "map50_95": 0.429, "p": 0.718, "r": 0.735},
    {"model": "YOLO11s", "params_m": 9.43, "size_mb": 19.2,
     "map50": 0.739, "map50_95": 0.427, "p": 0.709, "r": 0.718},
    {"model": "YOLO26s", "params_m": 9.95, "size_mb": 20.3,
     "map50": 0.758, "map50_95": 0.436, "p": 0.755, "r": 0.703},
]

RULES = [
    {"cond": "没有检出任何缺陷", "verdict": "合格", "status": "pass"},
    {"cond": "最高置信度 ≥ 不合格阈值", "verdict": "不合格", "status": "fail"},
    {"cond": "有检出但低于不合格阈值", "verdict": "建议复检", "status": "review"},
]
