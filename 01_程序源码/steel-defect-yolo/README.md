# 钢材表面缺陷检测系统（YOLO）

基于 Ultralytics YOLO 的热轧钢带表面缺陷检测课程设计工程。

**部署方式：云端 GPU 训练 + 本地 CPU 演示。**

---

## 一、整体架构

```
┌───────────── 本地（你的 4060 笔记本）─────────────┐        ┌────── 云端 GPU ──────┐
│                                                   │        │                      │
│  ① 数据整理 / 校验                                 │        │                      │
│       ↓                                           │        │                      │
│  ② 打包上传 ──────────────────────────────────────┼───────→│  ③ 训练              │
│                                                   │        │  ④ 导出 best.pt      │
│  ⑥ Web 系统演示 ←─────────────────────────────────┼────────│  ⑤ 下载权重          │
│                                                   │        │                      │
└───────────────────────────────────────────────────┘        └──────────────────────┘
```

本地不需要 CUDA，CPU 版 PyTorch 就能跑推理和界面。

---

## 二、目录结构

```
steel-defect-yolo/
├── data.yaml                  # 数据集配置（类别 + 相对路径）
├── requirements-local.txt     # 本地演示环境（CPU）
├── requirements-server.txt    # 云端训练环境
├── datasets/                  # 数据集说明（实际数据放在 D:\yolo\datasets）
├── scripts/
│   ├── prepare_dataset.py     # 未划分的 VOC 数据集 → YOLO 格式 + 划分
│   ├── import_neu_det.py      # 已划分的 NEU-DET → YOLO 格式（保留原划分）
│   ├── check_dataset.py       # 数据校验（抽样画框）
│   ├── train.py               # 训练（云端）
│   ├── val.py                 # 评估
│   ├── predict.py             # 推理 + 结构化 JSON 输出（本地）
│   ├── slice_predict.py       # 大图切片推理（钢板产线场景）
│   └── export.py              # 导出 ONNX / TensorRT
├── app/
│   └── streamlit_app.py       # 早期演示界面（备选）
├── cloud/
│   ├── setup.sh               # 服务器一键装环境
│   ├── train.sh               # 服务器一键训练
│   └── README.md              # 上传/下载流程
└── runs/                      # 训练产物（训练时在云端）
```

> Web 检测系统（`app.py`、`detector.py`、`db.py`、前端单页）已放在打包目录根上，
> 与本工程平级，不在这里。

---

## 三、五步工作流

### 第 1 步：本地准备数据

**情况 A：用的是已划分版本**（如 Hugging Face `KeenForgeAI/NEU-DET-corrected`，
目录里已有 `images/{train,val,test}` 和 `annotations/*.xml`）

```bash
python scripts/import_neu_det.py --src D:/yolo/datasets/NEU-DET-raw --dst D:/yolo/datasets/NEU-DET
```

**情况 B：用的是未划分的官方原始包**（只有扁平的 `IMAGES/` + `ANNOTATIONS/`）

```bash
python scripts/prepare_dataset.py --src "C:/path/to/NEU-DET" --dst D:/yolo/datasets/NEU-DET
```

两种方式产出的目录结构一致，都是 `images/{train,val,test}` + `labels/{train,val,test}`。

校验标签有没有错位（**必做**，能省掉后面一半的诡异 bug）：

```bash
python scripts/check_dataset.py --root D:/yolo/datasets/NEU-DET --samples 30
```

### 第 2 步：上传到服务器

见 [cloud/README.md](cloud/README.md)。NEU-DET 只有 ~50MB，几分钟就传完。

### 第 3 步：云端训练

```bash
bash cloud/setup.sh          # 装依赖
bash cloud/train.sh          # 开始训练
```

### 第 4 步：下载权重

把服务器上的 `runs/detect/train/weights/best.pt` 下载到本地：

```
D:/yolo/weights/best.pt
```

### 第 5 步：本地演示

在打包目录根上启动（`app.py` 与 `02_模型权重/` 平级）：

```bash
python app.py --port 5000
```

浏览器打开 `http://localhost:5000`，手机连同一个 WiFi 可以访问 `http://你的电脑IP:5000`。

权重路径可以省略：程序会依次在 `02_模型权重/` 和 `D:/yolo/weights/` 里找
`best.pt`；要指定就加 `--weights 路径/best.pt`。

界面是原生 HTML + CSS + JS 的单页应用（`static/app/`），数据全部走
`/api/*`：检测、批量、历史、统计、模型五个部分，另有可打印的 A4 检测报告。

> 早期的 Streamlit 版本仍保留在 `app/streamlit_app.py`，需要时
> `streamlit run app/streamlit_app.py`（端口 8501）。

---

## 四、常用命令

**单张/批量推理并导出 JSON**

```bash
python scripts/predict.py --weights D:/yolo/weights/best.pt --source test_images/ --out outputs/
```

**大图切片推理**（产线高分辨率钢板图）

```bash
python scripts/slice_predict.py --weights D:/yolo/weights/best.pt --source big.jpg --slice 640 --overlap 0.2
```

**导出 ONNX**

```bash
python scripts/export.py --weights D:/yolo/weights/best.pt --format onnx
```

---

## 五、NEU-DET 参数建议

| 参数 | 建议值 | 原因 |
|---|---|---|
| `--imgsz` | **320** | 原图只有 200×200，640 是浪费算力 |
| `--model` | `yolo11s.pt` | 8GB 显存和云端都合适 |
| `--batch` | 云端 32 / 本地 8 | 200×200 图片显存占用很小 |
| `--epochs` | 150 | 数据集小，容易过拟合，配合 `patience=30` |

---

## 六、常见问题

**Q：训练时提示 `CUDA out of memory`**
A：把 `--batch` 减半，或者 `--imgsz` 降到 256。

**Q：mAP 一直在 0.5 左右上不去**
A：先跑 `check_dataset.py` 看框有没有画偏；再确认 `data.yaml` 里 `names` 的顺序和标注一致。

**Q：本地推理很慢**
A：CPU 推理 yolov8s 大约 50–100ms/张，演示够用。批量处理建议用 `--imgsz 320`。

**Q：报错 `Download failure for Arial.ttf`**
A：ultralytics 画图要用的字体下载失败（国内常见）。直接把系统字体复制过去：

```bash
cp C:/Windows/Fonts/arial.ttf "$APPDATA/Ultralytics/Arial.ttf"
```

**Q：训练产物跑到 D:\yolo\runs 下面去了**
A：这是 ultralytics 的 `runs_dir` 设置在起作用。脚本会把 `--project` 转成绝对路径，
产物就在你指定的位置；想统一放 D 盘就把 `--project` 写成绝对路径。

**Q：换别的数据集**
A：改 `data.yaml` 的 `names` 和 `path`，类别数不需要改代码。
