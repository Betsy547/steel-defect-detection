# 钢材表面缺陷检测系统（YOLO）

基于 Ultralytics YOLO 的热轧钢带表面缺陷检测课程设计工程。

**部署方式：本地 GPU 训练 + 本地 GPU 演示（部署模型 YOLO26s）；三模型对比、种子重复和鲁棒性实验在云端 GPU 上完成。**

> 系统如何运行、Web 界面与接口说明见根目录 [README.md](../../README.md)
> 和 [系统说明.md](../../系统说明.md)。本文件只讲训练脚本与复现流程。

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

Web 系统有 NVIDIA 显卡时默认用 GPU 推理（RTX 4060 Laptop 上 YOLO26s 单张约 9 ms）；没有显卡时自动改用 CPU（约 25 ms），不需要改代码。

---

## 二、目录结构

```
steel-defect-yolo/
├── data.yaml                  # 数据集配置（类别 + 相对路径）
├── requirements-server.txt    # 云端训练环境（GPU）
├── requirements-web.txt       # Flask Web 系统（GPU 或 CPU 推理）
├── requirements-streamlit.txt # 旧版 Streamlit 演示（CPU）
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

### 环境依赖

| 文件 | 用途 | 对应入口 | 安装 |
|---|---|---|---|
| `requirements-server.txt` | 云端 GPU 训练与评估 | `scripts/train.py`、`scripts/val.py` | `pip install -r requirements-server.txt` |
| `requirements-web.txt` | Flask Web 系统（GPU 或 CPU 推理） | 打包根目录 `app.py` | `pip install -r requirements-web.txt` |
| `requirements-streamlit.txt` | 旧版 Streamlit 演示（CPU） | `app/streamlit_app.py` | `pip install -r requirements-streamlit.txt` |

> `requirements-web.txt` 对应报告里使用的 Flask 系统；`app/streamlit_app.py` 只是早期备选界面，两者不要混用。

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

Flask 系统依赖用 `requirements-web.txt` 安装（见上面的环境依赖表），然后在打包目录根上启动（`app.py` 与 `02_模型权重/` 平级）：

```bash
python app.py --port 5000
```

浏览器打开 `http://localhost:5000`，手机连同一个 WiFi 可以访问 `http://你的电脑IP:5000`。

权重路径可以省略：程序会依次在 `02_模型权重/` 和 `D:/yolo/weights/` 里找
`best.pt`；要指定就加 `--weights 路径/best.pt`。

界面是原生 HTML + CSS + JS 的单页应用（`static/app/`），数据全部走
`/api/*`：检测、批量、历史、统计、模型五个部分，另有可打印的 A4 检测报告。

> 早期的 Streamlit 版本仍保留在 `app/streamlit_app.py`，需要时先
> `pip install -r requirements-streamlit.txt`，再
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
| `--model` | `yolo26s.pt` | 部署模型；8GB 显存和云端都合适 |
| `--batch` | 云端 32 / 本地 8 | 200×200 图片显存占用很小 |
| `--epochs` | 150 | 数据集小，容易过拟合，配合 `patience=30` |

---

## 六、常见问题

**Q：训练时提示 `CUDA out of memory`**
A：把 `--batch` 减半，或者 `--imgsz` 降到 256。

**Q：mAP 一直在 0.5 左右上不去**
A：先跑 `check_dataset.py` 看框有没有画偏；再确认 `data.yaml` 里 `names` 的顺序和标注一致。

**Q：本地推理很慢**
A：有 NVIDIA 显卡时，先确认装的是 CUDA 版 PyTorch（`python -c "import torch; print(torch.cuda.is_available())"` 应输出 True），系统会自动用 GPU，YOLO26s 在 RTX 4060 Laptop 上约 9 ms/张。只能用 CPU 时 YOLO26s 约 25 ms/张（Ryzen 9 8945H），演示也够用。批量处理建议用 `--imgsz 320`。

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

## 许可

本仓库自写代码采用 MIT License，全文见 [LICENSE](../../LICENSE)。
Ultralytics YOLO 依赖为 AGPL-3.0；NEU-DET 数据集版权归原作者，仅用于学习与研究。
