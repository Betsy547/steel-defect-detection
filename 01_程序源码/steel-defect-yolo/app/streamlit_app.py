#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""钢材表面缺陷检测演示系统（Streamlit）。

启动：
    streamlit run app/streamlit_app.py

功能：单图检测 / 批量检测+CSV导出 / 视频检测 / 模型信息
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

DEFAULT_WEIGHTS = "D:/yolo/weights/best.pt"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]

st.set_page_config(page_title="钢材表面缺陷检测系统", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="正在加载模型…")
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def to_bgr(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detect(model, image: Image.Image, conf: float, iou: float, imgsz: int, device: str):
    """返回 (检测结果对象, 耗时毫秒, 结构化列表)。"""
    t0 = time.perf_counter()
    result = model.predict(
        to_bgr(image), conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False
    )[0]
    elapsed = (time.perf_counter() - t0) * 1000

    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        rows.append({
            "类别": model.names[int(box.cls)],
            "置信度": round(float(box.conf), 4),
            "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2),
            "面积(px)": round((x2 - x1) * (y2 - y1)),
        })
    return result, elapsed, rows


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.title("检测设置")
    weights = st.text_input("模型权重路径", DEFAULT_WEIGHTS)
    conf = st.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05,
                     help="调低会检出更多缺陷但误报增加；工业场景通常宁多勿漏")
    iou = st.slider("NMS IoU", 0.10, 0.90, 0.45, 0.05)
    imgsz = st.select_slider("推理尺寸", options=[256, 320, 416, 640, 800], value=320)

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    device = st.selectbox("推理设备", ["cpu", "0"] if cuda_ok else ["cpu"],
                          help="0 表示第一块 NVIDIA GPU")

    st.divider()
    st.caption("云端训练 · 本地演示")
    st.caption("课程设计 · 基于 Ultralytics YOLO")

if not Path(weights).is_file():
    st.error(f"找不到模型权重：{weights}")
    st.info(
        "请把训练好的 `best.pt` 放到该路径，或在左侧「模型权重路径」里改成实际位置。\n\n"
        "如果还没训练，可以先在服务器上跑 `bash cloud/train.sh`，再把权重下载回来。"
    )
    st.stop()

model = load_model(weights)

st.title("钢材表面缺陷检测系统")
st.caption(f"模型：{Path(weights).name}　类别数：{len(model.names)}　"
           f"设备：{device}")

tab_single, tab_batch, tab_video, tab_info = st.tabs(
    ["单图检测", "批量检测", "视频检测", "模型信息"]
)

# ---------------------------------------------------------------- 单图
with tab_single:
    uploaded = st.file_uploader("上传一张钢板图像", type=IMAGE_TYPES, key="single")

    if uploaded is not None:
        image = Image.open(uploaded)
        result, elapsed, rows = detect(model, image, conf, iou, imgsz, device)

        col1, col2 = st.columns(2)
        col1.image(image, caption="原图")
        col2.image(to_rgb(result.plot()), caption=f"检测结果（{elapsed:.0f} ms）")

        c1, c2, c3 = st.columns(3)
        c1.metric("检出缺陷数", len(rows))
        c2.metric("最高置信度", f"{max((r['置信度'] for r in rows), default=0):.3f}")
        c3.metric("判定", "合格" if not rows else "不合格")

        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.success("未检出缺陷")
    else:
        st.info("上传图片后会自动进行检测。支持 jpg / png / bmp / tif。")

# ---------------------------------------------------------------- 批量
with tab_batch:
    files = st.file_uploader(
        "上传多张图片", type=IMAGE_TYPES, accept_multiple_files=True, key="batch"
    )

    if files:
        st.write(f"已选择 {len(files)} 张图片")
        if st.button("开始批量检测", type="primary"):
            progress = st.progress(0.0)
            all_rows, per_image = [], []

            for i, f in enumerate(files):
                image = Image.open(f)
                _, elapsed, rows = detect(model, image, conf, iou, imgsz, device)
                per_image.append({
                    "图片": f.name, "缺陷数": len(rows),
                    "判定": "合格" if not rows else "不合格",
                    "耗时(ms)": round(elapsed),
                })
                for r in rows:
                    all_rows.append({"图片": f.name, **r})
                progress.progress((i + 1) / len(files))

            df_images = pd.DataFrame(per_image)
            st.subheader("逐图结果")
            st.dataframe(df_images, width="stretch")

            if all_rows:
                df_defects = pd.DataFrame(all_rows)
                st.subheader("缺陷明细")
                st.dataframe(df_defects, width="stretch")

                st.subheader("缺陷类别分布")
                st.bar_chart(df_defects["类别"].value_counts())

                csv = df_defects.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "下载缺陷明细 CSV", csv, "defects.csv", "text/csv"
                )

            total = int(df_images["缺陷数"].sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("图片总数", len(files))
            c2.metric("缺陷总数", total)
            c3.metric("合格率", f"{(df_images['判定'] == '合格').mean():.1%}")

# ---------------------------------------------------------------- 视频
with tab_video:
    video = st.file_uploader("上传视频", type=VIDEO_TYPES, key="video")

    if video is not None:
        st.video(video)

        if st.button("开始检测", type="primary"):
            out_root = Path("outputs/video").resolve()
            out_root.mkdir(parents=True, exist_ok=True)
            input_path = out_root / f"input{Path(video.name).suffix}"
            input_path.write_bytes(video.read())

            with st.spinner("正在逐帧检测，视频越长耗时越久…"):
                model.predict(
                    str(input_path), conf=conf, iou=iou, imgsz=imgsz,
                    device=device, save=True, project=str(out_root),
                    name="pred", exist_ok=True, verbose=False,
                )

            output = out_root / "pred" / input_path.name
            if output.exists():
                st.success("处理完成")
                st.video(str(output))
            else:
                st.error("处理失败，请检查视频格式（建议 mp4 / H.264 编码）")

# ---------------------------------------------------------------- 模型信息
with tab_info:
    st.subheader("模型概览")
    st.write(f"**任务类型**：{model.task}")
    st.write(f"**类别数**：{len(model.names)}")

    names_df = pd.DataFrame(
        [{"索引": i, "类别名": n} for i, n in model.names.items()]
    )
    st.dataframe(names_df, width="stretch")

    results_csv = Path(weights).resolve().parent.parent / "results.csv"
    st.subheader("训练指标")
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]

        def grab(key, default="—"):
            return round(float(last[key]), 4) if key in last else default

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP50", grab("metrics/mAP50(B)"))
        c2.metric("mAP50-95", grab("metrics/mAP50-95(B)"))
        c3.metric("Precision", grab("metrics/precision(B)"))
        c4.metric("Recall", grab("metrics/recall(B)"))

        with st.expander("完整训练曲线数据"):
            st.dataframe(df.tail(20), width="stretch")
    else:
        st.info(
            f"未找到 {results_csv.name}。把训练目录里的 results.csv "
            "和 best.pt 一起下载回来即可显示指标。"
        )

    st.subheader("使用说明")
    st.markdown(
        """
        - **置信度阈值**：工业质检建议调低到 0.15–0.25，宁可误报不要漏检
        - **推理尺寸**：NEU-DET 原图 200×200，320 足够；产线大图建议用
          `scripts/slice_predict.py` 做切片推理
        - **批量检测**：结果可导出 CSV，用于缺陷统计和报表
        """
    )
