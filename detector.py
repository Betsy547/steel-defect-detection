#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""推理 + 判定逻辑。

Web 层只负责展示，所有业务规则集中在这里，方便单独测试和复用
（Streamlit 版、脚本版都能直接 import）。
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 图上最多给多少个框写「编号 + 类别 + 置信度」，其余只留编号
MAX_FULL_LABELS = 8

# 画中文标签用的字体，Windows 自带；找不到就退回英文标签
FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)

# 类别英文名 -> 中文名
CLASS_ZH = {
    "crazing": "龟裂",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点",
    "rolled-in_scale": "氧化铁皮压入",
    "scratches": "划痕",
}

# 判定状态 -> 展示文案
STATUS_LABEL = {
    "pass": "合格",
    "review": "建议复检",
    "fail": "不合格",
}


@dataclass
class Defect:
    name: str          # 英文类别名
    name_zh: str       # 中文类别名
    conf: float        # 置信度
    bbox: tuple        # (x1, y1, x2, y2)
    area: float        # 像素面积


@dataclass
class Verdict:
    status: str                    # pass / review / fail
    label: str                     # 中文判定
    reason: str                    # 判定依据，展示在结果页
    defects: list = field(default_factory=list)
    elapsed_ms: float = 0.0
    image_size: tuple = (0, 0)     # (w, h)


def judge(defects: list, conf_high: float) -> tuple:
    """把检测框变成判定结论。

    规则：
        无缺陷                  -> 合格
        最高置信度 >= conf_high -> 不合格
        其余（有框但置信度低）  -> 建议复检
    """
    if not defects:
        return "pass", "未检出表面缺陷"

    top = max(defects, key=lambda d: d.conf)
    n = len(defects)

    if top.conf >= conf_high:
        return "fail", (
            f"检出 {n} 处缺陷，最高置信度 {top.conf:.2f}"
            f"（{top.name_zh}），超过不合格阈值 {conf_high:.2f}"
        )

    return "review", (
        f"检出 {n} 处疑似缺陷，最高置信度 {top.conf:.2f}"
        f"（{top.name_zh}），低于不合格阈值 {conf_high:.2f}，建议人工复检"
    )


def _pick_font(size: int) -> tuple:
    """返回 (字体, 是否支持中文)。"""
    for p in FONT_CANDIDATES:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size), True
            except OSError:
                continue
    return ImageFont.load_default(), False


def _overlaps(a: tuple, b: tuple) -> bool:
    """两个矩形 (x1, y1, x2, y2) 是否相交。"""
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def annotate(image: np.ndarray, defects: list, conf_high: float) -> np.ndarray:
    """在原图上框出缺陷。

    编号与结果表的行号一致（按置信度从高到低）；
    红框 = 置信度达到不合格阈值，橙框 = 低于阈值、建议复检。

    标签会依次尝试框的上方、下方、框内几个位置，避开已经画过的标签；
    实在挤不下就只留编号（完整信息在结果表里）。
    """
    if not defects:
        return image

    h, w = image.shape[:2]
    # 小图放大，避免框和文字在屏幕上糊成一团
    scale = 2 if min(h, w) < 320 else 1
    canvas = image if scale == 1 else cv2.resize(
        image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
    )

    ch, cw = canvas.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    line = max(2, round(min(ch, cw) / 110))
    text_size = max(12, min(26, round(min(ch, cw) / 20)))
    font, cjk = _pick_font(text_size)
    pad = max(3, text_size // 4)

    placed = []  # 已占用的标签矩形，后面的标签要躲开它们

    def measure(text: str) -> tuple:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left + pad * 2, bottom - top + pad * 2, (left, top)

    def spots(box: tuple, tw: float, th: float) -> list:
        x1, y1, x2, y2 = box
        raw = [
            (x1, y1 - th),                # 框上方，左对齐
            (x2 - tw, y1 - th),           # 框上方，右对齐
            (x1, y2),                     # 框下方，左对齐
            (x2 - tw, y2),                # 框下方，右对齐
            (x1 + line, y1 + line),       # 框内左上
            (x2 - tw - line, y1 + line),  # 框内右上
            (x1, y1),                     # 压在上边框上
        ]
        out = []
        for x, y in raw:
            x = min(max(x, 0.0), max(cw - tw, 0.0))
            y = min(max(y, 0.0), max(ch - th, 0.0))
            out.append((x, y))
        return out

    for i, d in enumerate(defects, 1):
        color = (220, 38, 38) if d.conf >= conf_high else (217, 119, 6)
        box = tuple(v * scale for v in d.bbox)
        draw.rectangle(box, outline=color, width=line)

        name = d.name_zh if cjk else d.name
        # 前几处写完整信息；太多框时其余只留编号，免得糊满整张图
        texts = (f"{i} {name} {d.conf:.2f}", str(i))
        if i > MAX_FULL_LABELS:
            texts = (str(i),)

        for text in texts:
            tw, th, (ox, oy) = measure(text)
            spot = None
            for x, y in spots(box, tw, th):
                rect = (x, y, x + tw, y + th)
                if not any(_overlaps(rect, p) for p in placed):
                    spot = (x, y, rect)
                    break
            if spot is None:
                continue

            x, y, rect = spot
            draw.rectangle(rect, fill=color)
            draw.text((x + pad - ox, y + pad - oy), text, font=font,
                      fill=(255, 255, 255))
            placed.append(rect)
            break

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


class Detector:
    """封装 YOLO 模型，对外只暴露 predict。"""

    def __init__(self, weights: str, imgsz: int = 320, device: str = "cpu"):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device

    @property
    def names(self) -> dict:
        return self.model.names

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        conf_high: float = 0.60,
        min_area: float = 0.0,
    ) -> tuple:
        """返回 (Verdict, 画好框的 BGR 图像)。"""
        t0 = time.perf_counter()
        result = self.model.predict(
            image, conf=conf, iou=iou, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )[0]
        elapsed = (time.perf_counter() - t0) * 1000

        h, w = result.orig_shape
        defects = []
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            if area < min_area:
                continue
            name = self.model.names[int(box.cls)]
            defects.append(Defect(
                name=name,
                name_zh=CLASS_ZH.get(name, name),
                conf=float(box.conf),
                bbox=(x1, y1, x2, y2),
                area=area,
            ))

        # 按置信度从高到低排，图中的编号才能和结果表的行号对上
        defects.sort(key=lambda d: d.conf, reverse=True)

        status, reason = judge(defects, conf_high)
        verdict = Verdict(
            status=status,
            label=STATUS_LABEL[status],
            reason=reason,
            defects=defects,
            elapsed_ms=elapsed,
            image_size=(w, h),
        )
        return verdict, annotate(image, defects, conf_high)
