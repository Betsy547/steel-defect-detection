#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""钢材表面缺陷检测系统 —— Web 版（Flask）。

启动（在 app.py 所在目录）：
    python app.py
    python app.py --weights D:/yolo/weights/best.pt
    python app.py --device cpu        # 强制用 CPU（默认有 NVIDIA 显卡就用 GPU）

页面（原生 HTML + CSS + JS 单页应用）：
    /               单张 / 批量检测
    /#/history      检测历史
    /#/stats        统计看板
    /#/model        模型信息
    /report/<id>    单张检测报告（可打印成 PDF）

接口：
    POST /api/predict        上传图片返回 JSON
    POST /api/predict-batch  批量检测
    GET  /api/history        检测记录
    GET  /api/stats          统计数据
    GET  /api/model          模型与指标信息
"""

import argparse
import base64
import math
import os
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)

import db
import model_meta
from detector import CLASS_ZH, STATUS_LABEL, Defect, Detector, Verdict, judge

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
SAMPLE_DIR = BASE_DIR / "static" / "samples"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILES = 20
PER_PAGE = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 单次请求上限 64MB

_state = {
    "weights": os.environ.get("STEEL_WEIGHTS", "D:/yolo/weights/best.pt"),
    "imgsz": int(os.environ.get("STEEL_IMGSZ", "320")),
    "device": os.environ.get("STEEL_DEVICE", ""),  # 留空：有 CUDA 用 GPU，否则 CPU
    "detector": None,
}
_lock = threading.Lock()

db.init()


# ---------------------------------------------------------------- 模型

def weights_candidates():
    """按顺序找权重：环境变量 → 同级 weights/ 或 02_模型权重/ → 逐级向上 → D:\\yolo。

    这样 app.py 放在 web/ 里（开发工程）或直接放在打包目录根上都能找到权重。
    """
    here = BASE_DIR
    out = []
    env = os.environ.get("STEEL_WEIGHTS")
    if env:
        out.append(Path(env))
    for base in (here, *here.parents):
        out.append(base / "weights" / "best.pt")
        out.append(base / "02_模型权重" / "best.pt")
    out.append(Path("D:/yolo/weights/best.pt"))
    return out


def pick_weights(prefer: str = None) -> Path:
    if prefer:
        p = Path(prefer)
        if p.is_file():
            return p
        print(f"[warn] 找不到指定权重 {prefer}，改为自动查找")
    for c in weights_candidates():
        if c.is_file():
            return c
    return Path(os.environ.get("STEEL_WEIGHTS", "D:/yolo/weights/best.pt"))


def get_detector():
    """懒加载模型。返回 None 表示权重文件不存在。"""
    with _lock:
        if _state["detector"] is None:
            weights = Path(_state["weights"])
            if not weights.is_file():
                return None
            det = Detector(str(weights), imgsz=_state["imgsz"], device=_state["device"])
            _state["device"] = det.device
            # 预热一次，避免首张图的耗时把 6 秒的模型初始化算进去
            try:
                det.predict(np.zeros((320, 320, 3), np.uint8),
                            conf=0.9, iou=0.45, conf_high=0.6, min_area=0)
            except Exception:
                pass
            _state["detector"] = det
        return _state["detector"]


def model_ready() -> bool:
    return get_detector() is not None


# ---------------------------------------------------------------- 参数

def read_params(src) -> dict:
    """从表单 / JSON 读取阈值参数，做范围保护。"""
    def num(key, default, lo, hi):
        try:
            v = float(src.get(key, default))
        except (TypeError, ValueError):
            return default
        if math.isnan(v):
            return default
        return min(max(v, lo), hi)

    return {
        "conf": num("conf", 0.25, 0.01, 0.95),
        "iou": num("iou", 0.45, 0.10, 0.90),
        "conf_high": num("conf_high", 0.60, 0.05, 0.99),
        "min_area": num("min_area", 0.0, 0.0, 1e9),
    }


# ---------------------------------------------------------------- 推理

def process(raw: bytes, ext: str, params: dict, display_name: str,
            source: str = "upload") -> dict:
    """推理一张图、落盘、入库，返回模板需要的数据。"""
    if ext not in ALLOWED_EXT:
        raise ValueError(f"不支持的格式：{ext or '未知'}")

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片解码失败，请换一张试试")

    uid = uuid.uuid4().hex
    orig_name = f"{uid}{ext}"
    pred_name = f"{uid}_pred.jpg"
    (UPLOAD_DIR / orig_name).write_bytes(raw)

    detector = get_detector()
    verdict, annotated = detector.predict(
        image,
        conf=params["conf"],
        iou=params["iou"],
        conf_high=params["conf_high"],
        min_area=params["min_area"],
    )
    cv2.imwrite(str(UPLOAD_DIR / pred_name), annotated)

    orig_url = url_for("static", filename=f"uploads/{orig_name}")
    pred_url = url_for("static", filename=f"uploads/{pred_name}")
    record_id = db.save(verdict, display_name, params, orig_url, pred_url, source)

    return {
        "verdict": verdict,
        "filename": display_name,
        "orig_url": orig_url,
        "pred_url": pred_url,
        "record_id": record_id,
    }


def handle_upload(file, params, source="upload"):
    return process(
        file.read(),
        Path(file.filename or "").suffix.lower(),
        params,
        file.filename or "未命名",
        source,
    )


def sample_list():
    if not SAMPLE_DIR.is_dir():
        return []
    return [
        {"file": p.name, "zh": CLASS_ZH.get(p.stem, p.stem)}
        for p in sorted(SAMPLE_DIR.glob("*.jpg"))
    ]


def record_to_view(rec: dict, defect_rows: list) -> dict:
    """把数据库里的一条记录还原成模板能用的结构。"""
    defects = [
        Defect(
            name=r["name"], name_zh=r["name_zh"], conf=r["conf"],
            bbox=(r["x1"], r["y1"], r["x2"], r["y2"]), area=r["area"],
        )
        for r in defect_rows
    ]
    status, reason = judge(defects, rec["conf_high"])
    verdict = _verdict(status, reason, defects, rec)
    return {
        "verdict": verdict,
        "filename": rec["filename"],
        "orig_url": rec["orig_url"],
        "pred_url": rec["pred_url"],
        "record_id": rec["id"],
        "recorded_at": rec["ts"],
        "params": {"conf": rec["conf"], "conf_high": rec["conf_high"],
                   "min_area": rec["min_area"]},
    }


def _verdict(status: str, reason: str, defects: list, rec: dict):
    return Verdict(
        status=status,
        label=STATUS_LABEL[status],
        reason=reason,
        defects=defects,
        elapsed_ms=rec["elapsed_ms"],
        image_size=(rec["width"], rec["height"]),
    )


# ---------------------------------------------------------------- 页面

@app.route("/")
def index():
    """前端单页应用（HTML + JS + CSS），数据全部走 /api/*。"""
    return send_from_directory(BASE_DIR / "static" / "app", "index.html")


@app.route("/batch")
def batch():
    """批量检测现在是单页应用里的一个模式。"""
    return redirect("/#/detect")


@app.route("/history")
def history():
    return redirect("/#/history")


@app.route("/history/<int:rid>")
def history_detail(rid):
    return redirect(f"/#/record/{rid}")


@app.route("/stats")
def stats():
    return redirect("/#/stats")


@app.route("/model")
def model_info():
    return redirect("/#/model")


@app.route("/report/<int:rid>")
def report(rid):
    rec, rows = db.get_record(rid)
    if rec is None:
        abort(404)
    data = record_to_view(rec, rows)
    data["model_name"] = model_meta.MODEL["name"]
    data["model_imgsz"] = _state["imgsz"]
    return render_template("report.html", **data)


# ---------------------------------------------------------------- API

def api_error(message: str, code: int = 400):
    return jsonify({"ok": False, "error": message}), code


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """图片 -> JSON 结果。支持 multipart 上传或 JSON base64。"""
    if not model_ready():
        return api_error("模型权重未找到", 503)

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        b64 = payload.get("image_base64") or ""
        if not b64:
            return api_error("缺少字段 image_base64")
        try:
            raw = base64.b64decode(b64.split(",")[-1])
        except Exception:
            return api_error("image_base64 解码失败")
        params = read_params(payload)
        filename = payload.get("filename") or "api"
        ext = Path(filename).suffix.lower() or ".jpg"
    else:
        file = request.files.get("image")
        if not file or not file.filename:
            return api_error("没有选择图片")
        raw = file.read()
        filename = file.filename
        ext = Path(filename).suffix.lower()
        params = read_params(request.form)

    if ext not in ALLOWED_EXT:
        return api_error(f"不支持的格式：{ext or '未知'}")

    source = (request.form.get("source") if not request.is_json
              else (request.get_json(silent=True) or {}).get("source")) or "api"
    if source not in ("upload", "batch", "demo", "api", "web"):
        source = "api"

    try:
        data = process(raw, ext, params, filename, source=source)
    except ValueError as e:
        return api_error(str(e))

    return jsonify(result_json(data, params))


def result_json(data: dict, params: dict) -> dict:
    """一次检测结果的统一 JSON 结构（前端单页应用与开放接口共用）。"""
    v = data["verdict"]
    return {
        "ok": True,
        "record_id": data["record_id"],
        "filename": data["filename"],
        "recorded_at": data.get("recorded_at") or "",
        "status": v.status,
        "label": v.label,
        "reason": v.reason,
        "count": len(v.defects),
        "defects": [
            {
                "name": d.name, "name_zh": d.name_zh,
                "conf": round(d.conf, 4),
                "bbox": [round(x, 1) for x in d.bbox],
                "area": round(d.area, 1),
            }
            for d in v.defects
        ],
        "elapsed_ms": round(v.elapsed_ms, 1),
        "image_size": list(v.image_size),
        "params": {
            "conf": params.get("conf"),
            "conf_high": params.get("conf_high"),
            "min_area": params.get("min_area"),
        },
        "orig_url": data["orig_url"],
        "annotated_url": data["pred_url"],
    }


@app.route("/api/demo/<name>", methods=["POST"])
def api_demo(name):
    """用内置示例图检测（前端示例按钮）。"""
    path = SAMPLE_DIR / name
    if path.parent != SAMPLE_DIR or not path.is_file():
        return api_error("示例图不存在", 404)
    if not model_ready():
        return api_error("模型权重未找到", 503)

    params = read_params(request.args)
    data = process(
        path.read_bytes(), path.suffix.lower(), params,
        f"示例 · {CLASS_ZH.get(path.stem, path.stem)}", source="demo",
    )
    return jsonify(result_json(data, params))


@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    """批量检测：一次最多 20 张，逐张返回结果，出错不影响其它张。"""
    if not model_ready():
        return api_error("模型权重未找到", 503)

    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        return api_error("没有选择图片")
    if len(files) > MAX_FILES:
        return api_error(f"一次最多 {MAX_FILES} 张")

    params = read_params(request.form)
    results, errors = [], []
    for f in files:
        try:
            results.append(handle_upload(f, params, source="batch"))
        except ValueError as e:
            errors.append(f"{f.filename}：{e}")

    counts = {"pass": 0, "review": 0, "fail": 0}
    for r in results:
        counts[r["verdict"].status] += 1
    total = len(results)

    return jsonify({
        "ok": True,
        "results": [result_json(r, params) for r in results],
        "errors": errors,
        "summary": {
            "total": total,
            "pass": counts["pass"],
            "review": counts["review"],
            "fail": counts["fail"],
            "pass_rate": round(counts["pass"] / total * 100, 1) if total else 0.0,
        },
    })


@app.route("/api/samples")
def api_samples():
    return jsonify({"ok": True, "samples": sample_list()})


@app.route("/api/history")
def api_history():
    status = request.args.get("status") or None
    q = (request.args.get("q") or "").strip() or None
    page = request.args.get("page", type=int, default=1)

    records, total = db.list_records(status, q, page=page, per_page=PER_PAGE)
    return jsonify({
        "ok": True,
        "records": records,
        "total": total,
        "page": page,
        "pages": max(1, math.ceil(total / PER_PAGE)),
        "per_page": PER_PAGE,
        "status_label": STATUS_LABEL,
    })


@app.route("/api/record/<int:rid>")
def api_record(rid):
    rec, rows = db.get_record(rid)
    if rec is None:
        return api_error("记录不存在", 404)
    data = record_to_view(rec, rows)
    payload = result_json(data, data["params"])
    payload["recorded_at"] = rec["ts"]
    payload["source"] = rec["source"]
    return jsonify(payload)


@app.route("/api/stats")
def api_stats():
    return jsonify({"ok": True, **db.stats(), "status_label": STATUS_LABEL})


@app.route("/api/model")
def api_model():
    ready = model_ready()  # 先触发模型加载，下面读到的才是实际使用的设备
    return jsonify({
        "ok": True,
        "model": model_meta.MODEL,
        "dataset": model_meta.DATASET,
        "train": model_meta.TRAIN,
        "test_metrics": model_meta.TEST,
        "per_class": model_meta.PER_CLASS,
        "compare": model_meta.COMPARE,
        "seed_sweep": model_meta.SEED_SWEEP,
        "classes": CLASS_ZH,
        "judge_rules": model_meta.RULES,
        "weights": _state["weights"],
        "imgsz": _state["imgsz"],
        "device": _state["device"] or "cpu",
        "ready": ready,
    })


# ---------------------------------------------------------------- 错误处理

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return api_error("接口不存在", 404)
    return render_template("error.html", message="页面不存在。"), 404


@app.errorhandler(413)
def too_large(e):
    if request.path.startswith("/api/"):
        return api_error("文件太大，单次上限 64MB", 413)
    return render_template("error.html", message="文件太大，单次上传上限 64MB。"), 413


@app.errorhandler(500)
def server_error(e):
    if request.path.startswith("/api/"):
        return api_error("服务器内部错误", 500)
    return render_template("error.html", message="服务器内部错误。"), 500


def open_browser(url: str, delay: float = 1.2) -> None:
    """等服务起来后自动打开浏览器。想关掉就设环境变量 STEEL_NO_BROWSER=1。"""
    def run():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="钢材表面缺陷检测系统")
    ap.add_argument("--weights", default=None,
                    help="权重路径，省略时自动在工程内和打包目录里查找 best.pt")
    ap.add_argument("--imgsz", type=int, default=_state["imgsz"])
    ap.add_argument("--device", default=_state["device"],
                    help="推理设备：0 为第一块 GPU，cpu 为 CPU；省略时有 GPU 就用 GPU")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    weights = pick_weights(args.weights)
    _state["weights"] = str(weights)
    _state["imgsz"] = args.imgsz
    _state["device"] = args.device

    print(f"模型权重：{weights}")
    if not weights.is_file():
        print("         （文件不存在，页面能打开，但检测会提示找不到权重）")
    else:
        model_ready()  # 启动时就加载模型并预热，第一个请求不用等
    if _state["device"] == "cpu":
        print("推理设备：CPU")
    else:
        import torch
        idx = int(_state["device"]) if str(_state["device"]).isdigit() else 0
        name = torch.cuda.get_device_name(idx) if torch.cuda.is_available() else "GPU"
        print(f"推理设备：GPU {idx}（{name}）")
    print(f"推理尺寸：{args.imgsz}")
    print(f"访问地址：http://localhost:{args.port}")
    if not os.environ.get("STEEL_NO_BROWSER"):
        open_browser(f"http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
