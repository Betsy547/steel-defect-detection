#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flask 检测系统的端到端功能测试。

用 Flask test_client 在进程内发请求，不需要启动端口。
测试数据与数据库都放在临时目录，不会污染项目里的 detections.db 和 static/uploads。

覆盖：
- 页面路由：首页、重定向、报告页
- 接口：模型信息、示例列表、示例检测、单张上传、JSON base64、自定义阈值、
  历史、单条记录、统计
- 异常：缺图片、扩展名不支持、图片损坏、记录不存在、示例不存在、批量超过 20 张
- 批量：2 张、21 张
- 静态资源

输出：
- 04_训练结果/system_test.csv
- 04_训练结果/system_test.md

用法：
    python scripts/system_test.py
"""

import base64
import csv
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

here = Path(__file__).resolve()
pkg_root = here.parents[3]
os.environ.setdefault("STEEL_WEIGHTS", str(pkg_root / "02_模型权重" / "best.pt"))
os.environ.setdefault("STEEL_NO_BROWSER", "1")
sys.path.insert(0, str(pkg_root))

import db  # noqa: E402
import model_meta  # noqa: E402

EXPECTED_MODEL = model_meta.MODEL["name"]  # 部署模型换了，这里自动跟着变

_tmp = tempfile.TemporaryDirectory()
db.DB_PATH = Path(_tmp.name) / "detections_test.db"
db.init()

import app as app_module  # noqa: E402

app_module.UPLOAD_DIR = Path(_tmp.name) / "uploads"
app_module.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app_module.app.config["TESTING"] = True

client = app_module.app.test_client()
img_bytes = (pkg_root / "static" / "samples" / "crazing.jpg").read_bytes()

results = []
record_id = None


def add(cid, name, request_desc, expected, resp, ok, detail=""):
    if resp.is_json:
        payload = resp.get_json() or {}
        suffix = f", ok={payload.get('ok')}"
        if payload.get("error"):
            suffix += f", error={payload['error']}"
        detail = (detail + suffix).strip(", ")
    results.append({
        "id": cid,
        "case": name,
        "request": request_desc,
        "expected": expected,
        "actual": f"{resp.status_code}" + (f"（{detail}）" if detail else ""),
        "result": "PASS" if ok else "FAIL",
    })
    print(f"[{'PASS' if ok else 'FAIL'}] {cid:02d} {name}  ->  {resp.status_code} {detail}")


# 1 首页
r = client.get("/")
add(1, "首页 HTML", "GET /", "200 + HTML", r,
    r.status_code == 200 and b"<html" in r.data.lower(), "HTML 页面")

# 2 历史页重定向
r = client.get("/history")
loc = r.headers.get("Location", "")
add(2, "历史页重定向", "GET /history", "302 -> /#/history", r,
    r.status_code == 302 and "/#/history" in loc, f"Location={loc}")

# 3 模型信息
r = client.get("/api/model")
j = r.get_json() or {}
add(3, "模型信息接口", "GET /api/model", f"200 + {EXPECTED_MODEL}", r,
    r.status_code == 200 and j.get("ok") and j.get("model", {}).get("name") == EXPECTED_MODEL,
    f"name={j.get('model', {}).get('name')}, mAP50={j.get('test_metrics', {}).get('mAP50')}")

# 4 示例列表
r = client.get("/api/samples")
j = r.get_json() or {}
add(4, "示例列表", "GET /api/samples", "200 + 6 张", r,
    r.status_code == 200 and j.get("ok") and len(j.get("samples", [])) == 6,
    f"samples={len(j.get('samples', []))}")

# 5 示例检测
r = client.post("/api/demo/crazing.jpg")
j = r.get_json() or {}
record_id = j.get("record_id")
add(5, "内置示例检测", "POST /api/demo/crazing.jpg", "200 + record_id", r,
    r.status_code == 200 and j.get("ok") and isinstance(record_id, int),
    f"record_id={record_id}, status={j.get('status')}")

# 6 示例不存在
r = client.post("/api/demo/nope.jpg")
j = r.get_json() or {}
add(6, "示例不存在", "POST /api/demo/nope.jpg", "404", r,
    r.status_code == 404 and not j.get("ok"), str(j.get("error", "")))

# 7 单张上传
r = client.post("/api/predict", data={"image": (BytesIO(img_bytes), "sample.jpg")},
                content_type="multipart/form-data")
j = r.get_json() or {}
if not record_id:
    record_id = j.get("record_id")
add(7, "单张图片检测", "POST /api/predict (multipart)", "200 + 判定", r,
    r.status_code == 200 and j.get("ok") and j.get("status") in ("pass", "review", "fail"),
    f"status={j.get('status')}, count={j.get('count')}, {j.get('elapsed_ms')} ms")

# 8 JSON base64
b64 = base64.b64encode(img_bytes).decode()
r = client.post("/api/predict", json={"image_base64": b64, "filename": "b64.jpg"})
j = r.get_json() or {}
add(8, "JSON + base64 检测", "POST /api/predict (json)", "200 + 判定", r,
    r.status_code == 200 and j.get("ok"), f"status={j.get('status')}")

# 9 自定义阈值
r = client.post("/api/predict",
                data={"image": (BytesIO(img_bytes), "thr.jpg"),
                      "conf": "0.20", "conf_high": "0.50"},
                content_type="multipart/form-data")
j = r.get_json() or {}
params = j.get("params", {})
add(9, "自定义阈值生效", "POST /api/predict conf=0.20 conf_high=0.50", "200 + 参数回传", r,
    r.status_code == 200 and j.get("ok")
    and abs(float(params.get("conf", -1)) - 0.20) < 1e-6
    and abs(float(params.get("conf_high", -1)) - 0.50) < 1e-6,
    f"conf={params.get('conf')}, conf_high={params.get('conf_high')}")

# 10 缺少图片
r = client.post("/api/predict", data={})
j = r.get_json() or {}
add(10, "缺少图片字段", "POST /api/predict (empty)", "400", r,
    r.status_code == 400 and "没有选择图片" in str(j.get("error", "")),
    str(j.get("error", "")))

# 11 扩展名不支持
r = client.post("/api/predict", data={"image": (BytesIO(img_bytes), "x.txt")},
                content_type="multipart/form-data")
j = r.get_json() or {}
add(11, "扩展名不支持", "POST /api/predict x.txt", "400", r,
    r.status_code == 400 and "不支持的格式" in str(j.get("error", "")),
    str(j.get("error", "")))

# 12 图片损坏
r = client.post("/api/predict", data={"image": (BytesIO(b"not an image"), "bad.jpg")},
                content_type="multipart/form-data")
j = r.get_json() or {}
add(12, "图片文件损坏", "POST /api/predict bad.jpg", "400", r,
    r.status_code == 400 and "图片解码失败" in str(j.get("error", "")),
    str(j.get("error", "")))

# 13 批量 2 张
r = client.post("/api/predict-batch",
                data={"images": [(BytesIO(img_bytes), "a.jpg"),
                                 (BytesIO(img_bytes), "b.jpg")]},
                content_type="multipart/form-data")
j = r.get_json() or {}
summary = j.get("summary", {})
add(13, "批量 2 张", "POST /api/predict-batch (2)", "200 + total=2", r,
    r.status_code == 200 and j.get("ok") and summary.get("total") == 2
    and len(j.get("results", [])) == 2,
    f"total={summary.get('total')}")

# 14 批量 21 张
files_21 = [(BytesIO(img_bytes), f"f{i}.jpg") for i in range(21)]
r = client.post("/api/predict-batch", data={"images": files_21},
                content_type="multipart/form-data")
j = r.get_json() or {}
add(14, "批量超过上限", "POST /api/predict-batch (21)", "400", r,
    r.status_code == 400 and "最多" in str(j.get("error", "")),
    str(j.get("error", "")))

# 15 历史记录
r = client.get("/api/history")
j = r.get_json() or {}
add(15, "历史记录列表", "GET /api/history", "200 + total>=1", r,
    r.status_code == 200 and j.get("ok") and j.get("total", 0) >= 1,
    f"total={j.get('total')}")

# 16 单条记录
r = client.get(f"/api/record/{record_id}")
j = r.get_json() or {}
add(16, "单条记录查询", f"GET /api/record/{record_id}", "200", r,
    r.status_code == 200 and j.get("ok"), f"record_id={record_id}")

# 17 记录不存在
r = client.get("/api/record/99999999")
j = r.get_json() or {}
add(17, "记录不存在", "GET /api/record/99999999", "404", r,
    r.status_code == 404 and not j.get("ok"), str(j.get("error", "")))

# 18 报告页
r = client.get(f"/report/{record_id}")
add(18, "检测报告页", f"GET /report/{record_id}", "200", r,
    r.status_code == 200 and b"<html" in r.data.lower(), "HTML 报告")

# 19 报告不存在
r = client.get("/report/99999999")
add(19, "报告不存在", "GET /report/99999999", "404", r,
    r.status_code == 404, "")

# 20 统计接口
r = client.get("/api/stats")
j = r.get_json() or {}
add(20, "统计看板接口", "GET /api/stats", "200 + total>=1", r,
    r.status_code == 200 and j.get("ok") and j.get("total", 0) >= 1,
    f"total={j.get('total')}, pass_rate={j.get('pass_rate')}")

# 21 静态示例图
r = client.get("/static/samples/crazing.jpg")
add(21, "静态示例图", "GET /static/samples/crazing.jpg", "200 + image/jpeg", r,
    r.status_code == 200 and r.mimetype == "image/jpeg", r.mimetype)

# 输出（--out-dir 可指定别的目录；省略时写 04_训练结果/）
_out = sys.argv[sys.argv.index("--out-dir") + 1] if "--out-dir" in sys.argv else None
out_dir = Path(_out) if _out else pkg_root / "04_训练结果"
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "system_test.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

passed = sum(1 for r in results if r["result"] == "PASS")
dev = app_module._state.get("device") or "cpu"
lines = [
    "# Flask 检测系统端到端测试\n",
    f"- 用例数：{len(results)}；通过：{passed}；失败：{len(results) - passed}",
    "- 方式：Flask `test_client()` 进程内请求，临时数据库与临时上传目录，不污染项目数据",
    f"- 模型：{EXPECTED_MODEL}；权重：`{Path(os.environ['STEEL_WEIGHTS']).name}`；设备：`{dev}`\n",
    "| # | 用例 | 请求 | 预期 | 实际 | 结果 |",
    "|---:|---|---|---|---|---|",
]
for r in results:
    lines.append(f"| {r['id']} | {r['case']} | `{r['request']}` | {r['expected']} | {r['actual']} | {r['result']} |")
lines.append("")
lines.append("## 说明\n")
lines.append("- 第 7、8、9 条会真实调用模型推理，首次请求包含 predictor 初始化；之后单张的耗时见第 7 条的实际值。")
lines.append("- 第 13、14 条测试批量上限，21 张在上传阶段就被拒绝，不会进入推理。")
lines.append("- 第 18 条报告页依赖第 5 或第 7 条生成的记录 id。")
(out_dir / "system_test.md").write_text("\n".join(lines), encoding="utf-8")

print(f"\n合计 {len(results)} 条，通过 {passed} 条，失败 {len(results) - passed} 条")
print(f"结果已写入：{out_dir}")
if passed != len(results):
    sys.exit(1)
