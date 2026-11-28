#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 Edge 无头浏览器 + CDP 给系统截图（Windows 专用）。

    python scripts/capture_screens.py --base http://127.0.0.1:5000 --out ../05_系统截图

需要 Edge（Windows 自带）和 websocket-client：

    pip install websocket-client

脚本会启动一个独立的无头 Edge，逐页截图后自动关闭，不影响正在用的浏览器。
"""
import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
PORT = 9223
VIEW = (1440, 900)

# 内置示例图所在目录（打包目录 static/samples/）：scripts/ → steel-defect-yolo/ → 01_程序源码/ → 打包目录
SAMPLE_DIR = Path(__file__).resolve().parents[3] / "static" / "samples"

# 批量检测页没有独立地址，用页面里的函数直接灌入示例图再点检测
BATCH_JS = """
(async () => {
  const mk = async (n) => {
    const b = await (await fetch('/static/samples/' + n)).blob();
    return new File([b], n, { type: 'image/jpeg' });
  };
  state.mode = 'batch';
  renderDetect();
  acceptFiles([
    await mk('patches.jpg'), await mk('scratches.jpg'),
    await mk('inclusion.jpg'), await mk('pitted_surface.jpg'),
    await mk('crazing.jpg'), await mk('rolled-in_scale.jpg'),
  ]);
  document.querySelector('[data-act="run"]').click();
})()
"""


def find_edge():
    for p in EDGE_CANDIDATES:
        if Path(p).is_file():
            return p
    raise SystemExit("找不到 Edge，请手动修改 EDGE_CANDIDATES")


WAIT_IMAGES = """
(async () => {
  await Promise.all([...document.images].map((i) => (
    i.decode ? i.decode().catch(() => {}) : Promise.resolve()
  )));
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
})()
"""


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=60)
        self.mid = 0

    def send(self, method, params=None):
        self.mid += 1
        self.ws.send(json.dumps({"id": self.mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def wait_port(url, tries=60):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                return json.load(r)
        except Exception:
            time.sleep(0.25)
    raise SystemExit(f"连不上 {url}")


def new_record(base: str, sample: str) -> int:
    """用内置示例图做一次单张检测，返回新记录的 id。"""
    boundary = "----steelscreen"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{sample}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + open(SAMPLE_DIR / sample, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{base}/api/predict", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    if not data.get("ok") or "record_id" not in data:
        raise SystemExit(f"单张检测失败，拿不到记录 id：{data}")
    return int(data["record_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--out", default="screenshots")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    profile = Path(tempfile.gettempdir()) / "steel-shot-profile"
    shutil.rmtree(profile, ignore_errors=True)

    edge = subprocess.Popen(
        [
            find_edge(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--no-first-run", "--no-default-browser-check",
            f"--user-data-dir={profile}", f"--remote-debugging-port={PORT}",
            f"--window-size={VIEW[0]},{VIEW[1]}", "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        targets = wait_port(f"http://127.0.0.1:{PORT}/json")
        page = next(t for t in targets if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Emulation.setDeviceMetricsOverride", {
            "width": VIEW[0], "height": VIEW[1], "deviceScaleFactor": 1, "mobile": False,
        })

        def shot(name, url, js=None, settle=2.4, full=True):
            cdp.send("Page.navigate", {"url": args.base + url})
            time.sleep(settle)
            if js:
                cdp.send("Runtime.evaluate", {
                    "expression": js, "awaitPromise": True, "returnByValue": True,
                })
                time.sleep(settle)
            # 等所有图片解码完再截，否则全页截图偶尔会拍到没画出来的图
            cdp.send("Runtime.evaluate", {
                "expression": WAIT_IMAGES, "awaitPromise": True, "returnByValue": True,
            })
            time.sleep(0.3)
            params = {"format": "jpeg", "quality": 92}
            if full:
                size = cdp.send("Page.getLayoutMetrics")["cssContentSize"]
                params["captureBeyondViewport"] = True
                params["clip"] = {
                    "x": 0, "y": 0,
                    "width": size["width"], "height": size["height"], "scale": 1,
                }
            data = cdp.send("Page.captureScreenshot", params)["data"]
            path = out / f"{name}.jpg"
            path.write_bytes(base64.b64decode(data))
            print("saved", path)

        # 结果页和报告页需要一条真实记录：先用示例图做一次单张检测，拿到它的 id
        record_id = new_record(args.base, "crazing.jpg")
        print(f"单张检测记录 id = {record_id}")

        shot("01-home", "/", settle=3.0)
        shot("02-result", f"/#/record/{record_id}")
        shot("03-batch", "/#/detect", js=BATCH_JS, settle=4.5)
        shot("04-history", "/#/history")
        shot("05-stats", "/#/stats")
        shot("06-model", "/#/model")
        shot("07-report", f"/report/{record_id}", settle=2.0)

        cdp.close()
    finally:
        edge.terminate()
        try:
            edge.wait(timeout=8)
        except Exception:
            edge.kill()


if __name__ == "__main__":
    main()
