#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测记录存储（SQLite）。

单张、批量、示例、API 四种入口的检测都会落一条记录，
历史 / 统计 / 报告三个页面都从这里取数。
"""

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "detections.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT    NOT NULL,
    filename   TEXT    NOT NULL,
    source     TEXT    NOT NULL DEFAULT 'upload',
    status     TEXT    NOT NULL,
    n_defects  INTEGER NOT NULL DEFAULT 0,
    top_conf   REAL    NOT NULL DEFAULT 0,
    top_class  TEXT    NOT NULL DEFAULT '',
    elapsed_ms REAL    NOT NULL DEFAULT 0,
    width      INTEGER NOT NULL DEFAULT 0,
    height     INTEGER NOT NULL DEFAULT 0,
    conf       REAL    NOT NULL DEFAULT 0,
    conf_high  REAL    NOT NULL DEFAULT 0,
    min_area   REAL    NOT NULL DEFAULT 0,
    orig_url   TEXT    NOT NULL DEFAULT '',
    pred_url   TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_records_ts ON records(ts);
CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);

CREATE TABLE IF NOT EXISTS defects (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    idx       INTEGER NOT NULL,
    name      TEXT    NOT NULL,
    name_zh   TEXT    NOT NULL,
    conf      REAL    NOT NULL,
    x1        REAL    NOT NULL,
    y1        REAL    NOT NULL,
    x2        REAL    NOT NULL,
    y2        REAL    NOT NULL,
    area      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_defects_record ON defects(record_id);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def save(verdict, filename: str, params: dict, orig_url: str, pred_url: str,
         source: str = "upload") -> int:
    """保存一次检测，返回记录 id。"""
    top = max(verdict.defects, key=lambda d: d.conf) if verdict.defects else None

    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO records
               (ts, filename, source, status, n_defects, top_conf, top_class,
                elapsed_ms, width, height, conf, conf_high, min_area,
                orig_url, pred_url)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                filename, source, verdict.status, len(verdict.defects),
                float(top.conf) if top else 0.0,
                top.name_zh if top else "",
                float(verdict.elapsed_ms),
                int(verdict.image_size[0]), int(verdict.image_size[1]),
                float(params.get("conf", 0)), float(params.get("conf_high", 0)),
                float(params.get("min_area", 0)), orig_url, pred_url,
            ),
        )
        rid = cur.lastrowid

        if verdict.defects:
            conn.executemany(
                """INSERT INTO defects
                   (record_id, idx, name, name_zh, conf, x1, y1, x2, y2, area)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [
                    (rid, i, d.name, d.name_zh, float(d.conf),
                     float(d.bbox[0]), float(d.bbox[1]),
                     float(d.bbox[2]), float(d.bbox[3]), float(d.area))
                    for i, d in enumerate(verdict.defects, 1)
                ],
            )

    return rid


def list_records(status: str = None, q: str = None, page: int = 1,
                 per_page: int = 20) -> tuple:
    """返回 (记录列表, 总数)。"""
    where, args = [], []
    if status in ("pass", "review", "fail"):
        where.append("status = ?")
        args.append(status)
    if q:
        where.append("filename LIKE ?")
        args.append(f"%{q}%")

    clause = ("WHERE " + " AND ".join(where)) if where else ""
    page = max(1, page)

    with connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM records {clause}", args
        ).fetchone()["c"]
        rows = conn.execute(
            f"SELECT * FROM records {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*args, per_page, (page - 1) * per_page],
        ).fetchall()

    return [dict(r) for r in rows], total


def get_record(rid: int) -> tuple:
    """返回 (记录, 缺陷列表)；不存在时返回 (None, [])。"""
    with connect() as conn:
        row = conn.execute("SELECT * FROM records WHERE id = ?", (rid,)).fetchone()
        if row is None:
            return None, []
        rows = conn.execute(
            "SELECT * FROM defects WHERE record_id = ? ORDER BY idx", (rid,)
        ).fetchall()

    return dict(row), [dict(r) for r in rows]


def stats() -> dict:
    """统计看板用的数据。"""
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM records").fetchone()["c"]
        by_status = {
            r["status"]: r["c"]
            for r in conn.execute(
                "SELECT status, COUNT(*) AS c FROM records GROUP BY status"
            )
        }
        avg = conn.execute(
            "SELECT AVG(elapsed_ms) AS ms, AVG(n_defects) AS n FROM records"
        ).fetchone()
        classes = [
            dict(r) for r in conn.execute(
                """SELECT name_zh, COUNT(*) AS c,
                          COUNT(DISTINCT record_id) AS records
                   FROM defects GROUP BY name_zh ORDER BY c DESC"""
            )
        ]
        daily = {
            r["d"]: r["c"]
            for r in conn.execute(
                "SELECT substr(ts, 1, 10) AS d, COUNT(*) AS c FROM records GROUP BY d"
            )
        }

    # 最近 7 天（含没有检测的日子）
    today = date.today()
    days = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        key = d.isoformat()
        days.append({"date": key, "label": key[5:], "count": daily.get(key, 0)})

    peak = max([d["count"] for d in days] + [1])
    for d in days:
        d["pct"] = round(d["count"] / peak * 100)

    class_total = max([c["c"] for c in classes] + [1])
    for c in classes:
        c["pct"] = round(c["c"] / class_total * 100)

    reviewed = by_status.get("pass", 0) + by_status.get("review", 0) + by_status.get("fail", 0)

    return {
        "total": total,
        "pass": by_status.get("pass", 0),
        "review": by_status.get("review", 0),
        "fail": by_status.get("fail", 0),
        "pass_rate": (by_status.get("pass", 0) / reviewed * 100) if reviewed else 0.0,
        "avg_ms": avg["ms"] or 0.0,
        "avg_defects": avg["n"] or 0.0,
        "classes": classes,
        "days": days,
    }
