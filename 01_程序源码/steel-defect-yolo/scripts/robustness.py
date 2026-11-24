#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NEU-DET 测试集鲁棒性压力测试。

固定训练好的模型，只改测试图像，不改标注：
    5 种扰动 × 3 个严重度 + clean 基线
    brightness   变暗（0.8 / 0.6 / 0.4）
    contrast     低对比度（0.8 / 0.6 / 0.4）
    noise        高斯噪声（sigma 8 / 16 / 24）
    motion_blur  水平运动模糊（核 3 / 5 / 7）
    jpeg         JPEG 压缩（质量 50 / 30 / 15）

用法：
    python scripts/robustness.py --data-root D:/yolo/datasets/NEU-DET \
        --runs-dir runs/seed_sweep --models yolov8s,yolo11s,yolo26s \
        --out runs/robustness

输出：
    robustness_runs.csv      每个模型 × 每个条件一行
    robustness_summary.csv   每个模型 × 每种扰动一行
    robustness.md            报告用表格
    robustness_heatmap.png   模型 × 条件 热力图
    robustness_curves.png    严重度-性能曲线
    robustness_examples.png  扰动示例图
"""

import _compat  # noqa: F401
import argparse
import csv
import io
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

TYPE_LABELS = {
    "brightness": "变暗",
    "contrast": "低对比度",
    "noise": "高斯噪声",
    "motion_blur": "运动模糊",
    "jpeg": "JPEG 压缩",
}

# 每个类型三个等级的参数，数值越大越严重
LEVEL_PARAMS = {
    "brightness": [0.8, 0.6, 0.4],
    "contrast": [0.8, 0.6, 0.4],
    "noise": [8, 16, 24],
    "motion_blur": [3, 5, 7],
    "jpeg": [50, 30, 15],
}


def parse_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def load_names(data_yaml):
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return [str(x) for x in names]


def find_test_images(data_root: Path):
    for sub in ("images/test", "test/images", "test"):
        d = data_root / sub
        if d.is_dir():
            imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
            if imgs:
                return d, imgs
    raise SystemExit(f"在 {data_root} 下找不到测试图片")


def find_labels(data_root: Path):
    for sub in ("labels/test", "test/labels"):
        d = data_root / sub
        if d.is_dir():
            return d
    raise SystemExit(f"在 {data_root} 下找不到测试标注")


def motion_kernel(size: int):
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, :] = 1.0 / size
    return k


def perturb(img: Image.Image, kind: str, param, rng: np.random.Generator) -> Image.Image:
    img = img.convert("L")
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(float(param))
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(float(param))
    if kind == "noise":
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + rng.normal(0, float(param), arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    if kind == "motion_blur":
        k = motion_kernel(int(param))
        arr = cv2.filter2D(np.asarray(img, dtype=np.uint8), -1, k)
        return Image.fromarray(arr, mode="L")
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(param))
        buf.seek(0)
        return Image.open(buf).convert("L")
    raise ValueError(kind)


def condition_dir(out: Path, kind: str, level: int):
    return out / "data" / (f"{kind}_L{level}" if kind != "clean" else "clean")


def make_condition(out: Path, kind: str, level: int, images, labels_dir: Path, names, force=False):
    """生成一个条件的 images/test 与 labels/test，返回 data.yaml 路径。"""
    cdir = condition_dir(out, kind, level)
    img_dir = cdir / "images" / "test"
    lbl_dir = cdir / "labels" / "test"
    yaml_path = cdir / "data.yaml"

    if (img_dir.is_dir() and len(list(img_dir.glob("*.png"))) == len(images)
            and lbl_dir.is_dir() and yaml_path.is_file() and not force):
        return yaml_path

    if cdir.exists() and force:
        shutil.rmtree(cdir)
    img_dir.mkdir(parents=True, exist_ok=True)

    if lbl_dir.exists() or lbl_dir.is_symlink():
        if lbl_dir.is_symlink() or lbl_dir.is_file():
            lbl_dir.unlink()
        else:
            shutil.rmtree(lbl_dir)
    lbl_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lbl_dir.symlink_to(labels_dir.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(labels_dir, lbl_dir)

    rng = np.random.default_rng(20261124 + level * 97 + sum(ord(c) for c in kind))
    for i, src in enumerate(images, 1):
        img = Image.open(src).convert("L")
        if kind != "clean":
            img = perturb(img, kind, LEVEL_PARAMS[kind][level - 1], rng)
        img.save(img_dir / f"{src.stem}.png", format="PNG")
        if i % 50 == 0:
            print(f"    {kind} L{level}: {i}/{len(images)}", flush=True)

    cfg = {
        "path": str(cdir.resolve()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path


def resolve_models(args):
    runs_dir = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else None
    models = []
    for spec in parse_list(args.models):
        if spec.endswith(".pt"):
            p = Path(spec).expanduser().resolve()
            run_name = p.parent.parent.name
            if run_name.startswith("seed-") and "-s" in run_name:
                name = run_name.replace("seed-", "").rsplit("-s", 1)[0]
            else:
                name = p.stem
            models.append((name, p))
            continue
        if runs_dir is None:
            raise SystemExit("没有 --runs-dir 时必须给 .pt 路径")
        p = runs_dir / f"seed-{spec}-s{args.seed}" / "weights" / "best.pt"
        if not p.is_file():
            raise SystemExit(f"权重不存在：{p}")
        models.append((spec, p))
    return models


def evaluate(model, yaml_path: Path, args, names):
    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        device=args.device or None,
        plots=False,
        verbose=False,
        workers=0,
        project=str(Path(args.out) / "val"),
        name=yaml_path.parent.name,
        exist_ok=True,
    )
    box = metrics.box
    ap50 = list(box.ap50) if len(box.ap50) else [None] * len(names)
    idx = list(getattr(box, "ap_class_index", range(len(names))))
    per_class = {names[int(c)]: float(v) for c, v in zip(idx, ap50) if v is not None}
    row = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    for cls in names:
        row[f"ap50_{cls}"] = per_class.get(cls)
    return row


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(v, d=4):
    return "" if v in ("", None) else f"{v:.{d}f}"


def make_figures(out: Path, rows, names, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = ["clean"] + [f"{k}_L{l}" for k in LEVEL_PARAMS for l in (1, 2, 3)]
    matrix = []
    for model, _ in models:
        vals = {f"{r['perturbation']}_L{r['level']}" if r["perturbation"] != "clean" else "clean": r["map50"] for r in rows if r["model"] == model}
        matrix.append([vals.get(c, float("nan")) for c in conds])

    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(np.array(matrix, dtype=float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_L", "-L") for c in conds], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m for m, _ in models], fontsize=9)
    for i in range(len(models)):
        for j in range(len(conds)):
            v = matrix[i][j]
            if v == v:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="black" if v < 0.4 else "white")
    fig.colorbar(im, ax=ax, label="mAP50")
    ax.set_title("Robustness stress test: mAP50 under test-time perturbations", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "robustness_heatmap.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, (model, _) in zip(axes, models):
        base = next((r["map50"] for r in rows if r["model"] == model and r["perturbation"] == "clean"), float("nan"))
        ax.axhline(base, color="gray", ls="--", lw=1, label="clean")
        for kind in LEVEL_PARAMS:
            vals = []
            for level in (1, 2, 3):
                hit = next((r["map50"] for r in rows if r["model"] == model and r["perturbation"] == kind and r["level"] == level), np.nan)
                vals.append(hit)
            ax.plot([1, 2, 3], vals, marker="o", label=kind)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("severity")
        ax.set_title(model, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mAP50")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "robustness_curves.png", dpi=300)
    plt.close(fig)


def make_examples(out: Path, sample: Path, level: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ["clean"] + list(LEVEL_PARAMS)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax, kind in zip(axes.ravel(), kinds):
        if kind == "clean":
            path = condition_dir(out, "clean", 0) / "images" / "test" / f"{sample.stem}.png"
            title = "clean"
        else:
            path = condition_dir(out, kind, level) / "images" / "test" / f"{sample.stem}.png"
            title = f"{kind} (severity {level})"
        if path.is_file():
            ax.imshow(Image.open(path).convert("L"), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "robustness_examples.png", dpi=300)
    plt.close(fig)


def build_markdown(args, rows, names, models):
    lines = []
    lines.append("# 鲁棒性压力测试\n")
    lines.append("## 协议\n")
    lines.append("- 模型：训练好的 seed-0 权重；只改测试图像，不改标注和训练")
    lines.append("- 条件：clean 基线 + 5 种扰动 × 3 个严重度")
    lines.append("- 扰动方向：变暗（亮度×0.8/0.6/0.4）、低对比度（×0.8/0.6/0.4）、高斯噪声（σ=8/16/24）、水平运动模糊（核 3/5/7）、JPEG 压缩（质量 50/30/15）")
    lines.append(f"- 测试：imgsz={args.imgsz}，batch={args.batch}，conf=0.001，mAP50 / mAP50-95")
    lines.append("")

    lines.append("## 基线（clean）\n")
    lines.append("| 模型 | mAP50 | mAP50-95 | P | R |")
    lines.append("|---|---:|---:|---:|---:|")
    for model, _ in models:
        r = next((x for x in rows if x["model"] == model and x["perturbation"] == "clean"), None)
        if r:
            lines.append(f"| {model} | {fmt(r['map50'])} | {fmt(r['map50_95'])} | {fmt(r['precision'])} | {fmt(r['recall'])} |")
    lines.append("")

    conds = [("clean", 0)] + [(k, l) for k in LEVEL_PARAMS for l in (1, 2, 3)]
    lines.append("## 全部条件 mAP50\n")
    lines.append("| 条件 | " + " | ".join(m for m, _ in models) + " |")
    lines.append("|---|" + "---:|" * len(models))
    for kind, level in conds:
        label = "clean" if kind == "clean" else f"{TYPE_LABELS[kind]} L{level}"
        vals = []
        for model, _ in models:
            r = next((x for x in rows if x["model"] == model and x["perturbation"] == kind and x["level"] == level), None)
            vals.append(fmt(r["map50"]) if r else "")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## 相对 clean 的 mAP50 下降（%）\n")
    lines.append("| 条件 | " + " | ".join(m for m, _ in models) + " |")
    lines.append("|---|" + "---:|" * len(models))
    for kind, level in conds:
        if kind == "clean":
            continue
        label = f"{TYPE_LABELS[kind]} L{level}"
        vals = []
        for model, _ in models:
            r = next((x for x in rows if x["model"] == model and x["perturbation"] == kind and x["level"] == level), None)
            vals.append(f"{r['drop_pct']:.1f}" if r and r.get("drop_pct") is not None else "")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## 按扰动类型汇总\n")
    lines.append("| 模型 | 扰动 | 三级平均 mAP50 | 最大下降 | 最差条件 |")
    lines.append("|---|---|---:|---:|---|")
    for model, _ in models:
        for kind in LEVEL_PARAMS:
            subset = [r for r in rows if r["model"] == model and r["perturbation"] == kind]
            if not subset:
                continue
            mean = sum(r["map50"] for r in subset) / len(subset)
            worst = max(subset, key=lambda r: r["drop_pct"] if r.get("drop_pct") is not None else -1)
            lines.append(f"| {model} | {TYPE_LABELS[kind]} | {mean:.4f} | {worst['drop_pct']:.1f}% | L{worst['level']} |")
    lines.append("")

    main_model = models[1][0] if len(models) > 1 else models[0][0]
    lines.append(f"## 每类 AP50（{main_model}，严重度 3）\n")
    lines.append("| 扰动 | " + " | ".join(names) + " |")
    lines.append("|---|" + "---:|" * len(names))
    for kind in LEVEL_PARAMS:
        r = next((x for x in rows if x["model"] == main_model and x["perturbation"] == kind and x["level"] == 3), None)
        if not r:
            continue
        lines.append(f"| {TYPE_LABELS[kind]} | " + " | ".join(fmt(r.get(f"ap50_{c}")) for c in names) + " |")
    lines.append("")

    lines.append("## 自动摘要（写报告时请用自己的话重写）\n")
    for model, _ in models:
        subset = [r for r in rows if r["model"] == model and r["perturbation"] != "clean"]
        if not subset:
            continue
        worst = max(subset, key=lambda r: r["drop_pct"] if r.get("drop_pct") is not None else -1)
        best = min(subset, key=lambda r: r["drop_pct"] if r.get("drop_pct") is not None else 999)
        lines.append(f"- **{model}** 最大下降 {worst['drop_pct']:.1f}%（{TYPE_LABELS[worst['perturbation']]} L{worst['level']}，mAP50={worst['map50']:.4f}）；最小下降 {best['drop_pct']:.1f}%（{TYPE_LABELS[best['perturbation']]} L{best['level']}）。")
    lines.append("- 若某一类扰动在 L1 就出现大幅下降，说明模型对该成像变化敏感，报告里应把它写成部署约束，而不是只报平均 mAP。")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="NEU-DET 鲁棒性压力测试")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--data", default=str(ROOT / "data.yaml"))
    ap.add_argument("--runs-dir", default=None, help="seed_sweep 输出目录（含 seed-<model>-s0/weights/best.pt）")
    ap.add_argument("--models", default="yolov8s,yolo11s,yolo26s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--types", default="brightness,contrast,noise,motion_blur,jpeg")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=str(ROOT / "runs" / "robustness"))
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 张测试图做快速验证")
    ap.add_argument("--force", action="store_true", help="重新生成扰动数据并重新评估")
    args = ap.parse_args()

    # 只保留请求的扰动类型，保持 LEVEL_PARAMS 顺序
    keep = parse_list(args.types)
    for k in list(LEVEL_PARAMS):
        if k not in keep:
            del LEVEL_PARAMS[k]
    levels = [int(x) for x in parse_list(args.levels)]

    data_root = Path(args.data_root).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()
    names = load_names(data_yaml)
    img_dir, images = find_test_images(data_root)
    labels_dir = find_labels(data_root)
    if args.limit:
        images = images[: args.limit]

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = resolve_models(args)
    print(f"测试图片: {len(images)} 张，来自 {img_dir}")
    print(f"模型: {[m for m, _ in models]}")
    print(f"输出: {out}")

    conditions = [("clean", 0)] + [(k, l) for k in LEVEL_PARAMS for l in levels]
    for kind, level in conditions:
        yaml_path = make_condition(out, kind, level, images, labels_dir, names, force=args.force)
        print(f"[data] {kind} L{level} -> {yaml_path}")

    rows = []
    for model_name, weights in models:
        print(f"\n=== {model_name} ({weights}) ===")
        model = YOLO(str(weights))
        for kind, level in conditions:
            yaml_path = condition_dir(out, kind, level) / "data.yaml"
            print(f"[eval] {model_name} {kind} L{level}", flush=True)
            metrics = evaluate(model, yaml_path, args, names)
            rows.append({
                "model": model_name,
                "perturbation": kind,
                "level": level,
                **metrics,
            })

    base = {r["model"]: r["map50"] for r in rows if r["perturbation"] == "clean"}
    for r in rows:
        b = base.get(r["model"])
        if b:
            r["delta_map50"] = r["map50"] - b
            r["drop_pct"] = (b - r["map50"]) / b * 100

    fields = ["model", "perturbation", "level", "map50", "map50_95", "precision", "recall",
              "delta_map50", "drop_pct"] + [f"ap50_{c}" for c in names]
    write_csv(out / "robustness_runs.csv", rows, fields)

    summary = []
    for model_name, _ in models:
        for kind in LEVEL_PARAMS:
            subset = [r for r in rows if r["model"] == model_name and r["perturbation"] == kind]
            if not subset:
                continue
            summary.append({
                "model": model_name,
                "perturbation": kind,
                "mean_map50": sum(r["map50"] for r in subset) / len(subset),
                "max_drop_pct": max(r["drop_pct"] for r in subset),
                "worst_level": max(subset, key=lambda r: r["drop_pct"])["level"],
            })
    write_csv(out / "robustness_summary.csv", summary,
              ["model", "perturbation", "mean_map50", "max_drop_pct", "worst_level"])

    md = build_markdown(args, rows, names, models)
    (out / "robustness.md").write_text(md, encoding="utf-8")

    make_figures(out, rows, names, models)
    sample = next((p for p in images if p.stem.startswith("scratches")), images[0])
    try:
        make_examples(out, sample, max(levels))
    except Exception as exc:
        print(f"[warn] 示例图生成失败：{exc}")

    print("\n" + md)
    print(f"\n结果已写入：{out}")


if __name__ == "__main__":
    main()
