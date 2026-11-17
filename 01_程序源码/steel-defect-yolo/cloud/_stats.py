import os

from ultralytics import YOLO

RUNS = {
    "yolov8s": "runs/neu-det-yolov8s/weights/best.pt",
    "yolo11s": "runs/neu-det-yolo11s/weights/best.pt",
    "yolo26s": "runs/neu-det-yolo26s/weights/best.pt",
}

for name, path in RUNS.items():
    m = YOLO(path)
    n = sum(p.numel() for p in m.model.parameters())
    size = os.path.getsize(path) / 1e6
    print(f"{name:9s} params={n/1e6:6.2f}M  file={size:5.1f}MB")
