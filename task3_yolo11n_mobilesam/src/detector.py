"""YOLO11n-Face detector — thin wrapper around ultralytics.YOLO using the
akanametov/yolo-face released weights (anchor-free YOLO nano, CSP backbone
with C3k2 blocks, decoupled head). Ultralytics' own predict() already
implements the full decode/NMS pipeline, so this wrapper just adapts its
Results object to our common {box, score, landmarks5} record shape (no
landmarks — YOLO11n-face here is box-only; MobileSAM does not need them).
"""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class Yolo11FaceDetector:
    def __init__(self, weight_path: str, device: str = "cuda:0",
                 conf_thres: float = 0.5, iou_thres: float = 0.5):
        self.model = YOLO(weight_path)
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        results = self.model.predict(
            source=image_rgb, device=self.device, conf=self.conf_thres,
            iou=self.iou_thres, verbose=False,
        )
        out = []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        for box, score in zip(boxes, scores):
            out.append({"box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "score": float(score), "landmarks5": None})
        return out
