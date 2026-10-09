"""SCRFD-2.5G face detector — thin wrapper around insightface's own
official `insightface.model_zoo.scrfd.SCRFD` class (same class used by
`FaceAnalysis` internally), so anchor decoding/NMS matches upstream exactly.
"""
from __future__ import annotations

import numpy as np
from insightface.model_zoo.scrfd import SCRFD


class ScrfdDetector:
    def __init__(self, model_path: str, device: str = "cuda:0",
                 det_thresh: float = 0.5, nms_thresh: float = 0.4, input_size: int = 640):
        self.det = SCRFD(model_file=model_path)
        ctx_id = 0 if device.startswith("cuda") else -1
        self.det.prepare(ctx_id, det_thresh=det_thresh, nms_thresh=nms_thresh,
                          input_size=(input_size, input_size))

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """SCRFD (like the rest of insightface) expects BGR input."""
        image_bgr = image_rgb[:, :, ::-1]
        bboxes, kpss = self.det.detect(image_bgr)
        out = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, score = bboxes[i]
            lm = kpss[i].tolist() if kpss is not None else None
            out.append({"box": [float(x1), float(y1), float(x2), float(y2)],
                        "score": float(score), "landmarks5": lm})
        return out
