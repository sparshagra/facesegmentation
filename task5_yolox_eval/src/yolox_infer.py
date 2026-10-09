"""YOLOX-s person detector — thin wrapper around Megvii's own reference
tools/demo.py Predictor logic (ValTransform preprocessing, exp-driven
conf/nms thresholds, postprocess, ratio-rescale to original coords),
imported unmodified from vendor/ so decoding matches the original exactly.
Filters to COCO class 0 ("person") only — this harness measures whether
each anonymization pipeline degrades a downstream PERSON detector, since
stock YOLOX (COCO-pretrained) has no "face" class to detect faces with.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

VENDOR = Path(__file__).resolve().parents[1] / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.data.datasets import COCO_CLASSES  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

PERSON_CLASS_ID = COCO_CLASSES.index("person")
assert PERSON_CLASS_ID == 0


class YoloxPersonDetector:
    def __init__(self, weight_path: str, device: str = "cuda:0", exp_name: str = "yolox-s"):
        self.device = torch.device(device)
        self.exp = get_exp(exp_name=exp_name)
        self.model = self.exp.get_model()
        ckpt = torch.load(weight_path, map_location="cpu")
        sd = ckpt["model"] if "model" in ckpt else ckpt
        self.model.load_state_dict(sd)
        self.model.eval()
        self.model.to(self.device)
        self.preproc = ValTransform(legacy=False)
        self.confthre = self.exp.test_conf
        self.nmsthre = self.exp.nmsthre
        self.test_size = self.exp.test_size
        self.num_classes = self.exp.num_classes

    @torch.inference_mode()
    def detect_persons(self, image_rgb: np.ndarray) -> list[dict]:
        """Returns list of {box:[x1,y1,x2,y2], score} for COCO 'person'
        detections, in original image pixel coordinates."""
        img_bgr = image_rgb[:, :, ::-1]  # vendor preproc expects BGR (matches cv2.imread)
        h, w = img_bgr.shape[:2]
        ratio = min(self.test_size[0] / h, self.test_size[1] / w)

        img, _ = self.preproc(img_bgr, None, self.test_size)
        img_t = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        outputs = self.model(img_t)
        outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre,
                               class_agnostic=True)
        out = outputs[0]
        if out is None:
            return []
        out = out.cpu()
        bboxes = (out[:, 0:4] / ratio).numpy()
        obj_conf = out[:, 4].numpy()
        cls_conf = out[:, 5].numpy()
        cls_id = out[:, 6].numpy().astype(int)
        scores = obj_conf * cls_conf

        results = []
        for box, score, cid in zip(bboxes, scores, cls_id):
            if cid != PERSON_CLASS_ID:
                continue
            results.append({"box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                             "score": float(score)})
        return results
