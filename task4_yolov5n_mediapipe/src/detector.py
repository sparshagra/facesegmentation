"""YOLOv5n-Face detector — thin wrapper around deepcam-cn/yolov5-face's own
reference detect_face.py logic (letterbox resize, attempt_load,
non_max_suppression_face, scale_coords/scale_coords_landmarks), imported
unmodified from vendor/ so decoding matches the original exactly.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

VENDOR = Path(__file__).resolve().parents[1] / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression_face, scale_coords  # noqa: E402

from detect_face import scale_coords_landmarks  # noqa: E402  (reuse vendor's own helper)


class YoloV5FaceDetector:
    def __init__(self, weight_path: str, device: str = "cuda:0",
                 img_size: int = 640, conf_thres: float = 0.6, iou_thres: float = 0.5):
        self.device = torch.device(device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = attempt_load(weight_path, map_location=self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """Returns list of {box: [x1,y1,x2,y2], score: float, landmarks5: [[x,y]x5]}
        in original image pixel coordinates. Follows vendor detect_face.py exactly:
        resize-to-img_size, letterbox pad, forward, NMS, rescale to original size."""
        orgimg = image_rgb  # already RGB
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.img_size, s=int(self.model.stride.max()))
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img.transpose(2, 0, 1).copy()
        img_t = torch.from_numpy(img).to(self.device).float() / 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        pred = self.model(img_t)[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)
        det = pred[0]

        out = []
        if len(det):
            det = det.clone()
            det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], img0.shape).round()
            det[:, 5:15] = scale_coords_landmarks(img_t.shape[2:], det[:, 5:15], img0.shape).round()
            # img0 may itself be a resized copy of the original (r != 1); map back to
            # original image_rgb coordinates by dividing by the same ratio r.
            for j in range(det.size(0)):
                xyxy = (det[j, :4].view(-1) / r).tolist()
                conf = float(det[j, 4].cpu())
                lm = (det[j, 5:15].view(-1) / r).tolist()
                out.append({
                    "box": xyxy,
                    "score": conf,
                    "landmarks5": [[lm[2 * i], lm[2 * i + 1]] for i in range(5)],
                })
        return out
