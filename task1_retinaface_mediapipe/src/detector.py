"""RetinaFace-R50 face detector — thin wrapper around biubug6/Pytorch_Retinaface's
own reference detect.py logic (PriorBox anchors, decode, decode_landm, py_cpu_nms),
imported unmodified from vendor/ so decoding matches the original exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

VENDOR = Path(__file__).resolve().parents[1] / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from data import cfg_re50  # noqa: E402
from layers.functions.prior_box import PriorBox  # noqa: E402
from models.retinaface import RetinaFace  # noqa: E402
from utils.box_utils import decode, decode_landm  # noqa: E402
from utils.nms.py_cpu_nms import py_cpu_nms  # noqa: E402


def _remove_prefix(state_dict, prefix="module."):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(k): v for k, v in state_dict.items()}


class RetinaFaceDetector:
    def __init__(self, weight_path: str, device: str = "cuda:0",
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.4,
                 top_k: int = 5000, keep_top_k: int = 750):
        self.device = torch.device(device)
        self.cfg = cfg_re50
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k

        net = RetinaFace(cfg=self.cfg, phase="test")
        sd = torch.load(weight_path, map_location="cpu")
        sd = _remove_prefix(sd.get("state_dict", sd) if isinstance(sd, dict) else sd)
        net.load_state_dict(sd, strict=False)
        net.eval()
        self.net = net.to(self.device)

    @torch.inference_mode()
    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """Returns list of {box: [x1,y1,x2,y2], score: float, landmarks5: [[x,y]x5]}
        in original image pixel coordinates."""
        img = image_rgb[:, :, ::-1].astype(np.float32)  # RGB -> BGR, matches vendor
        im_h, im_w = img.shape[:2]
        scale = torch.tensor([im_w, im_h, im_w, im_h], dtype=torch.float32, device=self.device)
        img -= (104, 117, 123)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        loc, conf, landms = self.net(img_t)

        priorbox = PriorBox(self.cfg, image_size=(im_h, im_w))
        priors = priorbox.forward().to(self.device)
        boxes = decode(loc.squeeze(0), priors, self.cfg["variance"])
        boxes = (boxes * scale).cpu().numpy()
        scores = conf.squeeze(0)[:, 1].cpu().numpy()

        landms_dec = decode_landm(landms.squeeze(0), priors, self.cfg["variance"])
        scale1 = torch.tensor([im_w, im_h] * 5, dtype=torch.float32, device=self.device)
        landms_dec = (landms_dec * scale1).cpu().numpy()

        inds = np.where(scores > 0.02)[0]  # candidate threshold, matches vendor default
        boxes, landms_dec, scores = boxes[inds], landms_dec[inds], scores[inds]

        order = scores.argsort()[::-1][: self.top_k]
        boxes, landms_dec, scores = boxes[order], landms_dec[order], scores[order]

        dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets, landms_dec = dets[keep], landms_dec[keep]
        dets, landms_dec = dets[: self.keep_top_k], landms_dec[: self.keep_top_k]

        out = []
        for d, lm in zip(dets, landms_dec):
            if d[4] < self.confidence_threshold:
                continue
            out.append({
                "box": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                "score": float(d[4]),
                "landmarks5": lm.reshape(5, 2).tolist(),
            })
        return out
