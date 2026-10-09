"""MobileSAM (Tiny-ViT image encoder + SAM mask decoder), box-prompted with
each YOLO11n-Face detection. The image embedding is computed once per image
via set_image(), then each face box is a cheap prompt-only decode.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.anonymize import mask_from_box, union_masks


class MobileSamMasker:
    def __init__(self, weight_path: str, device: str = "cuda:0"):
        vendor = Path(__file__).resolve().parents[1] / "vendor"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        from mobile_sam import SamPredictor, sam_model_registry

        self.device = torch.device(device)
        sam = sam_model_registry["vit_t"](checkpoint=weight_path)
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def masks_for_boxes(self, image_rgb: np.ndarray, boxes: list) -> tuple[np.ndarray, int]:
        h, w = image_rgb.shape[:2]
        if not boxes:
            return np.zeros((h, w), dtype=np.uint8), 0

        self.predictor.set_image(image_rgb, image_format="RGB")
        masks = []
        n_prompted = 0
        for box in boxes:
            box_np = np.array(box, dtype=np.float32)
            try:
                pred_masks, scores, _ = self.predictor.predict(
                    box=box_np, multimask_output=False,
                )
                m = (pred_masks[0].astype(np.uint8)) * 255
                if m.sum() == 0:
                    masks.append(mask_from_box(box, (h, w)))
                else:
                    masks.append(m)
                    n_prompted += 1
            except Exception:
                masks.append(mask_from_box(box, (h, w)))
        self.predictor.reset_image()
        return union_masks(masks, (h, w)), n_prompted
