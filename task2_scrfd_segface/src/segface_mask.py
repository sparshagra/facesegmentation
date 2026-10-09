"""SegFace-Mobile (MobileNetV3 encoder + lightweight transformer decoder,
learnable class tokens) 19-class CelebAMaskHQ face parsing -> pixel mask.

Per-face pipeline: crop the SCRFD box with margin, run SegFace at its
native 512x512 CelebA resolution (exact preprocessing from the vendored
datasets/celebamask_hq.py: BICUBIC resize, ImageNet normalization), argmax
over the 19 classes, map class indices back to an anonymization mask that
covers the face silhouette (skin/brows/eyes/nose/mouth/lips/ears/glasses/
earring) but excludes background/hair/neck/cloth/hat/necklace so hair and
context stay visible while identity-bearing features are covered.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as TF
import torchvision.transforms.functional as VF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.anonymize import mask_from_box, union_masks

# Matches the FaceDecoder token order in vendor/network/models/segface_celeb.py,
# identical to the color_mapping order in vendor/visualize.py.
CELEBA_CLASSES = [
    "background", "neck", "skin", "cloth", "l_ear", "r_ear", "l_brow", "r_brow",
    "l_eye", "r_eye", "nose", "mouth", "l_lip", "u_lip", "hair", "eye_g",
    "hat", "ear_r", "neck_l",
]
FACE_CLASS_NAMES = {"skin", "l_ear", "r_ear", "l_brow", "r_brow", "l_eye", "r_eye",
                     "nose", "mouth", "l_lip", "u_lip", "eye_g", "ear_r"}
FACE_CLASS_IDX = [i for i, n in enumerate(CELEBA_CLASSES) if n in FACE_CLASS_NAMES]

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class SegFaceMasker:
    def __init__(self, weight_path: str, device: str = "cuda:0",
                 resolution: int = 512, margin_frac: float = 0.25):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
        from network import get_model

        self.device = torch.device(device)
        self.resolution = resolution
        self.margin_frac = margin_frac

        model = get_model(backbone="segface_celeb", input_resolution=resolution, model="mobilenet")
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict_backbone"]
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.eval()
        self.model = model.to(self.device)

    def _crop_with_margin(self, image_rgb: np.ndarray, box):
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * self.margin_frac, bh * self.margin_frac
        cx1 = max(0, int(x1 - mx)); cy1 = max(0, int(y1 - my))
        cx2 = min(w, int(x2 + mx)); cy2 = min(h, int(y2 + my))
        if cx2 <= cx1 or cy2 <= cy1:
            return None, None
        return image_rgb[cy1:cy2, cx1:cx2], (cx1, cy1)

    @torch.inference_mode()
    def _parse_crop(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Returns a (crop_h, crop_w) uint8 face-region mask {0,255}."""
        ch, cw = crop_rgb.shape[:2]
        pil = Image.fromarray(crop_rgb)
        resized = VF.resize(pil, size=(self.resolution, self.resolution),
                             interpolation=VF.InterpolationMode.BICUBIC)
        tensor = VF.to_tensor(resized)
        tensor = VF.normalize(tensor, mean=_MEAN, std=_STD).unsqueeze(0).to(self.device)

        logits = self.model(tensor, None, None)  # (1, 19, res, res); labels/dataset unused in forward
        logits = TF.interpolate(logits, size=(ch, cw), mode="bilinear", align_corners=False)
        class_map = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        face_mask = np.isin(class_map, FACE_CLASS_IDX).astype(np.uint8) * 255
        return face_mask

    def masks_for_boxes(self, image_rgb: np.ndarray, boxes: list) -> tuple[np.ndarray, int]:
        h, w = image_rgb.shape[:2]
        masks = []
        n_parsed = 0
        for box in boxes:
            crop, offset = self._crop_with_margin(image_rgb, box)
            if crop is None or crop.size == 0:
                masks.append(mask_from_box(box, (h, w)))
                continue
            crop_mask = self._parse_crop(crop)
            if crop_mask.sum() == 0:  # parser found no face-region pixels; fall back
                masks.append(mask_from_box(box, (h, w)))
                continue
            full_mask = np.zeros((h, w), dtype=np.uint8)
            ox, oy = offset
            ch, cw = crop_mask.shape
            full_mask[oy:oy + ch, ox:ox + cw] = crop_mask
            masks.append(full_mask)
            n_parsed += 1
        if not masks:
            return np.zeros((h, w), dtype=np.uint8), 0
        return union_masks(masks, (h, w)), n_parsed
