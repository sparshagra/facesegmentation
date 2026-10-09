"""MediaPipe Face Mesh (468 3D landmarks) -> convex-hull mask, per detected
face box. The upstream detector (RetinaFace in Task 1, YOLOv5n-Face in Task
4) supplies the face box; we crop with margin and run MediaPipe's own
landmark regressor (BlazeFace-style CNN) on that crop so it only has to
find the one face already localized, then map landmarks back to full-image
pixel coordinates and build the mask via common.anonymize.mask_from_landmarks.

Falls back to an ellipse box-mask if MediaPipe cannot find a face in the
cropped region (rare — e.g. extreme profile/occlusion cases).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.anonymize import mask_from_box, mask_from_landmarks, union_masks


class MediaPipeFaceMesher:
    def __init__(self, model_path: str, min_confidence: float = 0.3, margin_frac: float = 0.35):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.margin_frac = margin_frac

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

    def masks_for_boxes(self, image_rgb: np.ndarray, boxes: list) -> tuple[np.ndarray, int]:
        """boxes: list of [x1,y1,x2,y2] from the upstream detector.
        Returns (union_mask HxW uint8, n_faces_meshed)."""
        h, w = image_rgb.shape[:2]
        masks = []
        n_meshed = 0
        for box in boxes:
            crop, offset = self._crop_with_margin(image_rgb, box)
            if crop is None or crop.size == 0:
                masks.append(mask_from_box(box, (h, w)))
                continue
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
            result = self.landmarker.detect(mp_image)
            if not result.face_landmarks:
                masks.append(mask_from_box(box, (h, w)))
                continue
            ch, cw = crop.shape[:2]
            ox, oy = offset
            pts = np.array([[lm.x * cw + ox, lm.y * ch + oy] for lm in result.face_landmarks[0]],
                            dtype=np.float32)
            masks.append(mask_from_landmarks(pts, (h, w), feather_px=0))
            n_meshed += 1
        if not masks:
            return np.zeros((h, w), dtype=np.uint8), 0
        return union_masks(masks, (h, w)), n_meshed

    def close(self):
        self.landmarker.close()
