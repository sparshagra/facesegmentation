"""Mask construction and anonymization ops shared across tasks.

Per the spec:
  Task 1: MediaPipe 468 landmarks -> convex hull -> Gaussian blur inside hull
  Task 2: SegFace pixel mask (face-region classes) -> Gaussian blur inside mask
  Task 3: MobileSAM box-prompted mask -> radial gradient pixelation
  Task 4: MediaPipe 468 landmarks -> convex hull -> Gaussian blur inside hull

All mask/blur helpers operate on uint8 HxWx3 RGB numpy arrays and HxW
uint8 {0,255} masks, so task code stays a thin wrapper around detector/
mask-model specifics plus a call into this module.
"""
from __future__ import annotations

import cv2
import numpy as np


def mask_from_landmarks(landmarks_xy: np.ndarray, image_shape: tuple[int, int],
                         feather_px: int = 0) -> np.ndarray:
    """Convex hull of 2D landmark points -> binary mask (H, W) uint8 {0,255}."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = landmarks_xy.astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    if feather_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_px)
    return mask


def mask_from_box(box_xyxy, image_shape: tuple[int, int], expand: float = 1.15) -> np.ndarray:
    """Fallback ellipse mask from a detector box when no finer mask/landmarks
    are available. expand grows the box slightly so the blur covers hair/
    chin edges."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * expand, (y2 - y1) * expand
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(bw / 2), int(bh / 2)), 0, 0, 360, 255, -1)
    return mask


def mask_from_binary(pred_mask: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary {0,1}/{0,255}/bool pixel mask to uint8 {0,255}."""
    m = pred_mask
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.max() <= 1:
        m = (m.astype(np.uint8)) * 255
    else:
        m = m.astype(np.uint8)
    return m


def union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        out = cv2.bitwise_or(out, m)
    return out


def gaussian_blur_region(image: np.ndarray, mask: np.ndarray,
                          ksize_frac: float = 0.12, feather_px: int = 3) -> np.ndarray:
    """Heavy Gaussian blur composited into `image` only where `mask` is set,
    with a feathered edge so the anonymized region doesn't have a hard seam.
    ksize_frac is relative to face bbox size (derived from mask extent) so
    blur strength scales with face size rather than being a fixed kernel.
    """
    h, w = image.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image.copy()
    face_w = xs.max() - xs.min() + 1
    face_h = ys.max() - ys.min() + 1
    k = max(3, int(max(face_w, face_h) * ksize_frac))
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(image, (k, k), 0)

    if feather_px > 0:
        soft_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_px).astype(np.float32) / 255.0
    else:
        soft_mask = (mask > 0).astype(np.float32)
    soft_mask = soft_mask[..., None]

    out = image.astype(np.float32) * (1 - soft_mask) + blurred.astype(np.float32) * soft_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def radial_gradient_pixelate(image: np.ndarray, mask: np.ndarray,
                              max_block: int = 24, min_block: int = 4) -> np.ndarray:
    """Radial gradient pixelation: pixelation block size is largest at the
    mask centroid and shrinks toward the mask boundary, so the anonymized
    region fades from heavily blocked at the center to lightly blocked at
    the edge (rather than a uniform block size / hard-edged pixelation).
    """
    h, w = image.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image.copy()
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    max_r = max(np.hypot(x2 - cx, y2 - cy), 1.0)

    out = image.copy()
    # Build a small set of discrete pixelation "rings" (block sizes) and
    # composite from coarsest (center) to finest (edge) so each ring only
    # overwrites pixels in its radius band.
    n_rings = 6
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy) / max_r  # 0 at center, ~1 at edge

    for i in range(n_rings):
        r_lo = i / n_rings
        r_hi = (i + 1) / n_rings
        block = int(max_block - (max_block - min_block) * (i / (n_rings - 1)))
        block = max(block, 1)

        small = cv2.resize(image, (max(w // block, 1), max(h // block, 1)),
                            interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        ring = (mask > 0) & (dist >= r_lo) & (dist < r_hi if i < n_rings - 1 else dist <= 1.0001)
        out[ring] = pixelated[ring]

    # Light feather at the mask boundary so the pixelated region blends in.
    soft_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2).astype(np.float32) / 255.0
    soft_mask = soft_mask[..., None]
    out = image.astype(np.float32) * (1 - soft_mask) + out.astype(np.float32) * soft_mask
    return np.clip(out, 0, 255).astype(np.uint8)
