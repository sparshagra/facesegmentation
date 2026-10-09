"""Segmentation-quality metrics for comparing each pipeline's face mask
against a reference (ground-truth) mask.

Standard mask IoU/Dice score the whole mask area, so a large correct
interior dilutes a genuinely wrong boundary — a landmark-hull mask (Tasks
1/4) and a pixel-level parse (Task 2/3) can disagree along the jawline by
only a few pixels and still read as "very different" or, worse, a mask
that nails 95% of the interior but misses the whole ear can still score a
deceptively high IoU. Boundary IoU and Boundary F1 (BF-score) instead
score agreement specifically along the contour, with an explicit slack
(`dilation_ratio` / `tolerance_px`) for small, harmless offsets.

References:
  Boundary IoU: Cheng, Girshick, Dollar, Kirillov, Girshick. CVPR 2021.
    https://arxiv.org/abs/2103.16562
  Boundary F1 / BF-score: Csurka et al., "What is a good evaluation
    measure for semantic segmentation?", BMVC 2013.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def _to_bool(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        return mask
    return mask > 0


def mask_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """Standard whole-mask IoU. Baseline metric — see module docstring for
    why this alone is misleading for comparing structurally different
    mask sources (polygon hull vs. pixel-level parse vs. SAM mask)."""
    g, p = _to_bool(gt), _to_bool(pred)
    inter = np.logical_and(g, p).sum()
    union = np.logical_or(g, p).sum()
    return float(inter / union) if union > 0 else 1.0


def mask_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Standard Dice / F1 over mask area (2|A∩B| / (|A|+|B|))."""
    g, p = _to_bool(gt), _to_bool(pred)
    inter = np.logical_and(g, p).sum()
    denom = g.sum() + p.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def _boundary_band(mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """The `dilation_px`-wide band of `mask` pixels nearest its own
    contour: mask \\ erode(mask, dilation_px). This is General Boundary
    IoU's G_d / P_d construction."""
    m = _to_bool(mask).astype(np.uint8)
    if dilation_px < 1:
        dilation_px = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
    eroded = cv2.erode(m, kernel, iterations=1)
    band = m.astype(bool) & ~eroded.astype(bool)
    return band


def boundary_iou(gt: np.ndarray, pred: np.ndarray, dilation_ratio: float = 0.02,
                  dilation_px: int | None = None) -> float:
    """Boundary IoU (Cheng et al., CVPR 2021).

    `dilation_ratio`: band width as a fraction of the image diagonal
    (paper default 0.02 = 2%), so it scales with resolution. Pass
    `dilation_px` directly to override with an absolute pixel width
    instead (useful for small crops where 2% of the diagonal rounds to 0).
    A wider band = more tolerance for small boundary offsets.
    """
    h, w = gt.shape[:2]
    if dilation_px is None:
        diag = float(np.hypot(h, w))
        dilation_px = max(1, int(round(dilation_ratio * diag)))
    g_band = _boundary_band(gt, dilation_px)
    p_band = _boundary_band(pred, dilation_px)
    inter = np.logical_and(g_band, p_band).sum()
    union = np.logical_or(g_band, p_band).sum()
    return float(inter / union) if union > 0 else 1.0


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Binary map of contour pixels (1px-wide) via morphological gradient."""
    m = _to_bool(mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
    return gradient > 0


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance_px: int = 3) -> dict:
    """Boundary F1 / BF-score (Csurka et al., BMVC 2013).

    A predicted boundary pixel counts as a match if it lies within
    `tolerance_px` of *any* ground-truth boundary pixel (via a distance
    transform of the GT boundary), and symmetrically for recall. This is
    the more literal "allow a little bit off" metric: `tolerance_px` is
    the exact pixel slack you're granting.

    Returns {"precision", "recall", "f1"}. Both masks empty -> all 1.0;
    exactly one empty -> all 0.0.
    """
    g_b = _boundary_pixels(gt)
    p_b = _boundary_pixels(pred)

    if not g_b.any() and not p_b.any():
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not g_b.any() or not p_b.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # distance_transform_edt(~boundary) gives, at every pixel, the distance
    # to the nearest boundary pixel of that mask.
    dist_to_g = distance_transform_edt(~g_b)
    dist_to_p = distance_transform_edt(~p_b)

    pred_matched = dist_to_g[p_b] <= tolerance_px
    gt_matched = dist_to_p[g_b] <= tolerance_px

    precision = float(pred_matched.mean())
    recall = float(gt_matched.mean())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_all(gt: np.ndarray, pred: np.ndarray, dilation_ratio: float = 0.02,
                 tolerance_px: int = 3) -> dict:
    """Convenience bundle: standard IoU/Dice plus both boundary-tolerant
    metrics, in one manifest-friendly dict."""
    bf = boundary_f1(gt, pred, tolerance_px=tolerance_px)
    return {
        "mask_iou": mask_iou(gt, pred),
        "mask_dice": mask_dice(gt, pred),
        "boundary_iou": boundary_iou(gt, pred, dilation_ratio=dilation_ratio),
        "boundary_precision": bf["precision"],
        "boundary_recall": bf["recall"],
        "boundary_f1": bf["f1"],
    }
