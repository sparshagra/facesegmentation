"""Greedy IoU matching between a "before" (original image) person-detection
set and an "after" (anonymized-image) person-detection set. This is the
core of the downstream-degradation check: for every person YOLOX found in
the original image, did it still find that same person after the image
went through each anonymization pipeline, and did confidence drop?
"""
from __future__ import annotations


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(before: list[dict], after: list[dict], iou_thresh: float = 0.5) -> dict:
    """Greedy best-IoU matching, before -> after.

    Returns {
      "n_before": int, "n_after": int, "n_matched": int,
      "retention_rate": float,           # n_matched / n_before (1.0 if n_before==0)
      "mean_iou_matched": float | None,
      "mean_conf_before_matched": float | None,
      "mean_conf_after_matched": float | None,
      "mean_conf_delta": float | None,   # after - before, negative = confidence dropped
      "lost_detections": [{"box": [...], "score": float}],  # in before, unmatched
      "pairs": [{"before": {...}, "after": {...}, "iou": float}],
    }
    """
    used_after = set()
    pairs = []
    lost = []
    for b in before:
        best_j, best_iou = -1, 0.0
        for j, a in enumerate(after):
            if j in used_after:
                continue
            v = iou(b["box"], a["box"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thresh:
            used_after.add(best_j)
            pairs.append({"before": b, "after": after[best_j], "iou": best_iou})
        else:
            lost.append(b)

    n_before, n_after, n_matched = len(before), len(after), len(pairs)
    result = {
        "n_before": n_before, "n_after": n_after, "n_matched": n_matched,
        "retention_rate": (n_matched / n_before) if n_before > 0 else 1.0,
        "mean_iou_matched": (sum(p["iou"] for p in pairs) / n_matched) if n_matched else None,
        "mean_conf_before_matched": (sum(p["before"]["score"] for p in pairs) / n_matched) if n_matched else None,
        "mean_conf_after_matched": (sum(p["after"]["score"] for p in pairs) / n_matched) if n_matched else None,
        "mean_conf_delta": (sum(p["after"]["score"] - p["before"]["score"] for p in pairs) / n_matched) if n_matched else None,
        "lost_detections": lost,
        "pairs": pairs,
    }
    return result
