#!/usr/bin/env python3
"""Per-face TP-matched segmentation evaluation (IoU >= iou_thresh box match,
confidence >= conf_thresh), the same methodology as a standard detection+
segmentation benchmark report: a GT face only contributes to the quality
metrics once it has a matched predicted detection; unmatched GT faces are
tracked separately as recall, not folded into the mask-quality average.

For each matched (GT face, predicted detection) pair, both masks are
cropped to the shared region (GT box union predicted box, expanded by a
margin) before scoring — an approximation of a true per-instance mask
comparison, since each pipeline saves one union mask per image rather than
one mask per detected face. This is exact for images with one face, and
accurate for images with well-separated faces; on genuinely overlapping
faces (some 05_occlusion / 06_multiface_mixed_scale images) a neighboring
face's mask can leak into the crop, which is a known limitation of this
approximation, called out in the report rather than hidden.

    python scripts/evaluate_faces.py --dataset data/combined_final/combined \
        --results results --output results/eval_faces --iou-thresh 0.50 --conf-thresh 0.40

Writes results/eval_faces/{per_face.csv, per_pipeline.csv, per_category.csv, summary.json}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.coco_gt import load_gt_masks
from common.manifest import load_manifest
from common.seg_metrics import compute_all

PIPELINES = [
    "task1_retinaface_mediapipe",
    "task2_scrfd_segface",
    "task3_yolo11n_mobilesam",
    "task4_yolov5n_mediapipe",
]


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crop_pair(gt_mask, pred_mask, box_a, box_b, image_shape, margin_frac=0.15):
    h, w = image_shape
    x1 = min(box_a[0], box_b[0]); y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2]); y2 = max(box_a[3], box_b[3])
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin_frac, bh * margin_frac
    cx1 = max(0, int(x1 - mx)); cy1 = max(0, int(y1 - my))
    cx2 = min(w, int(x2 + mx)); cy2 = min(h, int(y2 + my))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None
    return gt_mask[cy1:cy2, cx1:cx2], pred_mask[cy1:cy2, cx1:cx2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--output", default=str(ROOT / "results" / "eval_faces"))
    ap.add_argument("--iou-thresh", type=float, default=0.50)
    ap.add_argument("--conf-thresh", type=float, default=0.40)
    ap.add_argument("--dilation-ratio", type=float, default=0.02)
    ap.add_argument("--tolerance-px", type=int, default=3)
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    results_root = Path(args.results)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt_masks(dataset_root)
    total_gt_faces = sum(v["n_faces"] for v in gt.values())
    print(f"[evaluate_faces] {len(gt)} images, {total_gt_faces} GT faces, "
          f"IoU>={args.iou_thresh} conf>={args.conf_thresh}", file=sys.stderr)

    per_face_rows = []
    for pipeline in PIPELINES:
        mask_dir = results_root / pipeline / "masks"
        manifest_path = results_root / pipeline / "manifest.jsonl"
        if not manifest_path.exists():
            print(f"[evaluate_faces] WARNING: {manifest_path} missing, skipping {pipeline}", file=sys.stderr)
            continue
        manifest_by_id = {r["image_id"]: r for r in load_manifest(manifest_path)}

        for image_id, gt_entry in gt.items():
            rec = manifest_by_id.get(image_id)
            category = gt_entry["category"]
            h, w = gt_entry["height"], gt_entry["width"]

            pred_mask = None
            pred_dets = []
            if rec is not None and not rec.get("error"):
                pred_dets = [d for d in rec.get("detections", []) if d["score"] >= args.conf_thresh]
                mask_path = mask_dir / f"{image_id}.png"
                if mask_path.exists():
                    pred_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if pred_mask is not None and pred_mask.shape != (h, w):
                        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            used_pred = set()
            for face_idx, inst in enumerate(gt_entry["instances"]):
                best_j, best_iou = -1, 0.0
                for j, d in enumerate(pred_dets):
                    if j in used_pred:
                        continue
                    v = box_iou(inst["box_xyxy"], d["box_xyxy"])
                    if v > best_iou:
                        best_iou, best_j = v, j

                row = {"pipeline": pipeline, "image_id": image_id, "category": category,
                       "face_idx": face_idx, "matched": False, "match_iou": best_iou,
                       "match_conf": None, "mask_iou": None, "mask_dice": None,
                       "boundary_iou": None, "boundary_precision": None,
                       "boundary_recall": None, "boundary_f1": None}

                if best_j >= 0 and best_iou >= args.iou_thresh and pred_mask is not None:
                    used_pred.add(best_j)
                    row["matched"] = True
                    row["match_conf"] = pred_dets[best_j]["score"]
                    gt_crop, pred_crop = crop_pair(inst["mask"], pred_mask, inst["box_xyxy"],
                                                    pred_dets[best_j]["box_xyxy"], (h, w))
                    if gt_crop is not None:
                        m = compute_all(gt_crop, pred_crop, dilation_ratio=args.dilation_ratio,
                                         tolerance_px=args.tolerance_px)
                        row.update(m)
                per_face_rows.append(row)

    fieldnames = ["pipeline", "image_id", "category", "face_idx", "matched", "match_iou",
                  "match_conf", "mask_iou", "mask_dice", "boundary_iou", "boundary_precision",
                  "boundary_recall", "boundary_f1"]
    with open(out_dir / "per_face.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_face_rows:
            w.writerow(row)

    metric_cols = ["mask_iou", "mask_dice", "boundary_iou", "boundary_precision",
                   "boundary_recall", "boundary_f1"]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def aggregate(rows):
        matched = [r for r in rows if r["matched"]]
        agg = {c: mean([r[c] for r in matched]) for c in metric_cols}
        agg["n_gt_faces"] = len(rows)
        agg["n_matched_tp"] = len(matched)
        agg["recall"] = (len(matched) / len(rows)) if rows else None
        return agg

    per_pipeline_rows = []
    for pipeline in PIPELINES:
        rows = [r for r in per_face_rows if r["pipeline"] == pipeline]
        if not rows:
            continue
        per_pipeline_rows.append({"pipeline": pipeline, **aggregate(rows)})

    categories = sorted(set(r["category"] for r in per_face_rows))
    per_category_rows = []
    for pipeline in PIPELINES:
        for cat in categories:
            rows = [r for r in per_face_rows if r["pipeline"] == pipeline and r["category"] == cat]
            if not rows:
                continue
            per_category_rows.append({"pipeline": pipeline, "category": cat, **aggregate(rows)})

    with open(out_dir / "per_pipeline.csv", "w", newline="") as f:
        fn = ["pipeline"] + metric_cols + ["n_gt_faces", "n_matched_tp", "recall"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for row in per_pipeline_rows:
            w.writerow(row)

    with open(out_dir / "per_category.csv", "w", newline="") as f:
        fn = ["pipeline", "category"] + metric_cols + ["n_gt_faces", "n_matched_tp", "recall"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for row in per_category_rows:
            w.writerow(row)

    summary = {"per_pipeline": per_pipeline_rows, "per_category": per_category_rows,
               "iou_thresh": args.iou_thresh, "conf_thresh": args.conf_thresh,
               "total_gt_faces": total_gt_faces}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[evaluate_faces] wrote {out_dir}/per_face.csv ({len(per_face_rows)} rows)", file=sys.stderr)
    print(json.dumps(per_pipeline_rows, indent=2))


if __name__ == "__main__":
    main()
