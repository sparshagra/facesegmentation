#!/usr/bin/env python3
"""Aggregate evaluation: for each of the 4 anonymization pipelines, score
predicted masks against the combined_final COCO ground-truth masks
(mask IoU/Dice, Boundary IoU, Boundary F1), merge in Task 5's YOLOX
person-retention numbers, and break every metric down by category
(01_general, 02_tiny_face, ...) as well as overall per pipeline.

Run after run_all.sh has produced results/task{1..5}_*/ for the dataset.

    python scripts/evaluate.py --dataset data/combined_final/combined \
        --results results --output results/eval

Writes:
  results/eval/per_image.csv     -- one row per (pipeline, image)
  results/eval/per_category.csv  -- one row per (pipeline, category)
  results/eval/per_pipeline.csv  -- one row per pipeline (overall)
  results/eval/summary.json      -- same data, nested, for programmatic use
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.coco_gt import load_gt_masks
from common.io_utils import make_image_id
from common.manifest import load_manifest
from common.seg_metrics import compute_all

PIPELINES = [
    "task1_retinaface_mediapipe",
    "task2_scrfd_segface",
    "task3_yolo11n_mobilesam",
    "task4_yolov5n_mediapipe",
]


def load_mask_png(path: Path):
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--output", default=str(ROOT / "results" / "eval"))
    ap.add_argument("--dilation-ratio", type=float, default=0.02)
    ap.add_argument("--tolerance-px", type=int, default=3)
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    results_root = Path(args.results)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt_masks(dataset_root)
    print(f"[evaluate] loaded {len(gt)} GT-annotated images", file=sys.stderr)

    # Task 5 (downstream detector retention) manifest, keyed by
    # f"{pipeline}/{image_id}" exactly as task5's run.py writes it.
    task5_by_key = {}
    task5_manifest_path = results_root / "task5_yolox_eval" / "manifest.jsonl"
    if task5_manifest_path.exists():
        for rec in load_manifest(task5_manifest_path):
            task5_by_key[rec["image_id"]] = rec.get("extra", {})
    else:
        print(f"[evaluate] WARNING: {task5_manifest_path} not found, "
              f"retention columns will be blank", file=sys.stderr)

    per_image_rows = []
    for pipeline in PIPELINES:
        mask_dir = results_root / pipeline / "masks"
        img_manifest_path = results_root / pipeline / "manifest.jsonl"
        if not img_manifest_path.exists():
            print(f"[evaluate] WARNING: {img_manifest_path} not found, skipping {pipeline}",
                  file=sys.stderr)
            continue
        manifest_by_id = {r["image_id"]: r for r in load_manifest(img_manifest_path)}

        for image_id, gt_entry in gt.items():
            rec = manifest_by_id.get(image_id)
            gt_mask = gt_entry["mask"]
            category = gt_entry["category"]
            row = {
                "pipeline": pipeline, "image_id": image_id, "category": category,
                "n_faces_gt": gt_entry["n_faces"],
            }

            if rec is None or rec.get("error"):
                row.update({"n_faces_pred": None, "mask_iou": None, "mask_dice": None,
                            "boundary_iou": None, "boundary_precision": None,
                            "boundary_recall": None, "boundary_f1": None,
                            "error": (rec or {}).get("error", "no manifest record")})
            else:
                row["n_faces_pred"] = rec.get("n_faces")
                pred_mask_path = mask_dir / f"{image_id}.png"
                if gt_entry["n_faces"] == 0 and rec.get("n_faces", 0) == 0:
                    row.update({"mask_iou": 1.0, "mask_dice": 1.0, "boundary_iou": 1.0,
                                "boundary_precision": 1.0, "boundary_recall": 1.0,
                                "boundary_f1": 1.0, "error": None})
                elif not pred_mask_path.exists():
                    # pipeline found 0 faces (no mask saved) but GT has faces, or vice versa
                    import numpy as np
                    pred_mask = np.zeros_like(gt_mask)
                    m = compute_all(gt_mask, pred_mask, dilation_ratio=args.dilation_ratio,
                                     tolerance_px=args.tolerance_px)
                    row.update(m)
                    row["error"] = None
                else:
                    pred_mask = load_mask_png(pred_mask_path)
                    if pred_mask.shape != gt_mask.shape:
                        import cv2
                        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                    m = compute_all(gt_mask, pred_mask, dilation_ratio=args.dilation_ratio,
                                     tolerance_px=args.tolerance_px)
                    row.update(m)
                    row["error"] = None

            t5 = task5_by_key.get(f"{pipeline}/{image_id}", {})
            row["yolox_retention_rate"] = t5.get("retention_rate")
            row["yolox_mean_conf_delta"] = t5.get("mean_conf_delta")
            row["yolox_n_lost_detections"] = t5.get("n_lost_detections")

            if rec is not None:
                timing = rec.get("timing", {})
                row["detect_ms"] = timing.get("detect_ms")
                row["mask_ms"] = timing.get("mask_ms")
                row["anonymize_ms"] = timing.get("anonymize_ms")
                row["total_ms"] = timing.get("total_ms")

            per_image_rows.append(row)

    # --- write per_image.csv ---
    fieldnames = ["pipeline", "image_id", "category", "n_faces_gt", "n_faces_pred",
                  "mask_iou", "mask_dice", "boundary_iou", "boundary_precision",
                  "boundary_recall", "boundary_f1", "yolox_retention_rate",
                  "yolox_mean_conf_delta", "yolox_n_lost_detections",
                  "detect_ms", "mask_ms", "anonymize_ms", "total_ms", "error"]
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_image_rows:
            w.writerow(row)

    # --- aggregate helpers ---
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    metric_cols = ["mask_iou", "mask_dice", "boundary_iou", "boundary_precision",
                   "boundary_recall", "boundary_f1", "yolox_retention_rate",
                   "yolox_mean_conf_delta", "detect_ms", "mask_ms", "anonymize_ms", "total_ms"]

    def aggregate(rows):
        return {col: mean([r.get(col) for r in rows]) for col in metric_cols} | {
            "n_images": len(rows),
            "n_lost_detections_total": sum((r.get("yolox_n_lost_detections") or 0) for r in rows),
        }

    categories = sorted(set(r["category"] for r in per_image_rows))

    per_category_rows = []
    for pipeline in PIPELINES:
        for cat in categories:
            rows = [r for r in per_image_rows if r["pipeline"] == pipeline and r["category"] == cat]
            if not rows:
                continue
            agg = aggregate(rows)
            per_category_rows.append({"pipeline": pipeline, "category": cat, **agg})

    with open(out_dir / "per_category.csv", "w", newline="") as f:
        fn = ["pipeline", "category"] + metric_cols + ["n_images", "n_lost_detections_total"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for row in per_category_rows:
            w.writerow(row)

    per_pipeline_rows = []
    for pipeline in PIPELINES:
        rows = [r for r in per_image_rows if r["pipeline"] == pipeline]
        if not rows:
            continue
        agg = aggregate(rows)
        per_pipeline_rows.append({"pipeline": pipeline, **agg})

    with open(out_dir / "per_pipeline.csv", "w", newline="") as f:
        fn = ["pipeline"] + metric_cols + ["n_images", "n_lost_detections_total"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for row in per_pipeline_rows:
            w.writerow(row)

    summary = {"per_pipeline": per_pipeline_rows, "per_category": per_category_rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[evaluate] wrote {out_dir}/per_image.csv ({len(per_image_rows)} rows)", file=sys.stderr)
    print(f"[evaluate] wrote {out_dir}/per_category.csv ({len(per_category_rows)} rows)", file=sys.stderr)
    print(f"[evaluate] wrote {out_dir}/per_pipeline.csv ({len(per_pipeline_rows)} rows)", file=sys.stderr)
    print(json.dumps(per_pipeline_rows, indent=2))


if __name__ == "__main__":
    main()
