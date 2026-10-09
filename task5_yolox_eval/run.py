#!/usr/bin/env python3
"""Task 5: downstream-detector degradation check.

Runs YOLOX-s (official COCO weights, PERSON class only — stock YOLOX has
no "face" class, see README) on the original images and on each of the 4
anonymization pipelines' output images, then IoU-matches before/after
person detections per image. This answers: does anonymizing faces cause a
downstream detector to lose people it used to find, or lose confidence on
people it still finds?

    python run.py --input ../data --output ../results/task5_yolox_eval \
        --device cuda:0 --limit 120

Auto-discovers pipeline output dirs at ../results/task{1,2,3,4}_*/images/,
so it should be run only after those 4 tasks have already produced output
for the same input dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guards
from common.cli import build_arg_parser
from common.io_utils import list_images, load_image_rgb, make_image_id
from common.manifest import ImageRecord, ManifestWriter
from common.timing import timer_ms

TASK = "task5_yolox_eval"
HERE = Path(__file__).resolve().parent
FACEBLUR_ROOT = HERE.parent
WEIGHTS_DIR = FACEBLUR_ROOT / "weights"
RESULTS_ROOT = FACEBLUR_ROOT / "results"

# YOLOX-s (~34M params, fp32). Generous margin kept since VRAM here is
# shared with other users' processes.
REQUIRED_VRAM_MB = 1200.0

PIPELINE_DIRS = [
    "task1_retinaface_mediapipe",
    "task2_scrfd_segface",
    "task3_yolo11n_mobilesam",
    "task4_yolov5n_mediapipe",
]


def main():
    p = build_arg_parser(TASK)
    p.add_argument("--iou-thresh", type=float, default=0.5,
                    help="IoU threshold for matching a before/after person detection")
    args = p.parse_args()

    guards.print_preflight(TASK, REQUIRED_VRAM_MB if args.device.startswith("cuda") else None,
                            args.min_disk_gb, args.output, args.device)

    from src.compare import match_detections
    from src.yolox_infer import YoloxPersonDetector

    detector = YoloxPersonDetector(str(WEIGHTS_DIR / "yolox_s.pth"), device=args.device)

    pipelines = []
    for name in PIPELINE_DIRS:
        img_dir = RESULTS_ROOT / name / "images"
        if img_dir.is_dir():
            pipelines.append((name, img_dir))
        else:
            print(f"[{TASK}] WARNING: {img_dir} not found, skipping pipeline {name} "
                  f"(run that task first)", file=sys.stderr)

    if not pipelines:
        print(f"[{TASK}] no pipeline outputs found under {RESULTS_ROOT}; "
              f"run tasks 1-4 before task 5.", file=sys.stderr)
        sys.exit(1)

    images = list_images(args.input, limit=args.limit)
    print(f"[{TASK}] {len(images)} original images x {len(pipelines)} pipelines "
          f"= {len(images) * len(pipelines)} comparisons", file=sys.stderr)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # per-pipeline running totals for the final summary
    agg = {name: {"n_images": 0, "n_before": 0, "n_matched": 0,
                   "conf_delta_sum": 0.0, "conf_delta_n": 0,
                   "images_with_lost_detection": 0}
           for name, _ in pipelines}

    with ManifestWriter(manifest_path) as writer:
        n_done = n_skip = 0
        for img_path in images:
            image_id = make_image_id(args.input, img_path)
            try:
                original = load_image_rgb(img_path)
                with timer_ms(sync_cuda=args.device.startswith("cuda")) as t_before:
                    before_dets = detector.detect_persons(original)
                before_ms = t_before()
            except Exception as e:
                print(f"[{TASK}] ERROR on original {img_path}: {e!r}", file=sys.stderr)
                continue

            for pipeline_name, img_dir in pipelines:
                record_id = f"{pipeline_name}/{image_id}"
                if writer.already_done(record_id):
                    n_skip += 1
                    continue

                variant_path = img_dir / f"{image_id}.jpg"
                if not variant_path.exists():
                    writer.write(ImageRecord(
                        task=TASK, image_id=record_id, image_path=str(img_path),
                        device=args.device, error=f"missing pipeline output: {variant_path}",
                        extra={"pipeline": pipeline_name},
                    ))
                    continue

                try:
                    variant = load_image_rgb(variant_path)
                    with timer_ms(sync_cuda=args.device.startswith("cuda")) as t_after:
                        after_dets = detector.detect_persons(variant)
                    after_ms = t_after()

                    cmp = match_detections(before_dets, after_dets, iou_thresh=args.iou_thresh)

                    rec = ImageRecord(
                        task=TASK, image_id=record_id, image_path=str(img_path),
                        output_path=str(variant_path),
                        detector_name="YOLOX-s (COCO person)",
                        device=args.device,
                        extra={
                            "pipeline": pipeline_name,
                            "n_persons_before": cmp["n_before"],
                            "n_persons_after": cmp["n_after"],
                            "n_matched": cmp["n_matched"],
                            "retention_rate": cmp["retention_rate"],
                            "mean_iou_matched": cmp["mean_iou_matched"],
                            "mean_conf_before_matched": cmp["mean_conf_before_matched"],
                            "mean_conf_after_matched": cmp["mean_conf_after_matched"],
                            "mean_conf_delta": cmp["mean_conf_delta"],
                            "n_lost_detections": len(cmp["lost_detections"]),
                            "lost_detections": cmp["lost_detections"],
                        },
                    )
                    rec.timing.detect_ms = before_ms + after_ms
                    rec.timing.total_ms = before_ms + after_ms
                    writer.write(rec)
                    n_done += 1

                    a = agg[pipeline_name]
                    a["n_images"] += 1
                    a["n_before"] += cmp["n_before"]
                    a["n_matched"] += cmp["n_matched"]
                    if cmp["mean_conf_delta"] is not None:
                        a["conf_delta_sum"] += cmp["mean_conf_delta"] * cmp["n_matched"]
                        a["conf_delta_n"] += cmp["n_matched"]
                    if cmp["lost_detections"]:
                        a["images_with_lost_detection"] += 1
                except Exception as e:
                    writer.write(ImageRecord(task=TASK, image_id=record_id, image_path=str(img_path),
                                              device=args.device, error=repr(e),
                                              extra={"pipeline": pipeline_name}))
                    print(f"[{TASK}] ERROR on {variant_path}: {e!r}", file=sys.stderr)

    summary = {}
    for name, a in agg.items():
        summary[name] = {
            "n_images_compared": a["n_images"],
            "total_persons_in_originals": a["n_before"],
            "total_persons_retained": a["n_matched"],
            "overall_retention_rate": (a["n_matched"] / a["n_before"]) if a["n_before"] else None,
            "mean_confidence_delta_matched": (a["conf_delta_sum"] / a["conf_delta_n"]) if a["conf_delta_n"] else None,
            "n_images_with_at_least_one_lost_person": a["images_with_lost_detection"],
        }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[{TASK}] done. comparisons={n_done} skipped(resumed)={n_skip}", file=sys.stderr)
    print(f"[{TASK}] manifest={manifest_path}", file=sys.stderr)
    print(f"[{TASK}] summary={summary_path}", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
