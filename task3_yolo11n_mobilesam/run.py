#!/usr/bin/env python3
"""Task 3: YOLO11n-Face (detector) + MobileSAM box-prompted mask +
radial gradient pixelation.

    python run.py --input data/ --output results/task3_yolo11n_mobilesam/ \
        --device cuda:0 --limit 120
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guards
from common.anonymize import radial_gradient_pixelate
from common.cli import build_arg_parser
from common.io_utils import list_images, load_image_rgb, make_image_id, save_image_rgb
from common.manifest import Detection, ImageRecord, ManifestWriter
from common.timing import timer_ms

TASK = "task3_yolo11n_mobilesam"
HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE.parent / "weights"

# YOLO11n-Face (nano) + MobileSAM (Tiny-ViT-5M encoder, 10.1M params total).
# Generous margin kept since VRAM here is shared with other users' processes.
REQUIRED_VRAM_MB = 800.0


def main():
    args = build_arg_parser(TASK).parse_args()
    guards.print_preflight(TASK, REQUIRED_VRAM_MB if args.device.startswith("cuda") else None,
                            args.min_disk_gb, args.output, args.device)

    from src.detector import Yolo11FaceDetector
    from src.mobilesam_mask import MobileSamMasker

    detector = Yolo11FaceDetector(str(WEIGHTS_DIR / "yolov11n-face.pt"), device=args.device)
    masker = MobileSamMasker(str(WEIGHTS_DIR / "mobile_sam.pt"), device=args.device)

    images = list_images(args.input, limit=args.limit)
    print(f"[{TASK}] {len(images)} images to process", file=sys.stderr)

    out_dir = Path(args.output)
    img_out_dir = out_dir / "images"
    mask_out_dir = out_dir / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    with ManifestWriter(manifest_path) as writer:
        n_done = n_skip = 0
        for img_path in images:
            image_id = make_image_id(args.input, img_path)
            if writer.already_done(image_id):
                n_skip += 1
                continue
            try:
                image = load_image_rgb(img_path)
                with timer_ms(sync_cuda=args.device.startswith("cuda")) as t_detect:
                    dets = detector.detect(image)
                detect_ms = t_detect()

                boxes = [d["box"] for d in dets]
                with timer_ms(sync_cuda=args.device.startswith("cuda")) as t_mask:
                    mask, n_prompted = masker.masks_for_boxes(image, boxes)
                mask_ms = t_mask()

                with timer_ms() as t_anon:
                    out_img = radial_gradient_pixelate(image, mask) if boxes else image.copy()
                anon_ms = t_anon()

                out_path = img_out_dir / f"{image_id}.jpg"
                save_image_rgb(out_path, out_img)
                mask_path = None
                if args.save_masks and boxes:
                    mask_path = mask_out_dir / f"{image_id}.png"
                    mask_out_dir.mkdir(parents=True, exist_ok=True)
                    save_image_rgb(mask_path, mask[:, :, None].repeat(3, axis=2))

                rec = ImageRecord(
                    task=TASK, image_id=image_id, image_path=str(img_path),
                    output_path=str(out_path), mask_path=str(mask_path) if mask_path else None,
                    detector_name="YOLO11n-Face", mask_method="mobilesam_box_prompt",
                    anonymize_method="radial_gradient_pixelate", n_faces=len(boxes),
                    detections=[Detection(box_xyxy=d["box"], score=d["score"],
                                           landmarks=d["landmarks5"]) for d in dets],
                    device=args.device,
                    extra={"n_faces_sam_prompted": n_prompted},
                )
                rec.timing.detect_ms = detect_ms
                rec.timing.mask_ms = mask_ms
                rec.timing.anonymize_ms = anon_ms
                rec.timing.total_ms = detect_ms + mask_ms + anon_ms
                writer.write(rec)
                n_done += 1
            except Exception as e:
                writer.write(ImageRecord(task=TASK, image_id=image_id, image_path=str(img_path),
                                          device=args.device, error=repr(e)))
                print(f"[{TASK}] ERROR on {img_path}: {e!r}", file=sys.stderr)

    print(f"[{TASK}] done. processed={n_done} skipped(resumed)={n_skip} "
          f"manifest={manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
