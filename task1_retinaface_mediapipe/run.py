#!/usr/bin/env python3
"""Task 1: RetinaFace-R50 (detector) + MediaPipe Face Mesh (mask) + Gaussian blur.

    python run.py --input data/ --output results/task1_retinaface_mediapipe/ \
        --device cuda:0 --limit 120
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guards
from common.anonymize import gaussian_blur_region
from common.cli import build_arg_parser
from common.io_utils import list_images, load_image_rgb, make_image_id, save_image_rgb
from common.manifest import Detection, ImageRecord, ManifestWriter, run_metadata
from common.timing import timer_ms

TASK = "task1_retinaface_mediapipe"
HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE.parent / "weights"

# Measured on this box's RTX A6000: RetinaFace-R50 (fp32, single image) peaks
# well under 800MB resident; MediaPipe runs on CPU. Generous margin kept
# since VRAM here is shared with other users' processes.
REQUIRED_VRAM_MB = 900.0


def main():
    args = build_arg_parser(TASK).parse_args()
    guards.print_preflight(TASK, REQUIRED_VRAM_MB if args.device.startswith("cuda") else None,
                            args.min_disk_gb, args.output, args.device)

    from src.detector import RetinaFaceDetector
    from src.mediapipe_mask import MediaPipeFaceMesher

    detector = RetinaFaceDetector(str(WEIGHTS_DIR / "RetinaFace-R50.pth"), device=args.device)
    mesher = MediaPipeFaceMesher(str(WEIGHTS_DIR / "face_landmarker.task"))

    images = list_images(args.input, limit=args.limit)
    print(f"[{TASK}] {len(images)} images to process", file=sys.stderr)

    out_dir = Path(args.output)
    img_out_dir = out_dir / "images"
    mask_out_dir = out_dir / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    with ManifestWriter(manifest_path) as writer:
        if not manifest_path.exists() or manifest_path.stat().st_size == 0:
            pass  # header optional; per-record task/device fields suffice
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
                with timer_ms() as t_mask:
                    mask, n_meshed = mesher.masks_for_boxes(image, boxes)
                mask_ms = t_mask()

                with timer_ms() as t_anon:
                    out_img = gaussian_blur_region(image, mask) if boxes else image.copy()
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
                    detector_name="RetinaFace-R50", mask_method="mediapipe_facemesh_hull",
                    anonymize_method="gaussian_blur", n_faces=len(boxes),
                    detections=[Detection(box_xyxy=d["box"], score=d["score"],
                                           landmarks=d["landmarks5"]) for d in dets],
                    device=args.device,
                    extra={"n_faces_meshed": n_meshed},
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

    mesher.close()
    print(f"[{TASK}] done. processed={n_done} skipped(resumed)={n_skip} "
          f"manifest={manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
