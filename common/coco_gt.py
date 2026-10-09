"""Loads ground-truth face masks from the combined_final dataset's per-
category COCO-style annotations_coco.json files, keyed by the same
make_image_id() scheme every task uses, so GT masks line up directly with
each pipeline's predicted masks for boundary-metric scoring.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from common.io_utils import make_image_id


def _polygon_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask


def load_gt_masks(dataset_root: str | Path) -> dict[str, dict]:
    """Scans every */annotations_coco.json under dataset_root and returns
    {image_id: {"mask": HxW uint8 mask (union of all face polygons),
                "n_faces": int, "category": str, "width": int, "height": int}}.

    image_id matches common.io_utils.make_image_id(dataset_root, image_path)
    exactly, so this dict can be indexed with the same key any task uses.
    """
    import json

    dataset_root = Path(dataset_root)
    out = {}
    for ann_path in sorted(dataset_root.rglob("annotations_coco.json")):
        category_dir = ann_path.parent
        category = category_dir.name
        data = json.loads(ann_path.read_text())

        images_by_id = {im["id"]: im for im in data["images"]}
        anns_by_image = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for coco_img_id, im in images_by_id.items():
            img_path = category_dir / im["file_name"]
            if not img_path.exists():
                continue
            image_id = make_image_id(dataset_root, img_path)
            h, w = im["height"], im["width"]
            anns = anns_by_image.get(coco_img_id, [])

            mask = np.zeros((h, w), dtype=np.uint8)
            instances = []
            for ann in anns:
                seg = ann.get("segmentation")
                inst_mask = _polygon_mask(seg, h, w) if seg else np.zeros((h, w), dtype=np.uint8)
                mask = cv2.bitwise_or(mask, inst_mask)
                x, y, bw, bh = ann["bbox"]
                instances.append({"box_xyxy": [x, y, x + bw, y + bh], "mask": inst_mask})

            out[image_id] = {
                "mask": mask, "n_faces": len(anns), "category": category,
                "width": w, "height": h,
                "boxes": [ann["bbox"] for ann in anns],  # [x, y, w, h] each, COCO convention
                "instances": instances,  # per-face: {"box_xyxy": [...], "mask": HxW uint8}
            }
    return out
