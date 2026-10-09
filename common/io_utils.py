"""Image IO + dataset listing helpers shared across tasks."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_dir: str | Path, limit: int | None = None) -> list[Path]:
    input_dir = Path(input_dir)
    paths = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        paths = paths[:limit]
    return paths


def make_image_id(root: str | Path, path: str | Path) -> str:
    """Unique, filesystem-safe manifest key for an image, derived from its
    path relative to `root` (not just the bare filename stem).

    Bare stems collide across category subfolders in real datasets (e.g.
    the same screenshot re-used in two different test-scenario folders),
    which would silently overwrite one category's manifest record/output
    with another's if used as the key. Using the relative path instead
    guarantees uniqueness, and non-ASCII/space/punctuation characters
    (real-world filenames are messy: commas, mojibake, WhatsApp/ChatGPT
    export names) are normalized so results/*/images/<id>.jpg is always a
    safe path on disk.
    """
    rel = Path(path).resolve().relative_to(Path(root).resolve())
    rel_no_ext = rel.with_suffix("")
    raw = str(rel_no_ext).replace("/", "__").replace("\\", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def load_image_rgb(path: str | Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def preload_images(paths: list[Path]) -> dict[str, np.ndarray]:
    """Decode all images into RAM once up front (cheap at ~120 images,
    ~1 GB), so GPU inference never stalls on disk IO mid-run."""
    return {p.stem: load_image_rgb(p) for p in paths}
