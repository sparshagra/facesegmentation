"""Shared manifest schema so all 4 pipelines (and the Task 5 evaluator)
produce directly comparable records.

One JSON-lines file per task run: results/<task_name>/manifest.jsonl
One record per input image.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Optional


@dataclasses.dataclass
class Detection:
    box_xyxy: list[float]          # [x1, y1, x2, y2] in pixel coords
    score: float
    landmarks: Optional[list[list[float]]] = None  # [[x,y], ...] if available


@dataclasses.dataclass
class StageTiming:
    detect_ms: float = 0.0
    mask_ms: float = 0.0
    anonymize_ms: float = 0.0
    total_ms: float = 0.0


@dataclasses.dataclass
class ImageRecord:
    task: str
    image_id: str
    image_path: str
    output_path: Optional[str] = None
    mask_path: Optional[str] = None
    detector_name: str = ""
    mask_method: str = ""
    anonymize_method: str = ""
    n_faces: int = 0
    detections: list[Detection] = dataclasses.field(default_factory=list)
    timing: StageTiming = dataclasses.field(default_factory=StageTiming)
    device: str = ""
    error: Optional[str] = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


class ManifestWriter:
    """Append-only JSONL writer. Safe to resume: call already_done() to
    skip images a previous (interrupted) run already processed."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done_ids: set[str] = set()
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("error") is None:
                            self._done_ids.add(rec["image_id"])
                    except json.JSONDecodeError:
                        continue
        self._fh = open(self.path, "a")

    def already_done(self, image_id: str) -> bool:
        return image_id in self._done_ids

    def write(self, record: ImageRecord) -> None:
        self._fh.write(record.to_json() + "\n")
        self._fh.flush()
        if record.error is None:
            self._done_ids.add(record.image_id)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_manifest(path: str | Path) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def run_metadata(task: str, **kwargs) -> dict:
    """One-off header record with run-level info (weights hash, device,
    timestamp). Write this as the first line of a manifest, or as a
    sibling run_info.json — writer does not enforce which."""
    return {
        "task": task,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **kwargs,
    }
