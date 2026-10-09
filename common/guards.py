"""Preflight resource guards.

These exist because this box runs at 99% disk usage and typically has
only a few GB of free VRAM (shared with other users' jobs). Every task's
run.py calls check_disk() and check_vram() before loading any model, so a
run fails fast with a clear message instead of OOM-ing or filling the
volume midway through.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


class ResourceError(RuntimeError):
    pass


def check_disk(min_free_gb: float = 5.0, path: str = ".") -> float:
    """Abort if free disk space on `path`'s filesystem is below min_free_gb.

    `path` (e.g. an output dir) may not exist yet on a first run, so this
    walks up to the nearest existing ancestor to statvfs.

    Returns free space in GB on success.
    """
    from pathlib import Path
    p = Path(path).resolve()
    while not p.exists():
        if p.parent == p:
            raise ResourceError(f"no existing ancestor directory found for {path!r}")
        p = p.parent
    total, used, free = shutil.disk_usage(p)
    free_gb = free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise ResourceError(
            f"Only {free_gb:.1f} GB free at {path!r}, need >= {min_free_gb} GB. "
            f"This filesystem is shared with other users' data — refusing to run."
        )
    return free_gb


@dataclass
class VramInfo:
    device: int
    free_mb: float
    total_mb: float

    @property
    def used_mb(self) -> float:
        return self.total_mb - self.free_mb


def query_vram(device: int = 0) -> VramInfo:
    """Query free/total VRAM without importing torch (works even if torch
    import itself would be the thing that OOMs / is slow to import)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        ).decode().strip()
        free_mb, total_mb = (float(x) for x in out.split(","))
        return VramInfo(device=device, free_mb=free_mb, total_mb=total_mb)
    except Exception as e:  # nvidia-smi missing, no GPU, etc.
        raise ResourceError(f"Could not query GPU {device} via nvidia-smi: {e}")


def check_vram(required_mb: float, device: int = 0, headroom_mb: float = 256.0) -> VramInfo:
    """Abort if free VRAM on `device` is below required_mb + headroom_mb.

    headroom_mb is slack for CUDA context + fragmentation, since this GPU
    is shared and often near-full from other users' resident processes.
    """
    info = query_vram(device)
    needed = required_mb + headroom_mb
    if info.free_mb < needed:
        raise ResourceError(
            f"GPU {device} has {info.free_mb:.0f} MB free "
            f"({info.used_mb:.0f}/{info.total_mb:.0f} MB used by other processes), "
            f"but this pipeline needs ~{needed:.0f} MB. "
            f"Try --device cpu, or wait for VRAM to free up (nvidia-smi)."
        )
    return info


def print_preflight(task_name: str, required_vram_mb: float | None, min_disk_gb: float,
                     output_dir: str, device: str = "cuda:0") -> None:
    free_gb = check_disk(min_disk_gb, output_dir)
    print(f"[{task_name}] disk OK: {free_gb:.1f} GB free at {output_dir!r} "
          f"(>= {min_disk_gb} GB required)", file=sys.stderr)
    if device.startswith("cuda") and required_vram_mb is not None:
        dev_id = int(device.split(":")[1]) if ":" in device else 0
        info = check_vram(required_vram_mb, dev_id)
        print(f"[{task_name}] VRAM OK on GPU {dev_id}: {info.free_mb:.0f} MB free "
              f"(>= {required_vram_mb:.0f} MB required)", file=sys.stderr)
    else:
        print(f"[{task_name}] running on {device}, skipping VRAM check", file=sys.stderr)
