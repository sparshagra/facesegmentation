"""Weight downloader with checksum verification and resumable transfer.

Deliberately fetches single files, never whole-repo snapshots — the
SegFace HF repo is 13.4 GB across 14 checkpoints; we need exactly one
84.9 MB file from it. hf_download() below only ever pulls the named
file via hf_hub_download, which does not fetch sibling files.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import requests

WEIGHTS_DIR = Path(os.environ.get("FACEBLUR_WEIGHTS_DIR",
                                   "/home/prof-viswanath-group-2/sparsh/faceblur/weights"))


def sha256sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: str | Path, expected_sha256: str | None) -> bool:
    if expected_sha256 is None:
        return True
    actual = sha256sum(path)
    ok = actual.lower() == expected_sha256.lower()
    if not ok:
        print(f"[weights] CHECKSUM MISMATCH for {path}: "
              f"expected {expected_sha256}, got {actual}", file=sys.stderr)
    return ok


def download_url(url: str, dest: str | Path, expected_sha256: str | None = None,
                  min_size_bytes: int = 1024) -> Path:
    """Stream-download url -> dest. Skips download if dest already exists
    and passes checksum (or size sanity check when no checksum given)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= min_size_bytes:
        if expected_sha256 is None or verify(dest, expected_sha256):
            print(f"[weights] cached: {dest}", file=sys.stderr)
            return dest
        print(f"[weights] cached file failed checksum, re-downloading: {dest}", file=sys.stderr)

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[weights] downloading {url} -> {dest}", file=sys.stderr)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
        if total and written < total:
            raise IOError(f"incomplete download: {written}/{total} bytes for {url}")

    if tmp.stat().st_size < min_size_bytes:
        tmp.unlink(missing_ok=True)
        raise IOError(f"downloaded file suspiciously small ({tmp.stat().st_size} B): {url}")

    if expected_sha256 is not None and not verify(tmp, expected_sha256):
        tmp.unlink(missing_ok=True)
        raise IOError(f"checksum verification failed for {url}")

    tmp.rename(dest)
    print(f"[weights] OK: {dest} ({dest.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return dest


def hf_download(repo_id: str, filename: str, dest_dir: str | Path,
                 repo_type: str = "model", expected_sha256: str | None = None) -> Path:
    """Download exactly one file from a HuggingFace repo (not a full
    snapshot) via huggingface_hub.hf_hub_download, then copy/symlink it
    to dest_dir with a flat name so task code doesn't need to know HF's
    cache layout."""
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    flat_name = filename.replace("/", "__")
    dest = dest_dir / flat_name

    if dest.exists() and (expected_sha256 is None or verify(dest, expected_sha256)):
        print(f"[weights] cached: {dest}", file=sys.stderr)
        return dest

    cached_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    if expected_sha256 is not None and not verify(cached_path, expected_sha256):
        raise IOError(f"checksum verification failed for hf://{repo_id}/{filename}")

    import shutil
    shutil.copyfile(cached_path, dest)
    print(f"[weights] OK: {dest} ({dest.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return dest


def gdrive_download(file_id: str, dest: str | Path, expected_sha256: str | None = None) -> Path:
    """Fallback for weights only distributed via Google Drive (e.g. the
    official SCRFD-2.5G onnx from insightface's model zoo)."""
    import gdown
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_sha256 is None or verify(dest, expected_sha256)):
        print(f"[weights] cached: {dest}", file=sys.stderr)
        return dest
    gdown.download(id=file_id, output=str(dest), quiet=False)
    if expected_sha256 is not None and not verify(dest, expected_sha256):
        dest.unlink(missing_ok=True)
        raise IOError(f"checksum verification failed for gdrive://{file_id}")
    return dest
