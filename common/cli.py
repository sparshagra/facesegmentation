"""Shared argparse contract: every task's run.py takes the same flags, and
every task's manifest is written by common.manifest, so Task 5 can consume
all four uniformly."""
from __future__ import annotations

import argparse


def build_arg_parser(task_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Run {task_name} pipeline")
    p.add_argument("--input", required=True, help="input image directory")
    p.add_argument("--output", required=True, help="output directory for blurred images + manifest")
    p.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    p.add_argument("--batch", type=int, default=1, help="batch size (detector stage)")
    p.add_argument("--limit", type=int, default=None, help="only process first N images")
    p.add_argument("--min-disk-gb", type=float, default=5.0)
    p.add_argument("--save-masks", action="store_true", help="also write mask pngs")
    return p
