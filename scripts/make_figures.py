#!/usr/bin/env python3
"""Generates static PNG comparison charts from results/eval/*.csv.
    python scripts/make_figures.py
Writes to results/figures/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "eval"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same validated categorical palette (light mode) used in the HTML dashboard.
COLORS = {
    "task1_retinaface_mediapipe": "#2a78d6",
    "task2_scrfd_segface": "#eb6834",
    "task3_yolo11n_mobilesam": "#1baf7a",
    "task4_yolov5n_mediapipe": "#eda100",
}
NAMES = {
    "task1_retinaface_mediapipe": "T1 RetinaFace+MediaPipe",
    "task2_scrfd_segface": "T2 SCRFD+SegFace",
    "task3_yolo11n_mobilesam": "T3 YOLO11n+MobileSAM",
    "task4_yolov5n_mediapipe": "T4 YOLOv5n+MediaPipe",
}
PIPELINES = list(COLORS.keys())

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.labelcolor": SECONDARY,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
})


def load_csv(name):
    with open(EVAL_DIR / name) as f:
        return list(csv.DictReader(f))


def f(x):
    return float(x) if x not in (None, "", "None") else None


# ---------- Fig 1: grouped bars — mask IoU / boundary IoU / boundary F1 ----------
def fig_quality():
    rows = {r["pipeline"]: r for r in load_csv("per_pipeline.csv")}
    metrics = [("mask_iou", "Mask IoU\n(whole mask)"), ("boundary_iou", "Boundary IoU\n(2% band)"),
               ("boundary_f1", "Boundary F1\n(3px tolerance)")]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    n_metrics, n_pipe = len(metrics), len(PIPELINES)
    group_w = 0.8
    bar_w = group_w / n_pipe
    x = np.arange(n_metrics)

    for i, p in enumerate(PIPELINES):
        vals = [f(rows[p][m]) for m, _ in metrics]
        offs = (i - (n_pipe - 1) / 2) * bar_w
        bars = ax.bar(x + offs, vals, width=bar_w * 0.9, color=COLORS[p], label=NAMES[p],
                       edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=8, color=SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Mask agreement with ground truth\nWhole-mask IoU barely separates the pipelines; boundary metrics do",
                  fontsize=12.5, fontweight="bold", loc="left", color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_mask_boundary_quality.png", bbox_inches="tight")
    plt.close(fig)


# ---------- Fig 2: YOLOX retention rate ----------
def fig_retention():
    with open(ROOT / "results" / "task5_yolox_eval" / "summary.json") as fh:
        summary = json.load(fh)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    vals = [summary[p]["overall_retention_rate"] for p in PIPELINES]
    lost = [summary[p]["n_images_with_at_least_one_lost_person"] for p in PIPELINES]
    colors = [COLORS[p] for p in PIPELINES]
    bars = ax.bar(range(len(PIPELINES)), vals, color=colors, width=0.6, edgecolor=SURFACE, linewidth=1.5)
    for b, v, n in zip(bars, vals, lost):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v*100:.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=INK)
        ax.text(b.get_x() + b.get_width() / 2, 0.865, f"{n} imgs w/\nlost person", ha="center", va="bottom",
                 fontsize=8, color=MUTED)

    ax.set_xticks(range(len(PIPELINES)))
    ax.set_xticklabels([NAMES[p].split(" ", 1)[0] for p in PIPELINES])
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Person retention rate")
    ax.set_title("Downstream detector retention — YOLOX-s (COCO person)\n"
                  "412 people found in originals; bar = fraction still found after anonymization",
                  fontsize=12, fontweight="bold", loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_yolox_retention.png", bbox_inches="tight")
    plt.close(fig)


# ---------- Fig 3: latency stacked bars ----------
def fig_latency():
    rows = {r["pipeline"]: r for r in load_csv("per_pipeline.csv")}
    stages = [("detect_ms", "Detect"), ("mask_ms", "Mask"), ("anonymize_ms", "Anonymize")]
    alphas = [1.0, 0.68, 0.42]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    x = np.arange(len(PIPELINES))
    bottoms = np.zeros(len(PIPELINES))
    for (key, label), alpha in zip(stages, alphas):
        vals = np.array([f(rows[p][key]) for p in PIPELINES])
        colors = [COLORS[p] for p in PIPELINES]
        bars = ax.bar(x, vals, bottom=bottoms, width=0.55, color=colors, alpha=alpha,
                       edgecolor=SURFACE, linewidth=1.2, label=label)
        bottoms += vals

    for i, p in enumerate(PIPELINES):
        total = bottoms[i]
        ax.text(i, total + 8, f"{total:.0f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[p].split(" ", 1)[0] for p in PIPELINES])
    ax.set_ylabel("Mean time per image (ms)")
    ax.set_title("Per-image latency by stage\nStage opacity: Detect 100% / Mask 68% / Anonymize 42% (color = pipeline)",
                  fontsize=11.5, fontweight="bold", loc="left", color=INK)
    # custom legend: stage opacity swatches in neutral gray
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=MUTED, alpha=a, label=lab) for (_, lab), a in zip(stages, alphas)]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_latency_breakdown.png", bbox_inches="tight")
    plt.close(fig)


# ---------- Fig 4: heatmap — boundary F1 by category ----------
def fig_heatmap():
    rows = load_csv("per_category.csv")
    categories = sorted(set(r["category"] for r in rows))
    cat_labels = {
        "01_general": "General", "02_tiny_face": "Tiny face", "03_huge_close_face": "Huge close-up",
        "04_pose_variation": "Pose variation", "05_occlusion": "Occlusion",
        "06_multiface_mixed_scale": "Multi-face mixed scale", "07_challenging_lighting": "Challenging lighting",
    }
    grid = np.zeros((len(categories), len(PIPELINES)))
    lookup = {(r["pipeline"], r["category"]): f(r["boundary_f1"]) for r in rows}
    for ci, c in enumerate(categories):
        for pi, p in enumerate(PIPELINES):
            grid[ci, pi] = lookup[(p, c)]

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    im = ax.imshow(grid, cmap="Blues", vmin=grid.min(), vmax=grid.max(), aspect="auto")
    ax.set_xticks(range(len(PIPELINES)))
    ax.set_xticklabels([NAMES[p].split(" ", 1)[0] for p in PIPELINES], rotation=20, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([cat_labels[c] for c in categories])
    for ci in range(len(categories)):
        for pi in range(len(PIPELINES)):
            v = grid[ci, pi]
            txt_color = "white" if v > (grid.min() + grid.max()) / 2 else INK
            ax.text(pi, ci, f"{v:.2f}", ha="center", va="center", fontsize=9.5, color=txt_color, fontweight="bold")
    ax.set_title("Boundary F1 by test category\nDarker = better boundary agreement with ground truth",
                  fontsize=12, fontweight="bold", loc="left", color=INK)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Boundary F1", color=SECONDARY)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_boundary_f1_by_category.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_quality()
    fig_retention()
    fig_latency()
    fig_heatmap()
    print("wrote figures to", OUT_DIR)
    for p in sorted(OUT_DIR.glob("*.png")):
        print(" ", p.name)
