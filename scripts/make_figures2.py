#!/usr/bin/env python3
"""Extended comparison figures: TP-matched (IoU>=0.5, conf>=0.4) segmentation
quality including boundary metrics, accuracy-vs-speed bubble charts (bubble
size = combined model weight size), efficiency bars, and a per-category
("class") accuracy breakdown over the 7 test scenarios.

    python scripts/make_figures2.py
Reads results/eval_faces/*.csv + results/eval/per_pipeline.csv (for speed).
Writes to results/figures/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "results" / "eval"
FACES_DIR = ROOT / "results" / "eval_faces"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "task1_retinaface_mediapipe": "#2a78d6",
    "task2_scrfd_segface": "#eb6834",
    "task3_yolo11n_mobilesam": "#1baf7a",
    "task4_yolov5n_mediapipe": "#eda100",
}
NAMES = {
    "task1_retinaface_mediapipe": "Approach 1 — RetinaFace R50 + MediaPipe",
    "task2_scrfd_segface": "Approach 2 — SegFace-Mobile + SCRFD-2.5G",
    "task3_yolo11n_mobilesam": "Approach 3 — YOLO11n-Face + MobileSAM",
    "task4_yolov5n_mediapipe": "Approach 4 — YOLOv5n-Face + MediaPipe hull",
}
SHORT = {
    "task1_retinaface_mediapipe": "Approach 1\nRetinaFace R50\n+ MediaPipe",
    "task2_scrfd_segface": "Approach 2\nSegFace-Mobile\n+ SCRFD-2.5G",
    "task3_yolo11n_mobilesam": "Approach 3\nYOLO11n-Face\n+ MobileSAM",
    "task4_yolov5n_mediapipe": "Approach 4\nYOLOv5n-Face\n+ MediaPipe hull",
}
PIPELINES = list(COLORS.keys())

# Combined detector+mask weight file size, measured from weights/ on disk.
MODEL_MB = {
    "task1_retinaface_mediapipe": 113.256,
    "task2_scrfd_segface": 88.219,
    "task3_yolo11n_mobilesam": 46.184,
    "task4_yolov5n_mediapipe": 18.078,
}

INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": SECONDARY,
    "xtick.color": SECONDARY, "ytick.color": SECONDARY, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11,
})


def load_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def f(x):
    return float(x) if x not in (None, "", "None") else None


def get_speed_and_pipeline_rows():
    rows = {r["pipeline"]: r for r in load_csv(EVAL_DIR / "per_pipeline.csv")}
    speed = {p: 1000.0 / f(rows[p]["total_ms"]) for p in PIPELINES}
    return speed


def get_tp_rows():
    return {r["pipeline"]: r for r in load_csv(FACES_DIR / "per_pipeline.csv")}


# ---------- Fig: TP-matched segmentation quality (extends the reference bar chart) ----------
def fig_tp_quality():
    rows = get_tp_rows()
    with open(FACES_DIR / "summary.json") as fh:
        import json
        meta = json.load(fh)
    metrics = [("mask_iou", "Pixel IoU"), ("mask_dice", "Pixel F1 (Dice)"),
               ("boundary_iou", "Boundary IoU"), ("boundary_f1", "Boundary F1")]

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)
    n_metrics, n_pipe = len(metrics), len(PIPELINES)
    bar_w = 0.8 / n_pipe
    x = np.arange(n_metrics)

    for i, p in enumerate(PIPELINES):
        vals = [f(rows[p][m]) * 100 for m, _ in metrics]
        offs = (i - (n_pipe - 1) / 2) * bar_w
        bars = ax.bar(x + offs, vals, width=bar_w * 0.9, color=COLORS[p],
                       label=f"Approach {i+1}", edgecolor=SURFACE, linewidth=1.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                     fontsize=8, color=SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    fig.suptitle(f"Segmentation Quality — TP-matched faces  (IoU ≥ {meta['iou_thresh']:.2f}, "
                 f"Conf ≥ {meta['conf_thresh']:.2f})",
                 fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.02, color=INK)
    ax.set_title("Pixel metrics score the whole matched face region; Boundary IoU/F1 score only the contour —"
                 " note how much further the pipelines separate once the boundary is scored on its own.",
                 fontsize=9, color=MUTED, loc="left", pad=12, fontweight="normal")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[p]) for p in PIPELINES]
    ax.legend(handles, [NAMES[p] for p in PIPELINES], loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_tp_matched_quality.png", bbox_inches="tight")
    plt.close(fig)


# ---------- Fig: accuracy vs speed bubble charts ----------
def fig_bubble(metric_key, metric_label, out_name):
    rows = get_tp_rows()
    speed = get_speed_and_pipeline_rows()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sizes = np.array([MODEL_MB[p] for p in PIPELINES])
    size_scale = 2200 / sizes.max()

    for p in PIPELINES:
        x = speed[p]
        y = f(rows[p][metric_key]) * 100
        ax.scatter(x, y, s=MODEL_MB[p] * size_scale, color=COLORS[p], alpha=0.55,
                   edgecolor=COLORS[p], linewidth=1.8, zorder=3)
        ax.scatter(x, y, s=18, color=COLORS[p], zorder=4)
        dx = 0.10 if x < speed[PIPELINES[-1]] else -0.10
        ax.annotate(NAMES[p].split(" — ")[0], (x, y), xytext=(8, 8), textcoords="offset points",
                    fontsize=9, color=INK, fontweight="bold")
        ax.annotate(NAMES[p].split(" — ", 1)[1], (x, y), xytext=(8, -6), textcoords="offset points",
                    fontsize=8, color=SECONDARY)

    ax.set_xlabel("Speed (images/sec, end-to-end pipeline) → faster")
    ax.set_ylabel(f"{metric_label} (%) → more accurate")
    ax.set_title(f"Segmentation Accuracy vs Speed  (bubble = combined model size, MB)",
                 fontsize=12.5, fontweight="bold", loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    xpad = (max(speed.values()) - min(speed.values())) * 0.25
    ax.set_xlim(min(speed.values()) - xpad, max(speed.values()) + xpad * 1.6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


# ---------- Fig: efficiency (model size + speed) ----------
def fig_efficiency():
    speed = get_speed_and_pipeline_rows()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)

    ax = axes[0]
    vals = [MODEL_MB[p] for p in PIPELINES]
    bars = ax.bar(range(len(PIPELINES)), vals, color=[COLORS[p] for p in PIPELINES], width=0.6,
                  edgecolor=SURFACE, linewidth=1.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}", ha="center",
                 va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(range(len(PIPELINES)))
    ax.set_xticklabels([f"A{i+1}" for i in range(len(PIPELINES))])
    ax.set_ylabel("MB")
    ax.set_title("Model size (MB) — lower is better", fontsize=11.5, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    vals = [speed[p] for p in PIPELINES]
    bars = ax.bar(range(len(PIPELINES)), vals, color=[COLORS[p] for p in PIPELINES], width=0.6,
                  edgecolor=SURFACE, linewidth=1.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.2f}", ha="center",
                 va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(range(len(PIPELINES)))
    ax.set_xticklabels([f"A{i+1}" for i in range(len(PIPELINES))])
    ax.set_ylabel("images / sec")
    ax.set_title("Speed (img/s, GPU+CPU end-to-end) — higher is better", fontsize=11.5,
                 fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[p]) for p in PIPELINES]
    fig.legend(handles, [f"A{i+1}: {NAMES[p].split(' — ',1)[1]}" for i, p in enumerate(PIPELINES)],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Efficiency", fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_efficiency.png", bbox_inches="tight")
    plt.close(fig)


# ---------- Fig: per-category ("class") accuracy — the regression-test-set breakdown ----------
def fig_category_accuracy(metric_key, metric_label, out_name):
    rows = load_csv(FACES_DIR / "per_category.csv")
    categories = sorted(set(r["category"] for r in rows))
    cat_labels = {
        "01_general": "General", "02_tiny_face": "Tiny face", "03_huge_close_face": "Huge close-up",
        "04_pose_variation": "Pose variation", "05_occlusion": "Occlusion",
        "06_multiface_mixed_scale": "Multi-face\nmixed scale", "07_challenging_lighting": "Challenging\nlighting",
    }
    lookup = {(r["pipeline"], r["category"]): (f(r[metric_key]), int(r["n_gt_faces"]), int(r["n_matched_tp"]))
              for r in rows}

    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=150)
    n_pipe = len(PIPELINES)
    bar_w = 0.8 / n_pipe
    x = np.arange(len(categories))

    for i, p in enumerate(PIPELINES):
        vals = []
        for c in categories:
            v, n_gt, n_tp = lookup.get((p, c), (None, 0, 0))
            vals.append((v or 0) * 100)
        offs = (i - (n_pipe - 1) / 2) * bar_w
        bars = ax.bar(x + offs, vals, width=bar_w * 0.9, color=COLORS[p], label=f"A{i+1}",
                       edgecolor=SURFACE, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([cat_labels[c] for c in categories], fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"{metric_label} (%)")
    n_faces_line = "  ·  ".join(f"{cat_labels[c].replace(chr(10),' ')}: {sum(1 for r in rows if r['category']==c and r['pipeline']==PIPELINES[0])and lookup[(PIPELINES[0],c)][1]} faces" for c in categories)
    fig.suptitle(f"Accuracy by test category (class) — {metric_label}, TP-matched faces",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.02, color=INK)
    ax.set_title(n_faces_line, fontsize=8.5, color=MUTED, loc="left", pad=12, fontweight="normal")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[p]) for p in PIPELINES]
    ax.legend(handles, [f"A{i+1}: {NAMES[p].split(' — ',1)[1]}" for i, p in enumerate(PIPELINES)],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_tp_quality()
    fig_bubble("mask_iou", "Pixel IoU", "06_accuracy_vs_speed_pixel_iou.png")
    fig_bubble("boundary_f1", "Boundary F1", "07_accuracy_vs_speed_boundary_f1.png")
    fig_efficiency()
    fig_category_accuracy("mask_iou", "Pixel IoU", "09_category_accuracy_pixel_iou.png")
    fig_category_accuracy("boundary_f1", "Boundary F1", "10_category_accuracy_boundary_f1.png")
    print("done:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(" ", p.name)
