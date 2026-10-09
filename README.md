# faceblur — face anonymization pipeline comparison

Four independently-implemented face detection → mask → anonymization
pipelines, built to the same architecture spec, benchmarked on the same
98-image dataset, plus a fifth harness that checks whether anonymizing
faces costs a generic downstream person detector any of the people it
used to find.

The core question this repo answers: **does blurring/pixelating faces
degrade a downstream model's ability to detect people in the image?**
(Task 5, below.) Everything else — mask-quality metrics, timing,
per-category breakdowns — exists to explain *why* the pipelines differ,
not just that they do.

For the exact environment, parameter values and evaluation protocol (so
a new pipeline can be benchmarked on equal footing), see
[`TESTING_CONDITIONS.md`](TESTING_CONDITIONS.md).

## The four pipelines

| | Detector | Face mask | Anonymization |
|---|---|---|---|
| **Task 1** | RetinaFace-R50 (anchor-based ResNet-50) | MediaPipe Face Mesh (468 landmarks → convex hull) | Gaussian blur, feathered edge |
| **Task 2** | SCRFD-2.5G (anchor-based, ONNX) | SegFace-Mobile (MobileNetV3 + transformer decoder, 19-class CelebAMask-HQ parsing) | Gaussian blur, feathered edge |
| **Task 3** | YOLO11n-Face (anchor-free) | MobileSAM (Tiny-ViT encoder, box-prompted decoder) | Radial gradient pixelation (block size grows from mask center to edge) |
| **Task 4** | YOLOv5n-Face (anchor-based, + 5-point landmarks) | MediaPipe Face Mesh (468 landmarks → convex hull) | Gaussian blur, feathered edge |

**Task 5** runs YOLOX-s (COCO-pretrained, "person" class only — there is
no face-trained YOLOX) on the original images and on each pipeline's
anonymized output, then matches before/after person boxes to measure
how many detections survive anonymization.

## Repo layout

```
faceblur/
├── common/                  # Shared contract used by all 5 tasks
│   ├── manifest.py          #   ImageRecord schema written by every run.py
│   ├── guards.py            #   Disk/VRAM preflight checks (fail fast, before model load)
│   ├── anonymize.py         #   Blur / radial-pixelate / mask-union ops
│   ├── weights.py           #   Checksummed weight loader
│   ├── io_utils.py          #   make_image_id() — collision-safe manifest keys
│   ├── timing.py            #   CUDA-synchronized per-stage timers
│   ├── coco_gt.py           #   Loads ground-truth masks from the dataset's COCO annotations
│   └── seg_metrics.py       #   mask_iou, mask_dice, boundary_iou, boundary_f1
│
├── task1_retinaface_mediapipe/
├── task2_scrfd_segface/
├── task3_yolo11n_mobilesam/
├── task4_yolov5n_mediapipe/    # each task follows the same internal layout:
│   ├── run.py                  #   CLI entry point (see "Running a pipeline" below)
│   ├── src/                    #   detector + mask wrappers specific to this task
│   ├── vendor/                 #   vendored upstream reference implementation (see Credits)
│   └── requirements.txt        #   this task's pinned dependencies
│
├── task5_yolox_eval/           # downstream-detector retention check (depends on tasks 1-4's output)
│
├── weights/                 # pretrained checkpoints, see "Model weights" below
├── data/combined_final/     # the evaluation dataset, see "Dataset" below
├── results/                 # pipeline outputs + evaluation results, see "Results" below
├── scripts/
│   ├── evaluate.py           # whole-image mask comparison against ground truth
│   ├── evaluate_faces.py     # per-face TP-matched comparison (box IoU≥0.5, conf≥0.4)
│   ├── make_figures.py       # matplotlib figures, round 1 (quality/retention/latency/heatmap)
│   └── make_figures2.py      # matplotlib figures, round 2 (TP-matched quality, speed-vs-accuracy bubbles, efficiency, per-category)
│
├── results_dashboard.html   # standalone interactive HTML dashboard (open directly in a browser)
├── run_all.sh                # runs tasks 1→5 against a dataset directory end to end
├── requirements.txt           # top-level dependency overview (see per-task files for the real pins)
└── TESTING_CONDITIONS.md      # exact hardware/software/eval conditions, for reproducing or extending this benchmark
```

## Dataset

`data/combined_final/combined/` — **98 images across 7 category
folders**, each folder a different test scenario (not a training
class):

| Category | Images | Scenario |
|---|---|---|
| `01_general` | 8 | baseline, unremarkable faces |
| `02_tiny_face` | 16 | small/distant faces |
| `03_huge_close_face` | 11 | large close-up faces |
| `04_pose_variation` | 22 | profile/tilted/rotated faces |
| `05_occlusion` | 11 | partially occluded faces (hands, objects, overlap) |
| `06_multiface_mixed_scale` | 14 | multiple faces per image at different scales |
| `07_challenging_lighting` | 16 | low light / backlight / harsh shadow |

Each category folder has its own `annotations_coco.json` — standard
COCO instance-segmentation schema, single category `"face"`, one
polygon `segmentation` + `bbox` per face. **157 ground-truth face
annotations total.** `data/combined_final/combined/all_labels.csv` is
an auxiliary flat label index.

Image IDs used everywhere downstream (manifests, result filenames,
ground-truth lookups) are derived from the path **relative to the
dataset root**, not the bare filename — several images share a
filename across category folders, so a bare-stem key would silently
collide. See `common/io_utils.py:make_image_id`.

`data/smoke_test/` — two throwaway images used only to sanity-check a
pipeline end-to-end before running the real dataset; not part of the
benchmark.

## Model weights

`weights/` — 8 checkpoints, ~330 MB total, stored via **Git LFS**
(`git lfs pull` after cloning if your client doesn't do this
automatically — see "Getting the code" below).

| File | Used by | Size |
|---|---|---|
| `RetinaFace-R50.pth` | Task 1 detector | 104 MB |
| `face_landmarker.task` | Tasks 1 & 4 (MediaPipe) | 3.6 MB |
| `scrfd_2.5g_bnkps.onnx` | Task 2 detector | 3.1 MB |
| `mobilenet_celeba_512__model_299.pt` | Task 2 (SegFace-Mobile parser) | 81 MB |
| `yolov11n-face.pt` | Task 3 detector | 5.2 MB |
| `mobile_sam.pt` | Task 3 (SAM decoder) | 39 MB |
| `yolov5n-face.pt` | Task 4 detector | 14 MB |
| `yolox_s.pth` | Task 5 (downstream retention check) | 69 MB |

Full SHA-256 checksums and provenance notes (including the one weight
— `scrfd_2.5g_bnkps.onnx` — sourced from a community mirror rather than
an official release, since the official link is a scripted-download-blocking
OneDrive share) are in [`TESTING_CONDITIONS.md`](TESTING_CONDITIONS.md#weight-provenance-for-reference-not-required-to-match).

## Environments

Each task runs in its **own isolated Python 3.10 virtual environment**
(built with [`uv`](https://docs.astral.sh/uv/), sharing one package
cache so the pinned `torch==2.5.1+cu121` build is stored once on disk
rather than once per env). This is deliberate: the 4 pipelines use
different, sometimes conflicting dependency trees (`insightface`,
`ultralytics`, `mediapipe`, `yolox`), so isolation avoids version
clashes rather than fighting a single shared environment.

To set up a task's environment:
```bash
cd task1_retinaface_mediapipe
uv venv
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```
(substitute any `pip install -r requirements.txt --extra-index-url ...`
if not using `uv`). Task 3 additionally needs its vendored `mobile_sam`
package installed editable: `uv pip install -e vendor/`.

See the top-level [`requirements.txt`](requirements.txt) for a quick
overview of what each pipeline needs; the per-task `requirements.txt`
files are the ones that actually pin a reproducible environment.

## Running a pipeline

Every `run.py` follows the same CLI contract:
```bash
python run.py --input <dataset_dir> --output <results_dir> --device cuda:0 [--limit N] [--save-masks]
```
It's resumable — re-running with the same `--output` skips image IDs
already present in `manifest.jsonl` rather than erroring or
reprocessing.

To run everything end to end (tasks 1-4, then task 5, which depends on
their output):
```bash
./run_all.sh data/combined_final/combined
```

## Evaluation

Two scripts score pipeline output against the dataset's ground truth,
using two different, **non-comparable** methodologies — don't mix
numbers between them:

- **`scripts/evaluate.py`** — whole-image: compares each pipeline's
  single saved union-mask against the ground-truth union-mask for that
  image. Output: `results/eval/{per_image,per_category,per_pipeline}.csv` + `summary.json`.
- **`scripts/evaluate_faces.py`** — per-face: greedily matches each
  predicted detection to a ground-truth face (box IoU≥0.50, confidence≥0.40),
  then scores only the matched (TP) pairs on a shared crop around that
  face. Unmatched ground-truth faces count against recall, not against
  quality. Output: `results/eval_faces/{per_face,per_category,per_pipeline}.csv` + `summary.json`.

Metrics computed (`common/seg_metrics.py`):
- `mask_iou`, `mask_dice` — standard pixel overlap
- `boundary_iou` — band-restricted IoU near the mask contour only
  (Cheng et al., CVPR 2021) — tolerant of small boundary misalignment,
  band width scales with the comparison region
- `boundary_f1` — precision/recall/F1 of boundary pixels within a fixed
  3px tolerance (Csurka et al., BMVC 2013)

Exact parameter values (matching thresholds, band widths, crop margins,
YOLOX retention-check config) are in
[`TESTING_CONDITIONS.md`](TESTING_CONDITIONS.md#evaluation-methodology-this-is-the-part-that-must-match-exactly) —
read that section before comparing a new pipeline's numbers against
these.

## Results

### Whole-image mask quality vs. ground truth (`results/eval/summary.json`)

| Pipeline | Mask IoU | Mask Dice | Boundary IoU | Boundary F1 | Avg. total ms/image |
|---|---|---|---|---|---|
| Task 1 — RetinaFace + MediaPipe | 0.645 | 0.765 | 0.575 | 0.427 | 466 |
| Task 2 — SCRFD + SegFace-Mobile | 0.649 | 0.736 | **0.612** | **0.570** | **203** |
| Task 3 — YOLO11n + MobileSAM | 0.608 | 0.681 | 0.576 | 0.578 | 313 |
| Task 4 — YOLOv5n + MediaPipe | 0.608 | 0.720 | 0.535 | 0.391 | 240 |

### Per-face TP-matched quality (`results/eval_faces/summary.json`, box IoU≥0.50 & conf≥0.40, 157 GT faces)

| Pipeline | Recall | Matched faces | Mask IoU (TP) | Boundary IoU (TP)¹ | Boundary F1 (TP) |
|---|---|---|---|---|---|
| Task 1 — RetinaFace + MediaPipe | **0.898** | 141/157 | 0.689 | 0.122 | 0.469 |
| Task 2 — SCRFD + SegFace-Mobile | 0.828 | 130/157 | 0.769 | 0.240 | 0.695 |
| Task 3 — YOLO11n + MobileSAM | 0.739 | 116/157 | **0.796** | **0.328** | **0.755** |
| Task 4 — YOLOv5n + MediaPipe | 0.854 | 134/157 | 0.679 | 0.121 | 0.438 |

¹ Boundary IoU here is scaled to each face's local crop diagonal
(per-face-fair across tiny vs. huge faces), so these numbers are **not
comparable** to the whole-image Boundary IoU column above — see the
methodology note linked above.

### Downstream person-detector retention (`results/task5_yolox_eval/summary.json`)

YOLOX-s, COCO "person" class, run on the 98 original images (412 people
found) vs. each pipeline's anonymized output, matched by box IoU≥0.50:

| Pipeline | Persons retained | Retention rate | Mean confidence Δ (matched) | Images losing ≥1 person |
|---|---|---|---|---|
| Task 1 — RetinaFace + MediaPipe | 382/412 | 92.7% | −0.0040 | 20/98 |
| Task 2 — SCRFD + SegFace-Mobile | 387/412 | **93.9%** | −0.0037 | 17/98 |
| Task 3 — YOLO11n + MobileSAM | 378/412 | 91.7% | −0.0098 | 22/98 |
| Task 4 — YOLOv5n + MediaPipe | 382/412 | 92.7% | −0.0046 | 19/98 |

**Takeaway: anonymizing faces costs the downstream person detector
roughly 6-8% of the people it used to find, fairly consistently across
all 4 pipelines** — none of them stands out as substantially safer or
more damaging on this check. Task 2 is marginally best on both this
check and mask-boundary quality, while being the fastest by a wide
margin (SCRFD + ONNX runtime vs. the other three's PyTorch detectors).

### Figures (`results/figures/*.png`, generated by `scripts/make_figures.py` / `make_figures2.py`)

| File | Shows |
|---|---|
| `01_mask_boundary_quality.png` | Mask IoU/Dice/Boundary IoU/Boundary F1, all 4 pipelines, grouped bars |
| `02_yolox_retention.png` | YOLOX person-retention rate, all 4 pipelines |
| `03_latency_breakdown.png` | Per-stage (detect/mask/anonymize) latency breakdown |
| `04_boundary_f1_by_category.png` | Boundary F1 heatmap, pipeline × dataset category |
| `05_tp_matched_quality.png` | Same quality metrics as `01`, but on the per-face TP-matched evaluation |
| `06_accuracy_vs_speed_pixel_iou.png` | Bubble chart: mask IoU vs. images/sec, bubble size = combined weight size (MB) |
| `07_accuracy_vs_speed_boundary_f1.png` | Same, with Boundary F1 on the accuracy axis |
| `08_efficiency.png` | 2-panel bars: model size (MB) and throughput (img/s), all 4 pipelines |
| `09_category_accuracy_pixel_iou.png` | Mask IoU per dataset category, all 4 pipelines |
| `10_category_accuracy_boundary_f1.png` | Boundary F1 per dataset category, all 4 pipelines |

`results_dashboard.html` is a standalone interactive version of the
first batch of these charts (hover tooltips, full data table) — open
it directly in any browser, no server needed.

### Raw output

`results/<task_name>/` per pipeline:
- `manifest.jsonl` — one record per image: detections (box + score +
  landmarks), face count, per-stage timing, error status
- `images/<image_id>.jpg` — the anonymized output image
- `masks/<image_id>.png` — the union face mask (all detected faces merged)

## Credits / vendored code

Each task's `vendor/` directory contains a trimmed copy of the
reference implementation its detector or mask model is built on
(nested `.git` history removed; only the code actually imported is
kept). Licenses, per vendored repo:

| `vendor/` in | Upstream | License |
|---|---|---|
| `task1_retinaface_mediapipe/` | RetinaFace (PyTorch) | MIT |
| `task2_scrfd_segface/` | SegFace (Kartik Narayan) | MIT |
| `task3_yolo11n_mobilesam/` | MobileSAM | Apache-2.0 |
| `task4_yolov5n_mediapipe/` | YOLOv5-face | **GPL-3.0** |
| `task5_yolox_eval/` | YOLOX (Megvii) | Apache-2.0 |

Note the YOLOv5-face vendor code (Task 4) is GPL-3.0 — anyone reusing
*that* component specifically should comply with GPL-3.0's terms; the
other three vendored components are permissively licensed (MIT /
Apache-2.0). This repo's own code (`common/`, `src/`, `scripts/`,
`run.py` files) carries no separate license grant yet — ask the repo
owner before reusing it outside this project.
