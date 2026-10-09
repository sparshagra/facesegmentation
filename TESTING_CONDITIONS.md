# Testing conditions — face anonymization pipeline benchmark

Common conditions used across all 4 existing pipelines (RetinaFace+MediaPipe,
SCRFD+SegFace, YOLO11n+MobileSAM, YOLOv5n+MediaPipe). Match these for a 5th
pipeline's numbers to be directly comparable — anything that differs should
be called out explicitly rather than silently changing the baseline.

## Hardware

- GPU: NVIDIA RTX A6000, 49,140 MiB total VRAM, driver 580.82.07, CUDA 12.1 runtime
- **The GPU is shared with other users' jobs.** Free VRAM at run time fluctuated
  between ~2 GB and ~10 GB across our runs. Absolute latency numbers carry
  noise from this — treat single-run timings as indicative, not precise;
  re-run if you need a tight number.
- CPU: 12 vCPU, 61 GB RAM, Linux 5.15.0-139-generic

## Software environment

- Python 3.10.4, one venv per pipeline (isolated deps), built with `uv`
- Shared package versions across all pipelines (pin these for comparability):
  - `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`
  - `numpy==2.2.6`, `opencv-python-headless==5.0.0.93`, `scipy==1.15.3`
- Inference precision: **fp32 throughout, no autocast/fp16 anywhere.** (Worth
  noting since it's the deviation most likely to silently creep in and skew
  a speed comparison — if your pipeline uses fp16/autocast, say so rather
  than letting the numbers imply an apples-to-apples speed comparison.)
- Device policy: `cuda:0` for the detector + mask model where GPU-capable;
  MediaPipe's FaceLandmarker runs on CPU (its native TFLite/XNNPACK path) —
  this is a deliberate choice, not a limitation, since it offloads work
  that would otherwise compete for the same scarce VRAM.
- Each model is loaded **once per run**, before the per-image loop — never
  reloaded per image. This matters a lot at small N (our dataset is 98
  images); reloading per image would dominate the timing.
- Images are loaded and processed as **RGB** (OpenCV loads BGR; every
  pipeline converts immediately). State which convention your pipeline uses
  internally — several of the vendored detector reference implementations
  (RetinaFace, YOLOv5-face) internally expect BGR, so getting this backwards
  is a common, silent bug.

## Dataset & ground truth

- `combined_final` dataset: 98 images across 7 category folders (treat each
  as a test "class"/scenario, not a training class):
  `01_general`, `02_tiny_face`, `03_huge_close_face`, `04_pose_variation`,
  `05_occlusion`, `06_multiface_mixed_scale`, `07_challenging_lighting`
- 157 total GT face annotations, one `annotations_coco.json` per category
  folder: standard COCO instance-segmentation schema, single category
  `"face"`, per-instance `segmentation` (polygon), `bbox`, `area`, `image_id`.
- **Image IDs are derived from the path relative to the dataset root, not
  the bare filename** — several images share a filename across category
  folders (e.g. the same screenshot re-used in two scenarios), so a
  bare-stem key silently collides and overwrites results. Use something
  equivalent to: relative path with `/` → `__`, non-alphanumeric chars →
  `_`. (Our implementation: `common/io_utils.py:make_image_id`.)

## Evaluation methodology (this is the part that must match exactly)

**Per-face TP matching**, not whole-image comparison:
- A predicted detection counts as a true positive for a GT face when
  **box IoU ≥ 0.50** *and* **confidence ≥ 0.40**.
- Metrics below are computed **only over TP-matched faces** — an unmatched
  GT face doesn't contribute to quality scores, it counts against recall
  instead. Report both: quality-on-TPs and recall, never quality alone.
- Matching is greedy best-IoU, one predicted detection can't match two GT
  faces.

**Metrics computed per matched face** (see `common/seg_metrics.py`):
- `mask_iou`, `mask_dice` — standard pixel IoU/Dice over the mask region
- `boundary_iou` — band-restricted IoU, band width = 2% of the **local
  crop's diagonal** (not the whole image) — this is deliberately per-face
  scaled so it's fair across `02_tiny_face` and `03_huge_close_face` alike.
  If you compute this over the whole image instead, your numbers are not
  comparable to ours — say which one you did.
- `boundary_f1` (precision/recall/F1 of boundary pixels) — tolerance
  **3px**, fixed absolute pixel distance (not scaled), so this one *is*
  directly comparable regardless of crop-vs-whole-image choice.
- Mask comparison is done on a **shared crop** = (GT box ∪ matched
  predicted box), expanded by 15% margin — because each pipeline saves one
  union mask per image rather than one mask per detected face. This is
  exact for single-face images, and a reasonable approximation for
  well-separated multi-face images; on genuinely overlapping faces
  (`05_occlusion`, `06_multiface_mixed_scale`) a neighboring face's mask
  can leak into the crop. If your pipeline saves true per-instance masks,
  prefer that — just disclose the difference.

**Downstream-detector retention check** (YOLOX-s, official COCO weights,
`yolox_s.pth`, `test_conf=0.01`, `nmsthre=0.65`, `test_size=640×640`):
- Filters to the COCO **"person"** class only — stock YOLOX has no "face"
  class, so this checks whether anonymizing faces costs a generic
  downstream detector any of the people it used to find, not face
  detection itself.
- Before/after person boxes matched by box IoU ≥ 0.50, greedy.
- Retention rate reported **micro-averaged**: (total persons retained) /
  (total persons found in the originals) — summed across the whole
  dataset, not averaged per-image (per-image averaging skews toward
  images with only 1–2 people).

**Timing**: wall-clock per image, `torch.cuda.synchronize()` called
immediately before and after each GPU-bound stage so the number reflects
actual compute, not just kernel-launch latency. Reported per-stage
(detect / mask / anonymize) and total.

## Detector/mask-stage parameters actually used (defaults, unmodified)

| Pipeline | Detector params | Mask-stage params |
|---|---|---|
| RetinaFace-R50 + MediaPipe | conf≥0.5, NMS 0.4, top_k 5000, keep_top_k 750 | MediaPipe min_face_detection_confidence 0.3, crop margin 35% |
| SCRFD-2.5G + SegFace-Mobile | det_thresh 0.5, nms_thresh 0.4, input 640×640 | SegFace 512×512, ImageNet norm, crop margin 25% |
| YOLO11n-Face + MobileSAM | conf 0.5, IoU 0.5 (ultralytics) | MobileSAM box-prompt, multimask_output=False |
| YOLOv5n-Face + MediaPipe | conf 0.6, IoU 0.5, img_size 640 | MediaPipe min_face_detection_confidence 0.3, crop margin 35% |

## Output contract (so a 5th pipeline plugs into the same eval scripts unmodified)

- `results/<pipeline_name>/manifest.jsonl` — one JSON record per image
  (schema: `common/manifest.py:ImageRecord`), including `image_id`,
  `detections: [{box_xyxy, score, landmarks}]`, `n_faces`, per-stage
  timing, `error` (null on success).
- `results/<pipeline_name>/images/<image_id>.jpg` — anonymized output.
- `results/<pipeline_name>/masks/<image_id>.png` — one union mask per
  image (all detected faces merged), uint8 `{0,255}`, same resolution as
  the input image.
- Run `scripts/evaluate_faces.py` against this output directory to get
  directly comparable TP-matched metrics against the same 98-image GT set.

## Weight provenance (for reference, not required to match)

| File | SHA-256 |
|---|---|
| `RetinaFace-R50.pth` | `6d1de9c2...0a5a16d` |
| `scrfd_2.5g_bnkps.onnx` | `bc24bb34...bf7ed2` |
| `mobilenet_celeba_512__model_299.pt` | `4d67ea7d...4534993e8` |
| `yolov11n-face.pt` | `9420a9d4...985caae3bbbf7ed2` |
| `mobile_sam.pt` | `6dbb9052...37d6c2f` |
| `yolov5n-face.pt` | `794c94da...4064934e50` |
| `face_landmarker.task` | `64184e22...44e0bc9ff` |

One weight (`scrfd_2.5g_bnkps.onnx`) came from a community HuggingFace
mirror rather than an official release — the official SCRFD-2.5G checkpoint
is only distributed via a OneDrive link that blocks scripted downloads. We
verified the mirror structurally (correct 9-head/3-stride anchor grid,
correct param count vs. the published paper) rather than against an
official hash, since none exists to check against.
