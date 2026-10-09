#!/usr/bin/env bash
# Runs all 5 tasks in sequence against a shared dataset directory.
#
#   ./run_all.sh /path/to/dataset [--limit N] [--device cuda:0]
#
# Tasks 1-4 each activate their own venv, process the dataset, write
# results/<task>/{images,manifest.jsonl}. Task 5 then runs LAST (it
# depends on tasks 1-4's output existing) and writes
# results/task5_yolox_eval/{manifest.jsonl,summary.json}.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?usage: ./run_all.sh /path/to/dataset [extra run.py args...]}"
shift || true
EXTRA_ARGS=("$@")

# Resolve to absolute before any `cd` below, otherwise a relative path
# (the common case) silently resolves against each task's own directory
# instead of the caller's cwd, and every task reports "0 images to process".
DATASET="$(cd "$DATASET" && pwd)"

export UV_CACHE_DIR="$ROOT/.uv_cache"

run_task () {
  local task="$1"
  echo "=== [$task] starting $(date -Iseconds) ===" >&2
  ( cd "$ROOT/$task" \
    && source .venv/bin/activate \
    && python3 run.py --input "$DATASET" --output "$ROOT/results/$task" "${EXTRA_ARGS[@]}" \
    && deactivate )
  echo "=== [$task] done $(date -Iseconds) ===" >&2
}

run_task task1_retinaface_mediapipe
run_task task2_scrfd_segface
run_task task3_yolo11n_mobilesam
run_task task4_yolov5n_mediapipe
run_task task5_yolox_eval

echo "=== all tasks complete. Results under $ROOT/results/ ===" >&2
cat "$ROOT/results/task5_yolox_eval/summary.json" 2>/dev/null || true
