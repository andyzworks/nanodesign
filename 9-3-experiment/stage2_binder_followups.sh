#!/usr/bin/env bash
set -euo pipefail

REPO=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
PYTHON="$REPO/data/envs/rfd3na312/bin/python"
RUN_ROOT="$REPO/data/runs/nanodesign-v1/stage2-binder-followups"

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU NAME MODE" >&2
  exit 2
fi
gpu="$1"
name="$2"
mode="$3"

case "$mode" in
  joint-seq-w1)
    extra=(--sequence-loss-weight 1)
    ;;
  no-coordinate-augmentation)
    extra=(--no-coordinate-augmentation)
    ;;
  one-sample-reference)
    extra=(--overfit-samples-per-task 1)
    ;;
  joint-t05)
    extra=(--training-noise-level 0.5)
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
output="$RUN_ROOT/$name"
mkdir -p "$output"
[[ -s "$output/training_report.json" ]] && exit 0

common=(
  scripts/train_v0.py
  --tasks protein_binder
  --milestone-samples 300,900,3000
  --seed 17
  --validation-samples-per-task 32
  --diffusion-batch-size 16
  --output-dir "$output"
  --checkpoint-every 0
  --feature-cache-root data/cache/v0
  --no-feature-cache-fallback
  --data-workers 1
  --data-prefetch-factor 2
  --feature-cache-lru-size 128
  --learning-rate 5e-4
  --lr-schedule constant
  --weight-decay 1e-4
  --optimizer adamw
  --sequence-supervision design
  --coordinate-loss-weight 4
  --sequence-loss-weight 0.1
  --adam-beta2 0.95
  --gradient-clip 10
  --ema-decay 0.999
  --coordinate-augmentation
  --overfit-samples-per-task 32
  --final-generation
)

resume=$(find "$output/milestones" -type f -name 'samples-*.pt' 2>/dev/null | sort | tail -1 || true)
resume_args=()
[[ -n "$resume" ]] && resume_args=(--resume "$resume")
CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "${common[@]}" "${extra[@]}" "${resume_args[@]}" \
  > "$output/train.log" 2>&1
printf '%s\n' complete > "$output/status.txt"
