#!/usr/bin/env bash
set -euo pipefail

REPO=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
PYTHON="$REPO/data/envs/rfd3na312/bin/python"
RUN_ROOT="$REPO/data/runs/nanodesign-v1/stage2-one-sample"

if [[ $# != 2 ]]; then
  echo "usage: $0 GPU TASK" >&2
  exit 2
fi
gpu="$1"
task="$2"
case "$task" in
  protein_binder) short_name=binder ;;
  antibody_h3) short_name=h3 ;;
  rna) short_name=rna ;;
  *)
    echo "unknown task: $task" >&2
    exit 2
    ;;
esac

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
output="$RUN_ROOT/$short_name"
mkdir -p "$output"
[[ -s "$output/training_report.json" ]] && exit 0

CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_v0.py \
  --tasks "$task" \
  --milestone-samples 300,900,3000 \
  --seed 17 \
  --validation-samples-per-task 1 \
  --diffusion-batch-size 16 \
  --output-dir "$output" \
  --checkpoint-every 0 \
  --feature-cache-root data/cache/v0 \
  --no-feature-cache-fallback \
  --data-workers 1 \
  --data-prefetch-factor 2 \
  --feature-cache-lru-size 16 \
  --learning-rate 5e-4 \
  --lr-schedule constant \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --sequence-supervision design \
  --coordinate-loss-weight 4 \
  --sequence-loss-weight 0.1 \
  --adam-beta2 0.95 \
  --gradient-clip 10 \
  --ema-decay 0.999 \
  --coordinate-augmentation \
  --overfit-samples-per-task 1 \
  --final-generation \
  > "$output/train.log" 2>&1
printf '%s\n' complete > "$output/status.txt"
