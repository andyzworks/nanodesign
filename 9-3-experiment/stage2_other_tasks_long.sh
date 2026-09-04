#!/usr/bin/env bash
set -euo pipefail

REPO=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
PYTHON="$REPO/data/envs/rfd3na312/bin/python"
RUN_ROOT="$REPO/data/runs/nanodesign-v1/stage2-other-tasks-long"

if [[ $# != 3 ]]; then
  echo "usage: $0 GPU TASK FINAL_SAMPLES" >&2
  exit 2
fi
gpu="$1"
task="$2"
final_samples="$3"

case "$task" in
  antibody_h3)
    short_name=h3
    resume="$REPO/data/runs/nanodesign-v1/overfit32/h3-32/milestones/samples-00003000.pt"
    ;;
  rna)
    short_name=rna
    resume="$REPO/data/runs/nanodesign-v1/overfit32/rna-32/milestones/samples-00003000.pt"
    ;;
  *)
    echo "task must be antibody_h3 or rna" >&2
    exit 2
    ;;
esac

case "$final_samples" in
  24000)
    milestones=300,900,3000,6000,12000,24000
    ;;
  48000)
    milestones=300,900,3000,6000,12000,24000,48000
    ;;
  *)
    echo "final samples must be 24000 or 48000" >&2
    exit 2
    ;;
esac

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
output="$RUN_ROOT/${short_name}-reference-${final_samples}"
mkdir -p "$output"
[[ -s "$output/training_report.json" ]] && exit 0

CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/train_v0.py \
  --tasks "$task" \
  --milestone-samples "$milestones" \
  --seed 17 \
  --validation-samples-per-task 32 \
  --diffusion-batch-size 16 \
  --output-dir "$output" \
  --checkpoint-every 3000 \
  --feature-cache-root data/cache/v0 \
  --no-feature-cache-fallback \
  --data-workers 1 \
  --data-prefetch-factor 2 \
  --feature-cache-lru-size 128 \
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
  --overfit-samples-per-task 32 \
  --final-generation \
  --resume "$resume" \
  > "$output/train.log" 2>&1
printf '%s\n' complete > "$output/status.txt"
