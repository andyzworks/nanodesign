#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
output="$repo/data/runs/nanodesign-v1/stage2-rna-diagnostics/rna-sequence-only-6000"
cd "$repo"
mkdir -p "$output"

exec srun --overlap --jobid=7694689 --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 \
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$repo/src:$repo" PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  data/envs/rfd3na312/bin/python scripts/train_v0.py \
    --tasks rna \
    --milestone-samples 300,900,3000,6000 \
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
    --coordinate-loss-weight 0 \
    --sequence-loss-weight 0.1 \
    --adam-beta2 0.95 \
    --gradient-clip 10 \
    --ema-decay 0.999 \
    --coordinate-augmentation \
    --overfit-samples-per-task 32 \
    --no-final-generation
