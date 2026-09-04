#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
python="$repo/data/envs/rfd3na312/bin/python"
run="$repo/data/runs/nanodesign-v1/stage3-128/protein_binder-reference-6000"
checkpoint_6k="$run/milestones/samples-00006000.pt"
checkpoint_12k="$run/milestones/samples-00012000.pt"

cd "$repo"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$repo/src:$repo"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RFD3_LOW_MEMORY_MODE

if [[ ! -s "$checkpoint_6k" ]]; then
  echo "missing required Stage-3 Binder 6K checkpoint: $checkpoint_6k" >&2
  exit 1
fi

if [[ ! -s "$checkpoint_12k" ]]; then
  "$python" scripts/train_v0.py \
    --tasks protein_binder \
    --milestone-samples 300,900,3000,6000,12000 \
    --seed 17 \
    --validation-samples-per-task 128 \
    --diffusion-batch-size 16 \
    --output-dir "$run" \
    --resume "$checkpoint_6k" \
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
    --overfit-samples-per-task 128 \
    --no-final-generation
fi

"$python" scripts/audit_stage2_checkpoint.py \
  --checkpoint "$checkpoint_12k" \
  --task protein_binder \
  --weight-source ema \
  --generation-examples 8 \
  --protocol configs/evaluation/overfit128_v2.json \
  --output 9-3-experiment/stage3_128_protein_binder_12k_v2_audit.json
