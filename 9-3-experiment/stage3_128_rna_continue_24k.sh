#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
python="$repo/data/envs/rfd3na312/bin/python"
run="$repo/data/runs/nanodesign-v1/stage3-128/rna-reference-6000"
checkpoint_6k="$run/milestones/samples-00006000.pt"
checkpoint_24k="$run/milestones/samples-00024000.pt"

cd "$repo"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$repo/src:$repo"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RFD3_LOW_MEMORY_MODE

while true; do
  if [[ -s "$checkpoint_6k" ]] && CHECKPOINT="$checkpoint_6k" "$python" - <<'PY'
import os
import torch

state = torch.load(os.environ["CHECKPOINT"], map_location="cpu", weights_only=False)
records = state.get("milestone_records", [])
raise SystemExit(0 if any(r.get("global_samples_seen") == 6000 for r in records) else 1)
PY
  then
    break
  fi
  sleep 60
done

if [[ -s "$checkpoint_24k" ]]; then
  exit 0
fi

"$python" scripts/train_v0.py \
  --tasks rna \
  --milestone-samples 300,900,3000,6000,12000,24000 \
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
