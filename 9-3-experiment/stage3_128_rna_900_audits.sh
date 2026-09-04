#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
python="$repo/data/envs/rfd3na312/bin/python"
checkpoint="$repo/data/runs/nanodesign-v1/stage3-128/rna-reference-6000/milestones/samples-00000900.pt"

cd "$repo"
export PYTHONPATH="$repo/src:$repo"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RFD3_LOW_MEMORY_MODE

if [[ ! -s "$checkpoint" ]]; then
  echo "missing completed RNA 900 checkpoint: $checkpoint" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "$python" scripts/audit_stage2_checkpoint.py \
  --checkpoint "$checkpoint" \
  --task rna \
  --weight-source ema \
  --generation-examples 0 \
  --protocol configs/evaluation/overfit128_v2.json \
  --output 9-3-experiment/stage3_128_rna_900_v2_audit.json \
  >9-3-experiment/stage3_128_rna_900_v2_audit.log 2>&1 &
near_clean_pid=$!

CUDA_VISIBLE_DEVICES=1 "$python" scripts/audit_stage2_checkpoint.py \
  --checkpoint "$checkpoint" \
  --task rna \
  --weight-source ema \
  --generation-examples 0 \
  --protocol configs/evaluation/overfit128_v2.json \
  --diffusion-t 4.819107390595234 \
  --output 9-3-experiment/stage3_128_rna_900_median_noise_context_audit.json \
  >9-3-experiment/stage3_128_rna_900_median_noise_context_audit.log 2>&1 &
median_pid=$!

near_clean_status=0
median_status=0
wait "$near_clean_pid" || near_clean_status=$?
wait "$median_pid" || median_status=$?

if (( near_clean_status != 0 || median_status != 0 )); then
  echo "RNA 900 audit failed: near-clean=$near_clean_status median=$median_status" >&2
  exit 1
fi
