#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
python="$repo/data/envs/rfd3na312/bin/python"
checkpoint="$repo/data/runs/nanodesign-v1/stage3-128/rna-reference-6000/milestones/samples-00006000.pt"
full_output="$repo/9-3-experiment/stage3_128_rna_6k_v2_audit_qgpu3021.json"
median_output="$repo/9-3-experiment/stage3_128_rna_6k_median_noise_context_audit_qgpu3021.json"

cd "$repo"
export PYTHONPATH="$repo/src:$repo"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RFD3_LOW_MEMORY_MODE

while true; do
  if [[ -s "$checkpoint" ]] && CHECKPOINT="$checkpoint" "$python" - <<'PY'
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

full_status=0
median_status=0

if [[ ! -s "$full_output" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$python" scripts/audit_stage2_checkpoint.py \
    --checkpoint "$checkpoint" \
    --task rna \
    --weight-source ema \
    --generation-examples 8 \
    --protocol configs/evaluation/overfit128_v2.json \
    --output "$full_output" \
    >9-3-experiment/stage3_128_rna_6k_v2_audit_qgpu3021.log 2>&1 &
  full_pid=$!
else
  full_pid=""
fi

if [[ ! -s "$median_output" ]]; then
  CUDA_VISIBLE_DEVICES=1 "$python" scripts/audit_stage2_checkpoint.py \
    --checkpoint "$checkpoint" \
    --task rna \
    --weight-source ema \
    --generation-examples 0 \
    --protocol configs/evaluation/overfit128_v2.json \
    --diffusion-t 4.819107390595234 \
    --output "$median_output" \
    >9-3-experiment/stage3_128_rna_6k_median_noise_context_audit_qgpu3021.log 2>&1 &
  median_pid=$!
else
  median_pid=""
fi

if [[ -n "$full_pid" ]]; then
  wait "$full_pid" || full_status=$?
fi
if [[ -n "$median_pid" ]]; then
  wait "$median_pid" || median_status=$?
fi

if (( full_status != 0 || median_status != 0 )); then
  echo "RNA 6K fallback audit failed: full=$full_status median=$median_status" >&2
  exit 1
fi
