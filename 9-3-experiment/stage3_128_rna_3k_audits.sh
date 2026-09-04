#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
python="$repo/data/envs/rfd3na312/bin/python"
checkpoint="$repo/data/runs/nanodesign-v1/stage3-128/rna-reference-6000/milestones/samples-00003000.pt"
near_clean_output="$repo/9-3-experiment/stage3_128_rna_3k_v2_audit.json"
median_output="$repo/9-3-experiment/stage3_128_rna_3k_median_noise_context_audit.json"

cd "$repo"
export PYTHONPATH="$repo/src:$repo"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RFD3_LOW_MEMORY_MODE

# The training loop writes a checkpoint once before milestone validation and then
# rewrites it with the completed milestone record.  Do not audit the intermediate
# state merely because the path exists.
while true; do
  if [[ -s "$checkpoint" ]] && CHECKPOINT="$checkpoint" "$python" - <<'PY'
import os
import torch

state = torch.load(os.environ["CHECKPOINT"], map_location="cpu", weights_only=False)
records = state.get("milestone_records", [])
raise SystemExit(0 if any(r.get("global_samples_seen") == 3000 for r in records) else 1)
PY
  then
    break
  fi
  sleep 60
done

near_clean_status=0
median_status=0

if [[ ! -s "$near_clean_output" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$python" scripts/audit_stage2_checkpoint.py \
    --checkpoint "$checkpoint" \
    --task rna \
    --weight-source ema \
    --generation-examples 0 \
    --protocol configs/evaluation/overfit128_v2.json \
    --output "$near_clean_output" \
    >9-3-experiment/stage3_128_rna_3k_v2_audit.log 2>&1 &
  near_clean_pid=$!
else
  near_clean_pid=""
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
    >9-3-experiment/stage3_128_rna_3k_median_noise_context_audit.log 2>&1 &
  median_pid=$!
else
  median_pid=""
fi

if [[ -n "$near_clean_pid" ]]; then
  wait "$near_clean_pid" || near_clean_status=$?
fi
if [[ -n "$median_pid" ]]; then
  wait "$median_pid" || median_status=$?
fi

if (( near_clean_status != 0 || median_status != 0 )); then
  echo "RNA 3K audit failed: near-clean=$near_clean_status median=$median_status" >&2
  exit 1
fi
