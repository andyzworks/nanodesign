#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
checkpoint="$repo/data/runs/nanodesign-v1/stage2-other-tasks-long/rna-reference-24000/milestones/samples-00024000.pt"

while [[ ! -s "$checkpoint" ]]; do
  sleep 30
done

cd "$repo"
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=src:. \
PYTHONNOUSERSITE=1 \
data/envs/rfd3na312/bin/python scripts/audit_stage2_checkpoint.py \
  --checkpoint "$checkpoint" \
  --task rna \
  --weight-source ema \
  --generation-examples 8 \
  --output 9-3-experiment/rna_reference_24k_v2_audit.json
