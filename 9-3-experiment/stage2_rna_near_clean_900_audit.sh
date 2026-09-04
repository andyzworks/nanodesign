#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
checkpoint="$repo/data/runs/nanodesign-v1/stage2-rna-diagnostics/rna-near-clean-sequence-only-3000/milestones/samples-00000900.pt"
cd "$repo"

until [[ -s "$checkpoint" ]]; do
  sleep 30
done

exec srun --overlap --jobid=8829431 --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 \
  env CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$repo/src:$repo" PYTHONNOUSERSITE=1 \
  data/envs/rfd3na312/bin/python scripts/audit_stage2_checkpoint.py \
    --checkpoint "$checkpoint" \
    --task rna \
    --weight-source ema \
    --generation-examples 0 \
    --output 9-3-experiment/rna_near_clean_sequence_only_900_v2_context_geometry_audit.json
