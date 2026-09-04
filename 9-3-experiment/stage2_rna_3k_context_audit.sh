#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
cd "$repo"

exec srun --overlap --jobid=8829431 --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 \
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$repo/src:$repo" PYTHONNOUSERSITE=1 \
  data/envs/rfd3na312/bin/python scripts/audit_stage2_checkpoint.py \
    --checkpoint data/runs/nanodesign-v1/overfit32/rna-32/milestones/samples-00003000.pt \
    --task rna \
    --weight-source ema \
    --generation-examples 0 \
    --output 9-3-experiment/rna_reference_3k_v2_context_audit.json
