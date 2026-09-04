#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
cd "$repo"

exec srun --overlap --jobid=7694689 --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 \
  env CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$repo/src:$repo" PYTHONNOUSERSITE=1 \
  data/envs/rfd3na312/bin/python scripts/audit_stage2_checkpoint.py \
    --checkpoint data/runs/nanodesign-v1/stage2-other-tasks-long/h3-reference-48000/milestones/samples-00006000.pt \
    --task antibody_h3 \
    --weight-source ema \
    --generation-examples 8 \
    --output 9-3-experiment/h3_reference_6k_v2_audit.json
