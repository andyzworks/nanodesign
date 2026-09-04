#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
checkpoint="$repo/data/runs/nanodesign-v1/stage2-one-sample/rna/milestones/samples-00003000.pt"
cd "$repo"

until [[ -s "$checkpoint" ]]; do
  sleep 30
done

exec srun --overlap --jobid=7694689 --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 \
  env CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$repo/src:$repo" PYTHONNOUSERSITE=1 \
  data/envs/rfd3na312/bin/python scripts/audit_stage2_checkpoint.py \
    --checkpoint "$checkpoint" \
    --task rna \
    --weight-source ema \
    --protocol configs/evaluation/overfit1_v2.json \
    --generation-examples 1 \
    --output 9-3-experiment/rna_one_sample_3k_v2_audit.json
