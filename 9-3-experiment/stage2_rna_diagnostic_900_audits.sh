#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
cd "$repo"

wait_and_audit() {
  local gpu=$1 name=$2
  local checkpoint="$repo/data/runs/nanodesign-v1/stage2-rna-diagnostics/$name/milestones/samples-00000900.pt"
  until [[ -s "$checkpoint" ]]; do
    sleep 30
  done
  srun --overlap --jobid=7694689 --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 \
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo/src:$repo" PYTHONNOUSERSITE=1 \
    data/envs/rfd3na312/bin/python scripts/audit_stage2_checkpoint.py \
      --checkpoint "$checkpoint" \
      --task rna \
      --weight-source ema \
      --generation-examples 0 \
      --output "9-3-experiment/${name}_900_v2_context_geometry_audit.json"
}

wait_and_audit 2 rna-sequence-only-6000 & p0=$!
wait_and_audit 3 rna-seq-weight1-6000 & p1=$!
wait "$p0"
wait "$p1"
