#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
cd "$repo"

run_audit() {
  local task=$1
  local run=$2
  srun --overlap --jobid=8829431 --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 \
    env CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$repo/src:$repo" PYTHONNOUSERSITE=1 \
    data/envs/rfd3na312/bin/python scripts/audit_stage2_checkpoint.py \
      --checkpoint "data/runs/nanodesign-v1/overfit32/${run}/milestones/samples-00000000.pt" \
      --task "$task" \
      --weight-source online \
      --generation-examples 0 \
      --output "9-3-experiment/${task}_reference_0k_v2_context_audit.json"
}

run_audit protein_binder binder-32
run_audit antibody_h3 h3-32
run_audit rna rna-32
