#!/usr/bin/env bash
set -euo pipefail

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
python="$repo/data/envs/rfd3na312/bin/python"
checkpoint="$repo/data/runs/nanodesign-v1/overfit32/binder-32/milestones/samples-00000000.pt"

cd "$repo"
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$repo/src:$repo"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

for task in protein_binder antibody_h3 rna; do
  "$python" scripts/audit_stage2_checkpoint.py \
    --checkpoint "$checkpoint" \
    --task "$task" \
    --weight-source online \
    --generation-examples 0 \
    --protocol configs/evaluation/overfit128_v2.json \
    --output "9-3-experiment/stage3_128_${task}_0k_v2_audit.json"
done
