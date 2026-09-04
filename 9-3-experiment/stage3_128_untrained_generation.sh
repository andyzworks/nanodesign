#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 GPU TASK" >&2
  exit 2
fi

gpu="$1"
task="$2"
case "$task" in
  protein_binder|antibody_h3|rna) ;;
  *)
    echo "unsupported task: $task" >&2
    exit 2
    ;;
esac

repo=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
python="$repo/data/envs/rfd3na312/bin/python"
checkpoint="$repo/data/runs/nanodesign-v1/overfit32/binder-32/milestones/samples-00000000.pt"

cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset RFD3_LOW_MEMORY_MODE

"$python" scripts/audit_stage2_checkpoint.py \
  --checkpoint "$checkpoint" \
  --task "$task" \
  --weight-source online \
  --generation-examples 8 \
  --protocol configs/evaluation/overfit128_v2.json \
  --output "9-3-experiment/stage3_128_${task}_0k_v2_generation_audit.json"
