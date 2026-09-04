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
run="$repo/data/runs/nanodesign-v1/stage3-128/$task-reference-6000"
checkpoint="$run/milestones/samples-00006000.pt"
audit="$repo/9-3-experiment/stage3_128_${task}_6k_v2_audit.json"

cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/src:$repo"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Use the profiled size-aware execution policy by default.  A caller may still
# force the mathematically equivalent chunked path when a GPU is shared.
if [[ "${STAGE3_FORCE_CHUNKED:-0}" == 1 ]]; then
  export RFD3_LOW_MEMORY_MODE=1
else
  unset RFD3_LOW_MEMORY_MODE
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$run"

if [[ ! -s "$checkpoint" ]]; then
  "$python" scripts/train_v0.py \
    --tasks "$task" \
    --milestone-samples 300,900,3000,6000 \
    --seed 17 \
    --validation-samples-per-task 128 \
    --diffusion-batch-size 16 \
    --output-dir "$run" \
    --checkpoint-every 3000 \
    --feature-cache-root data/cache/v0 \
    --no-feature-cache-fallback \
    --data-workers 1 \
    --data-prefetch-factor 2 \
    --feature-cache-lru-size 128 \
    --learning-rate 5e-4 \
    --lr-schedule constant \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --sequence-supervision design \
    --coordinate-loss-weight 4 \
    --sequence-loss-weight 0.1 \
    --adam-beta2 0.95 \
    --gradient-clip 10 \
    --ema-decay 0.999 \
    --coordinate-augmentation \
    --overfit-samples-per-task 128 \
    --no-final-generation
fi

"$python" scripts/audit_stage2_checkpoint.py \
  --checkpoint "$checkpoint" \
  --task "$task" \
  --weight-source ema \
  --generation-examples 8 \
  --protocol configs/evaluation/overfit128_v2.json \
  --output "$audit"

if [[ "$task" == rna ]]; then
  "$python" scripts/audit_stage2_checkpoint.py \
    --checkpoint "$checkpoint" \
    --task rna \
    --weight-source ema \
    --generation-examples 0 \
    --protocol configs/evaluation/overfit128_v2.json \
    --diffusion-t 4.819107390595234 \
    --output 9-3-experiment/stage3_128_rna_6k_median_noise_context_audit.json
fi
