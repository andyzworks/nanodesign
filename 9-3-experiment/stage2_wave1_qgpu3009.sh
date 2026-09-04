#!/usr/bin/env bash
set -euo pipefail

REPO=/gpfs/projects/b1222/userdata/jianshu/code/nanodesign
PYTHON="$REPO/data/envs/rfd3na312/bin/python"
RUN_ROOT="$REPO/data/runs/nanodesign-v1/stage2-wave1"

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$RUN_ROOT"

run_one() {
  local gpu="$1" name="$2" task="$3"
  shift 3
  local output="$RUN_ROOT/$name"
  mkdir -p "$output"
  [[ -s "$output/training_report.json" ]] && return 0
  local common=(
    scripts/train_v0.py
    --tasks "$task"
    --milestone-samples 300,900,3000
    --seed 17
    --validation-samples-per-task 32
    --diffusion-batch-size 16
    --output-dir "$output"
    --checkpoint-every 0
    --feature-cache-root data/cache/v0
    --no-feature-cache-fallback
    --data-workers 1
    --data-prefetch-factor 2
    --feature-cache-lru-size 128
    --learning-rate 5e-4
    --lr-schedule constant
    --weight-decay 1e-4
    --optimizer adamw
    --sequence-supervision design
    --coordinate-loss-weight 4
    --sequence-loss-weight 0.1
    --adam-beta2 0.95
    --gradient-clip 10
    --ema-decay 0.999
    --coordinate-augmentation
    --overfit-samples-per-task 32
    --final-generation
  )
  local attempt resume
  for attempt in 1 2; do
    resume=$(find "$output/milestones" -type f -name 'samples-*.pt' 2>/dev/null | sort | tail -1 || true)
    local resume_args=()
    [[ -n "$resume" ]] && resume_args=(--resume "$resume")
    if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "${common[@]}" "$@" "${resume_args[@]}" \
      > "$output/train-attempt-$attempt.log" 2>&1; then
      printf '%s\n' complete > "$output/status.txt"
      return 0
    fi
  done
  printf '%s\n' failed > "$output/status.txt"
  return 1
}

failed=0
# Each run differs from the frozen reference in exactly one training variable.
run_one 0 h3-all-valid antibody_h3 --sequence-supervision all_valid & p0=$!
run_one 1 rna-all-valid rna --sequence-supervision all_valid & p1=$!
run_one 2 binder-af3-schedule protein_binder --lr-schedule af3 & p2=$!
run_one 3 binder-adam protein_binder --optimizer adam & p3=$!
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || failed=1
done
((failed == 0)) || exit 1

