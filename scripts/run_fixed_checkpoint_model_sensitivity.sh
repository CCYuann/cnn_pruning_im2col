#!/usr/bin/env bash
set -uo pipefail

MODEL=${MODEL:?Set MODEL to a supported model name}
GPU=${GPU:-1}
MAX_JOBS=${MAX_JOBS:-2}
ROOT=${ROOT:-runs/fixed_checkpoint_multi_model}
LOG_DIR="$ROOT/logs/sensitivity/$MODEL"
SAVE_DIR="$ROOT/sensitivity_validation"
AMP_DTYPE=float16
BATCH_SIZE=256
if [[ "$MODEL" == "efficientnet_v2_m" ]]; then
  AMP_DTYPE=bfloat16
  BATCH_SIZE=64
fi

mkdir -p "$LOG_DIR" "$SAVE_DIR"

run_job() {
  local seed=$1
  local tm=$2
  local run_name="${MODEL}_seed${seed}_tm${tm}_top2"
  local checkpoint="$ROOT/checkpoints/${MODEL}_seed${seed}_dense.pt"
  local ranked="$SAVE_DIR/$run_name/layer_sensitivity_ranked.csv"
  if [[ -s "$ranked" ]]; then
    echo "SKIP $run_name ranking exists"
    return 0
  fi
  if [[ ! -s "$checkpoint" ]]; then
    echo "MISSING $checkpoint"
    return 1
  fi

  echo "START $run_name"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/layer_sensitivity.py \
    --dataset cifar10 \
    --models "$MODEL" \
    --device cuda \
    --amp \
    --amp-dtype "$AMP_DTYPE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 2 \
    --pin-memory \
    --ratios 0.5,0.6,0.7 \
    --mode cross_filter \
    --criterion top2_mean_abs \
    --tm "$tm" \
    --dense-checkpoint-in "$checkpoint" \
    --seed "$seed" \
    --val-size 5000 \
    --split-seed 0 \
    --no-cudnn-benchmark \
    --save-dir "$SAVE_DIR" \
    --run-name "$run_name" \
    > "$LOG_DIR/${run_name}.log" 2>&1
  local status=$?
  echo "END $run_name status=$status"
  return "$status"
}

running=0
failed=0
for seed in 0 1 2; do
  for tm in 8 16; do
    run_job "$seed" "$tm" &
    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
      wait -n || failed=1
      running=$((running - 1))
    fi
  done
done

while (( running > 0 )); do
  wait -n || failed=1
  running=$((running - 1))
done

exit "$failed"
