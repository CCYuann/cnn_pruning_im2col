#!/usr/bin/env bash
set -uo pipefail

MODEL=${MODEL:?Set MODEL to a supported model name}
GPU=${GPU:-1}
MAX_JOBS=${MAX_JOBS:-2}
ROOT=${ROOT:-runs/fixed_checkpoint_multi_model}
SAVE_DIR="$ROOT/pruning_r0p7/$MODEL"
LOG_DIR="$ROOT/logs/pruning_r0p7/$MODEL"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

run_job() {
  local seed=$1
  local tm=$2
  local batch_size=128
  local amp_dtype=float16
  if [[ "$MODEL" == "efficientnet_v2_m" ]]; then
    batch_size=64
    amp_dtype=bfloat16
  fi
  local run_name="${MODEL}_seed${seed}_tm${tm}_r0p7"
  local checkpoint="$ROOT/checkpoints/${MODEL}_seed${seed}_dense.pt"
  local sensitivity_csv="$ROOT/sensitivity_validation/${MODEL}_seed${seed}_tm${tm}_top2/layer_sensitivity_ranked.csv"
  local result="$SAVE_DIR/${run_name}_results.csv"
  if [[ -s "$result" ]]; then
    echo "SKIP $run_name result exists"
    return 0
  fi
  if [[ ! -s "$checkpoint" || ! -s "$sensitivity_csv" ]]; then
    echo "MISSING input for $run_name"
    return 1
  fi

  echo "START $run_name batch=$batch_size"
  CUDA_VISIBLE_DEVICES="$GPU" python resnet34_pruning_experiments.py \
    --dataset cifar10 \
    --models "$MODEL" \
    --device cuda \
    --amp \
    --amp-dtype "$amp_dtype" \
    --epochs 100 \
    --finetune-epochs 10 \
    --finetune-lr 0.001 \
    --batch-size "$batch_size" \
    --num-workers 2 \
    --pin-memory \
    --prune-ratio 0.7 \
    --tm "$tm" \
    --cross-criteria l1_sum,top2_mean_abs \
    --cross-ratio-source global_cluster \
    --protection-criteria top2_mean_abs \
    --max-layer-prune-ratio 0.75 \
    --sensitivity-csv "$sensitivity_csv" \
    --sensitivity-criteria top2_mean_abs \
    --sensitivity-protect-top-k 4 \
    --sensitivity-skip-top-k 1 \
    --sensitivity-protected-prune-ratio 0.2 \
    --dense-checkpoint-in "$checkpoint" \
    --seed "$seed" \
    --val-size 5000 \
    --split-seed 0 \
    --deterministic \
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
