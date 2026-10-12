#!/usr/bin/env bash
set -uo pipefail

GPU=${GPU:-1}
MAX_JOBS=${MAX_JOBS:-6}
RATIOS=${RATIOS:-"0.85 0.90"}
SAVE_ROOT=${SAVE_ROOT:-runs/fixed_checkpoint_resnet34_r0p85_r0p90}

mkdir -p "$SAVE_ROOT/logs"

ratio_tag() {
  printf 'r%s' "${1/./p}"
}

run_job() {
  local ratio=$1
  local seed=$2
  local tm=$3
  local tag
  tag=$(ratio_tag "$ratio")
  local run_name="resnet34_seed${seed}_tm${tm}_${tag}"
  local checkpoint="runs/fixed_checkpoint_resnet34/checkpoints/resnet34_seed${seed}_dense.pt"
  local sensitivity_csv="runs/fixed_checkpoint_resnet34/sensitivity_validation/seed${seed}_tm${tm}_top2/layer_sensitivity_ranked.csv"

  if [[ -s "$SAVE_ROOT/${run_name}_results.csv" ]]; then
    echo "SKIP $run_name result exists"
    return 0
  fi
  if [[ ! -s "$checkpoint" || ! -s "$sensitivity_csv" ]]; then
    echo "MISSING input for $run_name"
    return 1
  fi

  echo "START $run_name"
  CUDA_VISIBLE_DEVICES="$GPU" python resnet34_pruning_experiments.py \
    --dataset cifar10 \
    --models resnet34 \
    --device cuda \
    --amp \
    --amp-dtype float16 \
    --epochs 100 \
    --finetune-epochs 10 \
    --finetune-lr 0.001 \
    --batch-size 128 \
    --num-workers 2 \
    --pin-memory \
    --prune-ratio "$ratio" \
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
    --save-dir "$SAVE_ROOT" \
    --run-name "$run_name" \
    > "$SAVE_ROOT/logs/${run_name}.log" 2>&1
  local status=$?
  echo "END $run_name status=$status"
  return "$status"
}

running=0
failed=0
for ratio in $RATIOS; do
  for seed in 0 1 2; do
    for tm in 8 16; do
      run_job "$ratio" "$seed" "$tm" &
      running=$((running + 1))
      if (( running >= MAX_JOBS )); then
        wait -n || failed=1
        running=$((running - 1))
      fi
    done
  done
done

while (( running > 0 )); do
  wait -n || failed=1
  running=$((running - 1))
done

exit "$failed"
