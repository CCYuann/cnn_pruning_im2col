#!/usr/bin/env bash
set -uo pipefail

GPU=${GPU:-1}
MAX_JOBS=${MAX_JOBS:-6}
PYTHON_BIN=${PYTHON_BIN:-python}
MODELS=${MODELS:-"resnet34 resnet50 googlenet densenet121 mobilenet_v2"}
RATIOS=${RATIOS:-"0.85 0.90 0.95"}
SAVE_ROOT=${SAVE_ROOT:-runs/fixed_checkpoint_high_ratio_sweep}
RESNET34_EXISTING_ROOT=${RESNET34_EXISTING_ROOT:-runs/fixed_checkpoint_resnet34_r0p85_r0p90}

mkdir -p "$SAVE_ROOT/logs" "$SAVE_ROOT/locks"

ratio_tag() {
  printf 'r%s' "${1/./p}"
}

run_job() {
  local model=$1
  local ratio=$2
  local seed=$3
  local tm=$4
  local tag
  tag=$(ratio_tag "$ratio")
  local run_name="${model}_seed${seed}_tm${tm}_${tag}"
  local result="$SAVE_ROOT/${run_name}_results.csv"
  local log_dir="$SAVE_ROOT/logs/$model/$tag"
  local checkpoint
  local sensitivity_csv

  if [[ -s "$result" ]]; then
    echo "SKIP $run_name result exists"
    return 0
  fi
  if [[ "$model" == "resnet34" && "$ratio" != "0.95" ]]; then
    local existing="$RESNET34_EXISTING_ROOT/${run_name}_results.csv"
    if [[ -s "$existing" ]]; then
      echo "REUSE $run_name from $RESNET34_EXISTING_ROOT"
      return 0
    fi
  fi

  exec 9>"$SAVE_ROOT/locks/${run_name}.lock"
  if ! flock -n 9; then
    echo "LOCKED $run_name by another scheduler"
    return 0
  fi
  if [[ -s "$result" ]]; then
    echo "SKIP $run_name result created by another scheduler"
    return 0
  fi

  if [[ "$model" == "resnet34" ]]; then
    checkpoint="runs/fixed_checkpoint_resnet34/checkpoints/resnet34_seed${seed}_dense.pt"
    sensitivity_csv="runs/fixed_checkpoint_resnet34/sensitivity_validation/seed${seed}_tm${tm}_top2/layer_sensitivity_ranked.csv"
  else
    checkpoint="runs/fixed_checkpoint_multi_model/checkpoints/${model}_seed${seed}_dense.pt"
    sensitivity_csv="runs/fixed_checkpoint_multi_model/sensitivity_validation/${model}_seed${seed}_tm${tm}_top2/layer_sensitivity_ranked.csv"
  fi
  if [[ ! -s "$checkpoint" || ! -s "$sensitivity_csv" ]]; then
    echo "MISSING input for $run_name"
    return 1
  fi

  mkdir -p "$log_dir"
  echo "START $run_name GPU=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" resnet34_pruning_experiments.py \
    --dataset cifar10 \
    --models "$model" \
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
    > "$log_dir/${run_name}.log" 2>&1
  local status=$?
  echo "END $run_name GPU=$GPU status=$status"
  return "$status"
}

running=0
failed=0
for model in $MODELS; do
  for ratio in $RATIOS; do
    for seed in 0 1 2; do
      for tm in 8 16; do
        run_job "$model" "$ratio" "$seed" "$tm" &
        running=$((running + 1))
        if (( running >= MAX_JOBS )); then
          wait -n || failed=1
          running=$((running - 1))
        fi
      done
    done
  done
done

while (( running > 0 )); do
  wait -n || failed=1
  running=$((running - 1))
done

exit "$failed"
