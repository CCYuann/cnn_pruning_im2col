#!/usr/bin/env bash
set -uo pipefail

GPU=${GPU:-1}
MAX_JOBS=${MAX_JOBS:-4}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-10}
ROOT=${ROOT:-runs/fixed_checkpoint_resnet34}
SAVE_DIR="$ROOT/pruning_comparison"
LOG_DIR="$SAVE_DIR/logs"

mkdir -p "$LOG_DIR"

run_job() {
  local seed=$1
  local tm=$2
  local ratio=$3
  local ratio_tag=${ratio/./p}
  local run_name="seed${seed}_tm${tm}_r${ratio_tag}"
  local checkpoint="$ROOT/checkpoints/resnet34_seed${seed}_dense.pt"
  local sensitivity_csv="$ROOT/sensitivity_validation/seed${seed}_tm${tm}_top2/layer_sensitivity_ranked.csv"

  echo "START $run_name"
  CUDA_VISIBLE_DEVICES="$GPU" python resnet34_pruning_experiments.py \
    --dataset cifar10 \
    --models resnet34 \
    --device cuda \
    --amp \
    --epochs 100 \
    --finetune-epochs "$FINETUNE_EPOCHS" \
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
    for ratio in 0.5 0.6 0.7; do
      run_job "$seed" "$tm" "$ratio" &
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
