#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RATIOS=${RATIOS:-"0.5 0.7 0.8"}
MODELS=${MODELS:-"resnet34 resnet50 googlenet densenet121 mobilenet_v2"}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-5}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_PARALLEL=${MAX_PARALLEL:-5}
SAVE_ROOT=${SAVE_ROOT:-runs/ratio_sweep_cifar10_gpu1_split}
LOG_DIR="$SAVE_ROOT/logs"
mkdir -p "$LOG_DIR"

run_job() {
  local model="$1"
  local ratio="$2"
  local ratio_tag="${ratio/./p}"
  local save_dir="$SAVE_ROOT/r${ratio_tag}_seed${SEED}"
  local run_name="gpu1_${model}_r${ratio_tag}_seed${SEED}"
  mkdir -p "$save_dir"
  echo "[$(date '+%F %T')] START model=$model ratio=$ratio seed=$SEED"
  CUDA_VISIBLE_DEVICES=1 python resnet34_pruning_experiments.py     --dataset cifar10     --models "$model"     --device cuda     --epochs "$EPOCHS"     --finetune-epochs "$FINETUNE_EPOCHS"     --batch-size "$BATCH_SIZE"     --num-workers 0     --prune-ratio "$ratio"     --cross-criteria l1_sum,max_abs     --cross-ratio-source global_cluster     --no-cudnn-benchmark     --save-dir "$save_dir"     --run-name "$run_name"     > "$LOG_DIR/${run_name}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  model=$model ratio=$ratio seed=$SEED"
}

running=0
for ratio in $RATIOS; do
  for model in $MODELS; do
    run_job "$model" "$ratio" &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

python scripts/aggregate_pruning_results.py "$SAVE_ROOT" --out "$SAVE_ROOT/all_results_combined.csv" | tee "$SAVE_ROOT/aggregate_summary.txt"
echo "[$(date '+%F %T')] ALL DONE -> $SAVE_ROOT"
