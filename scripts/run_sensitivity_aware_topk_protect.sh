#!/usr/bin/env bash
set -euo pipefail

# Sensitivity-aware cross-filter pruning.
# Uses runs/layer_sensitivity_summary/layer_sensitivity_ranked.csv to cap/skip
# layers that were empirically sensitive in layer-wise pruning tests.

SEED=${SEED:-0}
EPOCHS=${EPOCHS:-5}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
SENS_CSV=${SENS_CSV:-runs/layer_sensitivity_summary/layer_sensitivity_ranked.csv}
SAVE_ROOT=${SAVE_ROOT:-runs/sensitivity_aware_topk_cifar10_split}
RATIOS=${RATIOS:-"0.6 0.7"}
CRITERIA=${CRITERIA:-max_abs,top2_mean_abs,top4_mean_abs}
PROTECT_TOP_K=${PROTECT_TOP_K:-4}
SKIP_TOP_K=${SKIP_TOP_K:-1}
SENS_CAP=${SENS_CAP:-0.2}
MIN_DROP_PP=${MIN_DROP_PP:-0.0}

mkdir -p "$SAVE_ROOT/logs"

run_job() {
  local gpu="$1"
  local models="$2"
  local ratio="$3"
  local tag="r${ratio/./p}_gpu${gpu}"
  local run_name="sensaware_${tag}_seed${SEED}"
  echo "[$(date '+%F %T')] START gpu=$gpu ratio=$ratio models=$models"
  CUDA_VISIBLE_DEVICES="$gpu" python resnet34_pruning_experiments.py     --dataset cifar10     --models "$models"     --device cuda     --epochs "$EPOCHS"     --finetune-epochs "$FINETUNE_EPOCHS"     --batch-size "$BATCH_SIZE"     --num-workers 0     --prune-ratio "$ratio"     --cross-criteria "$CRITERIA"     --cross-ratio-source global_cluster     --sensitivity-csv "$SENS_CSV"     --sensitivity-protect-top-k "$PROTECT_TOP_K"     --sensitivity-skip-top-k "$SKIP_TOP_K"     --sensitivity-protected-prune-ratio "$SENS_CAP"     --sensitivity-min-mean-drop-pp "$MIN_DROP_PP"     --max-layer-prune-ratio 0.75     --no-cudnn-benchmark     --save-dir "$SAVE_ROOT"     --run-name "$run_name"     > "$SAVE_ROOT/logs/${run_name}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  gpu=$gpu ratio=$ratio models=$models"
}

for ratio in $RATIOS; do
  run_job 0 "resnet34,resnet50,googlenet" "$ratio" &
  run_job 1 "densenet121,mobilenet_v2" "$ratio" &
  wait
 done
