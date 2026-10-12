#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Default sweep: keep EfficientNet out because it is much slower and did not converge in 5 epochs.
# Override from terminal if needed, for example:
#   RATIOS="0.5 0.7 0.8 0.9" SEED=1 bash scripts/run_ratio_sweep_cifar10.sh
RATIOS=${RATIOS:-"0.5 0.7 0.8"}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-5}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
SAVE_ROOT=${SAVE_ROOT:-runs/ratio_sweep_cifar10}

GPU1_MODELS=${GPU1_MODELS:-"resnet34,googlenet,mobilenet_v2"}
GPU0_MODELS=${GPU0_MODELS:-"resnet50,densenet121"}

mkdir -p "$SAVE_ROOT"

for ratio in $RATIOS; do
  ratio_tag=${ratio/./p}
  save_dir="$SAVE_ROOT/r${ratio_tag}_seed${SEED}"
  mkdir -p "$save_dir"

  echo "=== CIFAR-10 ratio=$ratio seed=$SEED ==="
  echo "GPU1 models: $GPU1_MODELS"
  echo "GPU0 models: $GPU0_MODELS"

  CUDA_VISIBLE_DEVICES=1 python resnet34_pruning_experiments.py     --dataset cifar10     --models "$GPU1_MODELS"     --device cuda     --epochs "$EPOCHS"     --finetune-epochs "$FINETUNE_EPOCHS"     --batch-size "$BATCH_SIZE"     --num-workers 0     --prune-ratio "$ratio"     --cross-criteria l1_sum,max_abs     --cross-ratio-source global_cluster     --no-cudnn-benchmark     --save-dir "$save_dir"     --run-name "gpu1_seed${SEED}_r${ratio_tag}" &
  pid_gpu1=$!

  CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py     --dataset cifar10     --models "$GPU0_MODELS"     --device cuda     --epochs "$EPOCHS"     --finetune-epochs "$FINETUNE_EPOCHS"     --batch-size "$BATCH_SIZE"     --num-workers 0     --prune-ratio "$ratio"     --cross-criteria l1_sum,max_abs     --cross-ratio-source global_cluster     --no-cudnn-benchmark     --save-dir "$save_dir"     --run-name "gpu0_seed${SEED}_r${ratio_tag}" &
  pid_gpu0=$!

  wait "$pid_gpu1"
  wait "$pid_gpu0"
  echo "=== Finished ratio=$ratio seed=$SEED -> $save_dir ==="
done
