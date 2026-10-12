#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Usage examples:
#   DATASET=animals10 DATA_DIR=/path/to/animals10 bash scripts/run_imagefolder_dataset_sweep.sh
#   DATASET=imagenette DATA_DIR=/path/to/imagenette2-160 bash scripts/run_imagefolder_dataset_sweep.sh
#   DATASET=imagefolder DATA_DIR=/path/to/custom_imagefolder bash scripts/run_imagefolder_dataset_sweep.sh
#
# Expected ImageFolder formats:
#   root/train/class_x/*.jpg and root/val/class_x/*.jpg
#   root/train/class_x/*.jpg and root/test/class_x/*.jpg
# If train/val or train/test are absent, the script will use the Python code's 80/20 split.

DATASET=${DATASET:-animals10}
DATA_DIR=${DATA_DIR:-}
if [[ -z "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR is required. Example: DATASET=animals10 DATA_DIR=/path/to/animals10 bash $0" >&2
  exit 1
fi

RATIOS=${RATIOS:-"0.6"}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-5}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
IMAGE_SIZE=${IMAGE_SIZE:-32}
MODELS=${MODELS:-"resnet34,googlenet,densenet121"}
SAVE_ROOT=${SAVE_ROOT:-"runs/${DATASET}_sweep"}
GPU=${GPU:-1}

case "$DATASET" in
  animals10)
    DATA_ARG=(--dataset animals10 --animals-dir "$DATA_DIR")
    ;;
  imagenette)
    DATA_ARG=(--dataset imagenette --imagenette-dir "$DATA_DIR")
    ;;
  imagefolder)
    DATA_ARG=(--dataset imagefolder --imagefolder-dir "$DATA_DIR")
    ;;
  *)
    echo "ERROR: DATASET must be animals10, imagenette, or imagefolder" >&2
    exit 1
    ;;
esac

mkdir -p "$SAVE_ROOT"

for ratio in $RATIOS; do
  ratio_tag=${ratio/./p}
  save_dir="$SAVE_ROOT/r${ratio_tag}_seed${SEED}"
  mkdir -p "$save_dir"

  echo "=== $DATASET ratio=$ratio seed=$SEED models=$MODELS gpu=$GPU ==="
  CUDA_VISIBLE_DEVICES="$GPU" python resnet34_pruning_experiments.py     "${DATA_ARG[@]}"     --models "$MODELS"     --device cuda     --image-size "$IMAGE_SIZE"     --epochs "$EPOCHS"     --finetune-epochs "$FINETUNE_EPOCHS"     --batch-size "$BATCH_SIZE"     --num-workers 0     --prune-ratio "$ratio"     --cross-criteria l1_sum,max_abs     --cross-ratio-source global_cluster     --no-cudnn-benchmark     --save-dir "$save_dir"     --run-name "${DATASET}_seed${SEED}_r${ratio_tag}"

done
