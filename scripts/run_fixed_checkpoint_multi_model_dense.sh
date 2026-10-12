#!/usr/bin/env bash
set -uo pipefail

GPU=${GPU:-1}
MAX_JOBS=${MAX_JOBS:-2}
ROOT=${ROOT:-runs/fixed_checkpoint_multi_model}
LOG_DIR="$ROOT/logs/dense"
MODELS=(mobilenet_v2 googlenet densenet121 resnet50 efficientnet_v2_m)

mkdir -p "$ROOT/checkpoints" "$LOG_DIR"

run_job() {
  local model=$1
  local seed=$2
  local batch_size=128
  local lr=0.1
  local amp_dtype=float16
  if [[ "$model" == "efficientnet_v2_m" ]]; then
    batch_size=64
    lr=0.02
    amp_dtype=bfloat16
  fi

  local run_name="dense_${model}_seed${seed}"
  local checkpoint="$ROOT/checkpoints/${model}_seed${seed}_dense.pt"
  if [[ -s "$checkpoint" ]]; then
    echo "SKIP $run_name checkpoint exists"
    return 0
  fi

  echo "START $run_name batch=$batch_size"
  CUDA_VISIBLE_DEVICES="$GPU" python resnet34_pruning_experiments.py \
    --dataset cifar10 \
    --models "$model" \
    --device cuda \
    --amp \
    --amp-dtype "$amp_dtype" \
    --epochs 100 \
    --lr "$lr" \
    --lr-scheduler cosine \
    --min-lr 0.0001 \
    --batch-size "$batch_size" \
    --num-workers 2 \
    --pin-memory \
    --seed "$seed" \
    --val-size 5000 \
    --split-seed 0 \
    --deterministic \
    --no-cudnn-benchmark \
    --train-dense-only \
    --dense-checkpoint-out "$checkpoint" \
    --save-dir "$ROOT" \
    --run-name "$run_name" \
    > "$LOG_DIR/${run_name}.log" 2>&1
  local status=$?
  echo "END $run_name status=$status"
  return "$status"
}

running=0
failed=0
for model in "${MODELS[@]}"; do
  for seed in 0 1 2; do
    run_job "$model" "$seed" &
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
