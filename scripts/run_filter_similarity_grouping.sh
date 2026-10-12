#!/usr/bin/env bash
set -uo pipefail

GPU=${GPU:-0}
SHARD=${SHARD:-0}
SHARDS=${SHARDS:-2}
MAX_JOBS=${MAX_JOBS:-5}
PYTHON_BIN=${PYTHON_BIN:-python}
MODELS=${MODELS:-"resnet34 resnet50 googlenet densenet121 mobilenet_v2"}
RATIOS=${RATIOS:-"0.80 0.85 0.90"}
SAVE_ROOT=${SAVE_ROOT:-runs/filter_similarity_grouping}

mkdir -p "$SAVE_ROOT/logs" "$SAVE_ROOT/locks"

ratio_tag() {
  printf 'r%s' "${1/./p}"
}

checkpoint_for() {
  local model=$1
  local seed=$2
  if [[ "$model" == "resnet34" ]]; then
    printf 'runs/fixed_checkpoint_resnet34/checkpoints/resnet34_seed%s_dense.pt' "$seed"
  else
    printf 'runs/fixed_checkpoint_multi_model/checkpoints/%s_seed%s_dense.pt' "$model" "$seed"
  fi
}

run_job() {
  local model=$1
  local ratio=$2
  local seed=$3
  local tag run_name result checkpoint log
  tag=$(ratio_tag "$ratio")
  run_name="${model}_seed${seed}_tm8_${tag}_similarity_jaccard"
  result="$SAVE_ROOT/${run_name}_results.csv"
  checkpoint=$(checkpoint_for "$model" "$seed")
  log="$SAVE_ROOT/logs/${run_name}.log"

  if [[ -s "$result" ]]; then
    echo "SKIP $run_name result exists"
    return 0
  fi
  if [[ ! -s "$checkpoint" ]]; then
    echo "MISSING checkpoint: $checkpoint"
    return 1
  fi

  exec 9>"$SAVE_ROOT/locks/${run_name}.lock"
  if ! flock -n 9; then
    echo "LOCKED $run_name by another scheduler"
    return 0
  fi

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
    --tm 8 \
    --cross-criteria l1_sum \
    --cross-ratio-source global_cluster \
    --filter-grouping similarity_jaccard \
    --cross-only \
    --dense-checkpoint-in "$checkpoint" \
    --seed "$seed" \
    --val-size 5000 \
    --split-seed 0 \
    --deterministic \
    --no-cudnn-benchmark \
    --save-dir "$SAVE_ROOT" \
    --run-name "$run_name" \
    >"$log" 2>&1
  local status=$?
  echo "END $run_name GPU=$GPU status=$status"
  return "$status"
}

index=0
running=0
failed=0
for model in $MODELS; do
  for ratio in $RATIOS; do
    for seed in 0 1 2; do
      if (( index % SHARDS == SHARD )); then
        run_job "$model" "$ratio" "$seed" &
        running=$((running + 1))
        if (( running >= MAX_JOBS )); then
          wait -n || failed=1
          running=$((running - 1))
        fi
      fi
      index=$((index + 1))
    done
  done
done

while (( running > 0 )); do
  wait -n || failed=1
  running=$((running - 1))
done

exit "$failed"
