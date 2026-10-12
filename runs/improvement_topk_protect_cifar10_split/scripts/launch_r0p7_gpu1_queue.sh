#!/usr/bin/env bash
set -u

ROOT="runs/improvement_topk_protect_cifar10_split"
mkdir -p "$ROOT/logs"

MAX_PARALLEL="${MAX_PARALLEL:-4}"
GPU="${GPU:-1}"
COMMON="--dataset cifar10 --device cuda --epochs 5 --finetune-epochs 2 --batch-size 128 --num-workers 0 --prune-ratio 0.7 --cross-ratio-source global_cluster --no-cudnn-benchmark --save-dir $ROOT"

running=0

launch_job() {
  local run_name="$1"
  shift
  local log="$ROOT/logs/${run_name}_rerun_gpu${GPU}.log"
  CUDA_VISIBLE_DEVICES="$GPU" python -u resnet34_pruning_experiments.py $COMMON "$@" --run-name "$run_name" > "$log" 2>&1 &
  echo "Launched $run_name on GPU$GPU PID $! log $log"
  running=$((running + 1))
  if [ "$running" -ge "$MAX_PARALLEL" ]; then
    wait -n
    running=$((running - 1))
  fi
}

echo "Launching r0p7 improvement jobs on GPU$GPU with MAX_PARALLEL=$MAX_PARALLEL"

for model in resnet34 resnet50 googlenet densenet121 mobilenet_v2; do
  launch_job "a_topk_${model}_r0p7_seed0" \
    --models "$model" \
    --cross-criteria top2_mean_abs,top4_mean_abs
done

for model in resnet34 resnet50 googlenet densenet121 mobilenet_v2; do
  launch_job "b_protect_cap_${model}_r0p7_seed0" \
    --models "$model" \
    --cross-criteria max_abs,top2_mean_abs,top4_mean_abs \
    --protect-early-layers 2 \
    --protected-prune-ratio 0.2 \
    --max-layer-prune-ratio 0.75
done

wait
echo "All r0p7 GPU$GPU improvement jobs finished"
