#!/usr/bin/env bash
set -u
ROOT="runs/improvement_topk_protect_cifar10_split"
mkdir -p "$ROOT/logs"
COMMON="--dataset cifar10 --device cuda --epochs 5 --finetune-epochs 2 --batch-size 128 --num-workers 0 --prune-ratio 0.7 --cross-ratio-source global_cluster --no-cudnn-benchmark --save-dir $ROOT"
echo "Launching CIFAR-10 improvement r0p7 jobs on GPU0"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models resnet34 --cross-criteria top2_mean_abs,top4_mean_abs --run-name a_topk_resnet34_r0p7_seed0 > $ROOT/logs/a_topk_resnet34_r0p7_seed0.log 2>&1 &
echo "A topk r0p7 resnet34 PID $! log $ROOT/logs/a_topk_resnet34_r0p7_seed0.log"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models resnet50 --cross-criteria top2_mean_abs,top4_mean_abs --run-name a_topk_resnet50_r0p7_seed0 > $ROOT/logs/a_topk_resnet50_r0p7_seed0.log 2>&1 &
echo "A topk r0p7 resnet50 PID $! log $ROOT/logs/a_topk_resnet50_r0p7_seed0.log"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models googlenet --cross-criteria top2_mean_abs,top4_mean_abs --run-name a_topk_googlenet_r0p7_seed0 > $ROOT/logs/a_topk_googlenet_r0p7_seed0.log 2>&1 &
echo "A topk r0p7 googlenet PID $! log $ROOT/logs/a_topk_googlenet_r0p7_seed0.log"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models densenet121 --cross-criteria top2_mean_abs,top4_mean_abs --run-name a_topk_densenet121_r0p7_seed0 > $ROOT/logs/a_topk_densenet121_r0p7_seed0.log 2>&1 &
echo "A topk r0p7 densenet121 PID $! log $ROOT/logs/a_topk_densenet121_r0p7_seed0.log"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models mobilenet_v2 --cross-criteria top2_mean_abs,top4_mean_abs --run-name a_topk_mobilenet_v2_r0p7_seed0 > $ROOT/logs/a_topk_mobilenet_v2_r0p7_seed0.log 2>&1 &
echo "A topk r0p7 mobilenet_v2 PID $! log $ROOT/logs/a_topk_mobilenet_v2_r0p7_seed0.log"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models resnet34 --cross-criteria max_abs,top2_mean_abs,top4_mean_abs --protect-early-layers 2 --protected-prune-ratio 0.2 --max-layer-prune-ratio 0.75 --run-name b_protect_cap_resnet34_r0p7_seed0 > $ROOT/logs/b_protect_cap_resnet34_r0p7_seed0.log 2>&1 &
echo "B protect+cap r0p7 resnet34 PID $! log $ROOT/logs/b_protect_cap_resnet34_r0p7_seed0.log"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models resnet50 --cross-criteria max_abs,top2_mean_abs,top4_mean_abs --protect-early-layers 2 --protected-prune-ratio 0.2 --max-layer-prune-ratio 0.75 --run-name b_protect_cap_resnet50_r0p7_seed0 > $ROOT/logs/b_protect_cap_resnet50_r0p7_seed0.log 2>&1 &
echo "B protect+cap r0p7 resnet50 PID $! log $ROOT/logs/b_protect_cap_resnet50_r0p7_seed0.log"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models googlenet --cross-criteria max_abs,top2_mean_abs,top4_mean_abs --protect-early-layers 2 --protected-prune-ratio 0.2 --max-layer-prune-ratio 0.75 --run-name b_protect_cap_googlenet_r0p7_seed0 > $ROOT/logs/b_protect_cap_googlenet_r0p7_seed0.log 2>&1 &
echo "B protect+cap r0p7 googlenet PID $! log $ROOT/logs/b_protect_cap_googlenet_r0p7_seed0.log"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models densenet121 --cross-criteria max_abs,top2_mean_abs,top4_mean_abs --protect-early-layers 2 --protected-prune-ratio 0.2 --max-layer-prune-ratio 0.75 --run-name b_protect_cap_densenet121_r0p7_seed0 > $ROOT/logs/b_protect_cap_densenet121_r0p7_seed0.log 2>&1 &
echo "B protect+cap r0p7 densenet121 PID $! log $ROOT/logs/b_protect_cap_densenet121_r0p7_seed0.log"
CUDA_VISIBLE_DEVICES=0 python resnet34_pruning_experiments.py $COMMON --models mobilenet_v2 --cross-criteria max_abs,top2_mean_abs,top4_mean_abs --protect-early-layers 2 --protected-prune-ratio 0.2 --max-layer-prune-ratio 0.75 --run-name b_protect_cap_mobilenet_v2_r0p7_seed0 > $ROOT/logs/b_protect_cap_mobilenet_v2_r0p7_seed0.log 2>&1 &
echo "B protect+cap r0p7 mobilenet_v2 PID $! log $ROOT/logs/b_protect_cap_mobilenet_v2_r0p7_seed0.log"
wait
echo "All GPU0 r0p7 improvement jobs finished"
