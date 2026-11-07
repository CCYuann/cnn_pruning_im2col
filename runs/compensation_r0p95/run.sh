#!/bin/bash
# Prepares masks one seed at a time, then runs all A/B FT10 branches across both GPUs.
cd /home/yenmh3/11457019/cnn_pruning_im2col
R=runs/compensation_r0p95
W="python3 scripts/compensation_ratio_worker.py"
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=1 $W prepare --seed $s --ratio 0.95 --root $R > $R/logs/prepare_$s.log 2>&1 || { echo "prepare $s failed"; exit 1; }
done
i=0; pids=()
for s in 0 1 2; do for v in original dense_input; do
  CUDA_VISIBLE_DEVICES=$((i%2)) OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $W train --seed $s --variant $v --ratio 0.95 --root $R > $R/logs/train_${s}_$v.log 2>&1 &
  pids+=($!); i=$((i+1))
done; done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
echo "all done rc=$rc"; exit $rc
