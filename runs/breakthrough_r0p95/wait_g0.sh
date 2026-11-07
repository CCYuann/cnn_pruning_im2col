#!/bin/bash
cd /home/yenmh3/11457019/cnn_pruning_im2col/runs/breakthrough_r0p95
until [ "$(ls logs/g0_dense_r34_?.rc 2>/dev/null | wc -l)" -ge 3 ]; do sleep 60; done
for s in 3 4 5; do [ "$(cat logs/g0_dense_r34_$s.rc)" = 0 ] || { echo "dense $s failed"; exit 1; }; done
P=/usr/local/anaconda3/bin/python3; W=scripts/retrain_recovery_worker.py; D=/home/yenmh3/11457019/cnn_pruning_im2col/runs
for s in 3 4 5; do ln -s $D/fixed_checkpoint_resnet34_new/checkpoints/resnet34_seed${s}_dense.pt $D/fixed_checkpoint_resnet34/checkpoints/resnet34_seed${s}_dense.pt; done
{ for s in 3 4 5; do
  for row in "rewind .8 _r34" "rewind .95 _r34r95" "rewind .98 _r34r98" "gradual .98 _r34r98" "unstructured .98 _r34r98"; do set -- $row
    printf 'g2_%s%s_%s\t%s %s --root runs/confirmation_r0p95 --seed %s --group-lambda 3e-5 --arm %s --model resnet34 --ratio %s --tag %s\n' $1 $3 $s $P $W $s $1 $2 $3
  done; done
  cat queue.txt; } > queue.new && mv queue.new queue.txt
until [ "$(ls logs/g2_*.rc 2>/dev/null | wc -l)" -ge 15 ]; do sleep 60; done
date
