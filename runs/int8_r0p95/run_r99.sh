#!/bin/bash
cd /home/yenmh3/11457019/cnn_pruning_im2col
export CUDA_VISIBLE_DEVICES=1
/usr/local/anaconda3/bin/python3 scripts/int8_hw_spec_eval.py --models $(cat runs/int8_r0p95/models_r99.txt) --out runs/int8_r0p95/hw_r99_minmax.csv --clip 1 > runs/int8_r0p95/hw_r99_minmax.log 2>&1
/usr/local/anaconda3/bin/python3 scripts/int8_hw_spec_eval.py --models $(cat runs/int8_r0p95/models_r99.txt) --out runs/int8_r0p95/hw_r99_mse.csv --clip .5 > runs/int8_r0p95/hw_r99_mse.log 2>&1
