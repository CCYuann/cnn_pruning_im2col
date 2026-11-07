#!/bin/bash
cd /home/yenmh3/11457019/cnn_pruning_im2col/runs/breakthrough_r0p95/logs
until [ "$(ls f_*.rc 2>/dev/null | wc -l)" -ge 10 ]; do sleep 60; done
