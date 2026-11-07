#!/bin/bash
cd /home/yenmh3/11457019/cnn_pruning_im2col/runs/breakthrough_r0p95/logs
until [ "$(ls d[1-5]_*.rc conf98_unstructured_?.rc next_dpf_group*_2.rc 2>/dev/null | wc -l)" -ge 22 ]; do sleep 60; done
