#!/bin/bash
cd /home/yenmh3/11457019/cnn_pruning_im2col/runs/breakthrough_r0p95/logs
n=0; for t in ggrad_mac05 dpf group_lamp ggrad_mac025 ggrad_lamp rewind_mac05 group_e100; do for s in 1 2; do :; done; done
until [ "$(ls ggrad_mac05_?.rc dpf_?.rc group_lamp_?.rc ggrad_mac025_?.rc ggrad_lamp_?.rc rewind_mac05_?.rc 2>/dev/null | wc -l)" -ge 12 ]; do sleep 60; done
