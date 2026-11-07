#!/bin/bash
cd /home/yenmh3/11457019/cnn_pruning_im2col/runs/breakthrough_r0p95/logs
until [ "$(ls conf_rewind_?.rc conf_group_?.rc conf_group_gradual_?.rc conf_unstructured_?.rc 2>/dev/null | wc -l)" -ge 12 ]; do sleep 60; done
