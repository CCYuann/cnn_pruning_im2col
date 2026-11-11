#!/usr/bin/env python3
"""Queue plan O: fill15 on five CIFAR-100 models, VGG16-BN one-shot + floor."""
import os
from pathlib import Path

Q = Path("runs/breakthrough_r0p95/queue.txt")
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
jobs = [(f"o1_{m}_fill15_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm gradual --ratio .{r} --q-floor --q-floor-n 15 --free-fill --discard-model")
        for m in ("resnet56", "vgg16_bn", "googlenet", "densenet121", "resnet34") for r in (98, 99) for s in (0, 1, 2)]
jobs += [(f"o2_vgg16_bn_rewind_floor_r{r}_{s}", f"{DS} --model vgg16_bn --seed {s} --arm rewind --ratio .{r} --q-floor --discard-model") for r in (98, 99) for s in (0, 1, 2)]
rows = Q.read_text().splitlines()
tmp = Q.with_suffix(".new")
tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in jobs])+"\n")
os.replace(tmp, Q)
print(len(jobs), "queued")
