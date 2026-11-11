#!/usr/bin/env python3
"""Queue plan X: ft_const at LR 0.0003, 90%, eight CIFAR-100 models, seeds 0/1/2 (24 jobs). Re-running skips queued names."""
import os, time
from pathlib import Path

Q = Path("runs/breakthrough_r0p95/queue.txt")
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
jobs = [(f"x_{m}_ftlr0003_r90_{s}", f"{DS} --model {m} --seed {s} --arm ft_const --ratio .90 --ft-lr .0003 --discard-model")
        for m in ("densenet121", "resnet50", "resnet110", "googlenet", "resnet34", "resnet18", "vgg16_bn", "resnet56") for s in (0, 1, 2)]
rows = Q.read_text().splitlines()
have = {x.split("\t", 1)[0] for x in rows}
new = [j for j in jobs if j[0] not in have]
tmp = Q.with_suffix(".new")
tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in new])+"\n")
os.replace(tmp, Q)
print(len(new), "queued,", len(jobs)-len(new), "already present", time.strftime("%Y-%m-%d %H:%M"))
