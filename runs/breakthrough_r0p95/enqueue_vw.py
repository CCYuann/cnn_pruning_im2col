#!/usr/bin/env python3
"""Queue plan V (plan U swap controls on seeds 3/4/5) and plan W (ft_const LR grid at 90%). Re-running skips queued names.

Usage: enqueue_vw.py           -> plan V only (12 jobs)
       enqueue_vw.py --with-w   -> plan V and plan W (12 + 72 jobs)
"""
import os, sys, time
from pathlib import Path

Q = Path("runs/breakthrough_r0p95/queue.txt")
R = "runs/other_datasets"
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
jobs = []
for m in ("resnet56", "vgg16_bn"):
    for s in (3, 4, 5):
        src = f"{R}/cifar100_{m}_seed{s}_gradual_r95_floor"
        for key, flag in (("qfrom", "--quota-from"), ("mfrom", "--mask-from")):
            jobs.append((f"v_{m}_{key}_r95_{s}", f"{DS} --model {m} --seed {s} --arm rewind --ratio .95 {flag} {src} --discard-model"))
if "--with-w" in sys.argv:
    # Slow models first so the tail of the queue is short.
    for m in ("densenet121", "resnet50", "resnet110", "googlenet", "resnet34", "resnet18", "vgg16_bn", "resnet56"):
        for lr in (".003", ".01", ".03"):
            for s in (0, 1, 2):
                jobs.append((f"w_{m}_ftlr{lr[1:]}_r90_{s}", f"{DS} --model {m} --seed {s} --arm ft_const --ratio .90 --ft-lr {lr} --discard-model"))
rows = Q.read_text().splitlines()
have = {x.split("\t", 1)[0] for x in rows}
new = [j for j in jobs if j[0] not in have]
tmp = Q.with_suffix(".new")
tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in new])+"\n")
os.replace(tmp, Q)
print(len(new), "queued,", len(jobs)-len(new), "already present", time.strftime("%Y-%m-%d %H:%M"))
