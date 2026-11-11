#!/usr/bin/env python3
"""Queue plan T (95% across eight CIFAR-100 models). All Dense checkpoints already exist; 39 jobs.

Run once from the software repo root. Re-running is safe: names already in the queue are skipped.
"""
import os, time
from pathlib import Path

ROOT = Path("runs/breakthrough_r0p95")
Q = ROOT/"queue.txt"
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
ARMS = {"ft_const": "--arm ft_const --discard-model", "rewind": "--arm rewind --discard-model", "floor": "--arm gradual --q-floor"}
PLAN = [  # (model, seeds, arms)
    *[(m, (0, 1, 2), ("ft_const",)) for m in ("resnet18", "resnet50", "resnet56", "resnet110", "vgg16_bn", "googlenet", "densenet121")],
    *[(m, (0, 1, 2), ("rewind", "floor")) for m in ("resnet18", "resnet110")],
    *[(m, (3, 4, 5), ("ft_const",)) for m in ("resnet56", "vgg16_bn")],
]
jobs = [(f"t_{m}_{a}_r95_{s}", f"{DS} --model {m} --seed {s} {ARMS[a]} --ratio .95") for m, seeds, arms in PLAN for s in seeds for a in arms]
assert len(jobs) == 39, len(jobs)
rows = Q.read_text().splitlines()
have = {r.split("\t", 1)[0] for r in rows}
new = [j for j in jobs if j[0] not in have]
tmp = Q.with_suffix(".new")
tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in new])+"\n")
os.replace(tmp, Q)
print(len(new), "queued,", len(jobs)-len(new), "already present", time.strftime("%Y-%m-%d %H:%M"))
