#!/usr/bin/env python3
"""Queue plan Y (hardware S5): cluster vs unstructured vs 2:4, CIFAR-100 ResNet34/ResNet-56, seeds 0/1/2 (24 jobs). Re-running skips queued names."""
import os, time
from pathlib import Path

Q = Path("runs/breakthrough_r0p95/queue.txt")
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
ARMS = {"cluster": "--arm gradual --q-floor", "unstr": "--arm gradual_unstructured", "nm24": "--arm gradual_nm24"}
jobs = []
for m in ("resnet34", "resnet56"):
    for s in (0, 1, 2):
        for a, flags in ARMS.items():
            jobs.append((f"y_{m}_{a}_r50_{s}", f"{DS} --model {m} --seed {s} {flags} --ratio .50"))
        jobs.append((f"y_{m}_unstr_r90_{s}", f"{DS} --model {m} --seed {s} --arm gradual_unstructured --ratio .90"))
rows = Q.read_text().splitlines()
have = {x.split("\t", 1)[0] for x in rows}
new = [j for j in jobs if j[0] not in have]
tmp = Q.with_suffix(".new")
tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in new])+"\n")
os.replace(tmp, Q)
print(len(new), "queued,", len(jobs)-len(new), "already present", time.strftime("%Y-%m-%d %H:%M"))
