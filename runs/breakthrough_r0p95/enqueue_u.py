#!/usr/bin/env python3
"""Queue plan U (factor decomposition of gradual+floor vs rewind). 42 jobs; re-running skips names already queued."""
import os, time
from pathlib import Path

Q = Path("runs/breakthrough_r0p95/queue.txt")
R = "runs/other_datasets"
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
SETS = [("resnet56", 95), ("vgg16_bn", 95), ("googlenet", 95), ("densenet121", 95), ("resnet110", 95), ("resnet34", 98), ("densenet121", 90)]
jobs = []
for m, r in SETS:
    for s in (0, 1, 2):
        src = f"{R}/cifar100_{m}_seed{s}_gradual_r{r}_floor"
        for key, flag in (("qfrom", "--quota-from"), ("mfrom", "--mask-from")):
            jobs.append((f"u_{m}_{key}_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm rewind --ratio .{r} {flag} {src} --discard-model"))
assert len(jobs) == 42
rows = Q.read_text().splitlines()
have = {x.split("\t", 1)[0] for x in rows}
new = [j for j in jobs if j[0] not in have]
tmp = Q.with_suffix(".new")
tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in new])+"\n")
os.replace(tmp, Q)
print(len(new), "queued", time.strftime("%Y-%m-%d %H:%M"))
