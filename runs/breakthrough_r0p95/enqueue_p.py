#!/usr/bin/env python3
"""Queue plan P (92%, 95%) ahead of the unstarted plan O jobs; 95% first, faster models first."""
import os
from pathlib import Path

Q = Path("runs/breakthrough_r0p95/queue.txt")
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
ARMS = {"rewind": "--arm rewind", "floor": "--arm gradual --q-floor", "fill15": "--arm gradual --q-floor --q-floor-n 15 --free-fill",
        "gu": "--arm gradual_unstructured"}
jobs = []
for r in (95, 92):
    for m in ("vgg16_bn", "resnet56", "googlenet", "resnet34", "densenet121"):
        for arm, flags in ARMS.items():
            if m == "resnet34" and r == 95 and arm in ("rewind", "floor"):
                continue
            keep = "" if arm == "floor" else " --discard-model"
            jobs += [(f"p_{m}_{arm}_r{r}_{s}", f"{DS} --model {m} --seed {s} {flags} --ratio .{r}{keep}") for s in (0, 1, 2)]
rows = [l for l in Q.read_text().splitlines() if l.strip()]
head = [l for l in rows if not l.startswith("o")]
tail = [l for l in rows if l.startswith("o")]
tmp = Q.with_suffix(".new")
tmp.write_text("\n".join(head+[f"{n}\t{c}" for n, c in jobs]+tail)+"\n")
os.replace(tmp, Q)
print(len(jobs), "queued")
