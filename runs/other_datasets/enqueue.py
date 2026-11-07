#!/usr/bin/env python3
"""Insert dataset jobs after G2 and before exploratory jobs; prune jobs follow once all Dense checkpoints exist."""
import os, sys, time
from pathlib import Path
Q, LOGS = Path("runs/breakthrough_r0p95/queue.txt"), Path("runs/breakthrough_r0p95/logs")
PY = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py"
PAIRS = [p.split(":") for p in sys.argv[1:]]
SEEDS = (0, 1, 2)


def insert(lines):
    rows = Q.read_text().splitlines()
    at = max((i+1 for i, r in enumerate(rows) if r.startswith(("g2_", "ds_"))), default=0)
    tmp = Q.with_suffix(".new")
    tmp.write_text("\n".join(rows[:at]+lines+rows[at:])+"\n")
    os.replace(tmp, Q)


dense = [f"ds_{d}_{m}_dense_{s}\t{PY} --dataset {d} --model {m} --seed {s} --arm dense" for d, m in PAIRS for s in SEEDS]
insert(dense)
names = [line.split("\t")[0] for line in dense]
while not all((LOGS/f"{n}.rc").exists() for n in names):
    time.sleep(60)
failed = [n for n in names if (LOGS/f"{n}.rc").read_text() != "0"]
if failed:
    sys.exit(f"Dense failed: {failed}")
prune = []
for d, m in PAIRS:
    for s in SEEDS:
        for r in (90, 95, 98, 99):
            for arm, flag in (("rewind", ""), ("gradual", " --q-floor"), ("unstructured", "")):
                prune.append(f"ds_{d}_{m}_{arm}_r{r}_{s}\t{PY} --dataset {d} --model {m} --seed {s} --arm {arm} --ratio .{r}{flag}")
insert(prune)
print(len(prune), "prune jobs queued", flush=True)
