#!/usr/bin/env python3
"""Queue CIFAR-100 Dense seeds 3/4/5 ahead of the unstarted work, then the frozen plan J runs."""
import os, time
from pathlib import Path

ROOT = Path("runs/breakthrough_r0p95")
Q, LOGS = ROOT/"queue.txt", ROOT/"logs"
BASE = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100 --model resnet34"
SEEDS = (3, 4, 5)


def insert(lines):
    rows = Q.read_text().splitlines()
    at = next((i for i, r in enumerate(rows) if r.strip() and not any((LOGS/f"{r.split(chr(9))[0]}{e}").exists() for e in (".log", ".rc", ".adopted"))), len(rows))
    tmp = Q.with_suffix(".new")
    tmp.write_text("\n".join(rows[:at]+lines+rows[at:])+"\n")
    os.replace(tmp, Q)


def wait(names):
    while not all((LOGS/f"{n}.rc").exists() for n in names):
        time.sleep(60)
    failed = [n for n in names if (LOGS/f"{n}.rc").read_text().strip() != "0"]
    if failed:
        raise SystemExit(f"failed: {failed}")


dense = [(f"j_dense_{s}", f"{BASE} --seed {s} --arm dense") for s in SEEDS]
insert([f"{n}\t{c}" for n, c in dense])
wait([n for n, _ in dense])
print("Dense finished", time.strftime("%H:%M"), flush=True)
runs = []
for s in SEEDS:
    for r in (98, 99):
        runs += [(f"j_kd_r{r}_{s}", f"{BASE} --seed {s} --arm gradual --ratio .{r} --q-floor --kd"),
                 (f"j_floor_r{r}_{s}", f"{BASE} --seed {s} --arm gradual --ratio .{r} --q-floor"),
                 (f"j_rewind_r{r}_{s}", f"{BASE} --seed {s} --arm rewind --ratio .{r} --discard-model")]
insert([f"{n}\t{c}" for n, c in runs])
print(len(runs), "J runs queued", flush=True)
wait([n for n, _ in runs])
print("J finished", time.strftime("%H:%M"), flush=True)
