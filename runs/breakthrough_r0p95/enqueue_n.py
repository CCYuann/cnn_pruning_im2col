#!/usr/bin/env python3
"""Queue plan N: runs without prerequisites now, VGG16-BN and ResNet-56 pruned runs once their Dense finishes."""
import os, time
from pathlib import Path

ROOT = Path("runs/breakthrough_r0p95")
Q, LOGS = ROOT/"queue.txt", ROOT/"logs"
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
NEW, OLD, SEEDS = ("vgg16_bn", "resnet56"), ("resnet34", "googlenet", "densenet121"), (0, 1, 2)


def append(lines):
    rows = Q.read_text().splitlines()
    tmp = Q.with_suffix(".new")
    tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in lines])+"\n")
    os.replace(tmp, Q)


def finished(names):
    return all((LOGS/f"{n}.rc").exists() for n in names)


def failed(names):
    return [n for n in names if (LOGS/f"{n}.rc").exists() and (LOGS/f"{n}.rc").read_text().strip() != "0"]


def structure(m):
    return [(f"n2_{m}_gu_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm gradual_unstructured --ratio .{r} --discard-model") for s in SEEDS for r in (98, 99)]


def kd(m):
    return [(f"n3_{m}_kd_r98_{s}", f"{DS} --model {m} --seed {s} --arm gradual --ratio .98 --q-floor --kd") for s in SEEDS]


dense = {m: [(f"n1_{m}_dense_{s}", f"{DS} --model {m} --seed {s} --arm dense") for s in SEEDS] for m in NEW}
now = [j for m in NEW for j in dense[m]]+[j for m in OLD for j in structure(m)]+[j for m in OLD[1:] for j in kd(m)]
later = {}
for m in NEW:
    later[m] = []
    for s in SEEDS:
        for r in (90, 98, 99):
            later[m] += [(f"n1_{m}_rewind_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm rewind --ratio .{r}"),
                         (f"n1_{m}_floor_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm gradual --ratio .{r} --q-floor")]
        later[m] += [(f"n1_{m}_unstructured_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm unstructured --ratio .{r} --discard-model") for r in (98, 99)]
    later[m] += structure(m)+kd(m)
append(now)
print(len(now), "queued", time.strftime("%H:%M"), flush=True)
pending = dict(later)
while pending:
    for m in list(pending):
        names = [n for n, _ in dense[m]]
        if failed(names):
            print("Dense failed", m, failed(names), flush=True)
            del pending[m]
        elif finished(names):
            append(pending.pop(m))
            print(m, "Dense finished, queued", len(later[m]), time.strftime("%H:%M"), flush=True)
    time.sleep(60)
everything = [n for n, _ in now]+[n for v in later.values() for n, _ in v]
while not finished(everything):
    time.sleep(120)
print("N finished", time.strftime("%H:%M"), "failed", failed(everything), flush=True)
