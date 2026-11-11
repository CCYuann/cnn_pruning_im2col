#!/usr/bin/env python3
"""Queue plan M: Dense runs and prerequisite-free runs now, pruned runs once their Dense finishes."""
import os, time
from pathlib import Path

ROOT = Path("runs/breakthrough_r0p95")
Q, LOGS = ROOT/"queue.txt", ROOT/"logs"
PY = "/usr/local/anaconda3/bin/python3"
DS = f"{PY} scripts/dataset_pruning_worker.py --dataset cifar100"
C10 = f"{PY} scripts/retrain_recovery_worker.py --root runs/confirmation_r0p95 --group-lambda 3e-5 --model resnet34"


def append(lines):
    rows = Q.read_text().splitlines()
    tmp = Q.with_suffix(".new")
    tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in lines])+"\n")
    os.replace(tmp, Q)


def finished(names):
    return all((LOGS/f"{n}.rc").exists() for n in names)


def failed(names):
    return [n for n in names if (LOGS/f"{n}.rc").exists() and (LOGS/f"{n}.rc").read_text().strip() != "0"]


dense = {"m2": [(f"m2_dense_{s}", f"{DS} --model resnet34 --seed {s} --arm dense") for s in (6, 7, 8)]}
for m in ("googlenet", "densenet121"):
    dense[m] = [(f"m3_{m}_dense_{s}", f"{DS} --model {m} --seed {s} --arm dense") for s in (0, 1, 2)]
m1 = [(f"m1_c100_ft_r{r}_{s}", f"{DS} --model resnet34 --seed {s} --arm ft_const --ratio .{r} --discard-model") for s in (0, 1, 2) for r in (90, 95, 98, 99)]
m1 += [(f"m1_c10_ft_r{r}_{s}", f"{C10} --seed {s} --ratio .{r} --arm ft_const --tag _r34r{r}") for s in (6, 7, 8) for r in (98, 99)]
later = {"m2": [], "googlenet": [], "densenet121": []}
for s in (6, 7, 8):
    for r in (90, 95, 98):
        keep = "" if r == 98 else " --discard-model"
        later["m2"] += [(f"m2_floor_r{r}_{s}", f"{DS} --model resnet34 --seed {s} --arm gradual --ratio .{r} --q-floor{keep}"),
                        (f"m2_kd_r{r}_{s}", f"{DS} --model resnet34 --seed {s} --arm gradual --ratio .{r} --q-floor --kd{keep}")]
    later["m2"].append((f"m2_rewind_r98_{s}", f"{DS} --model resnet34 --seed {s} --arm rewind --ratio .98 --discard-model"))
for m in ("googlenet", "densenet121"):
    for s in (0, 1, 2):
        for r in (90, 98, 99):
            later[m] += [(f"m3_{m}_rewind_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm rewind --ratio .{r}"),
                         (f"m3_{m}_floor_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm gradual --ratio .{r} --q-floor")]
        for r in (98, 99):
            later[m].append((f"m3_{m}_unstructured_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm unstructured --ratio .{r}"))

append([j for v in dense.values() for j in v]+m1)
print(sum(len(v) for v in dense.values())+len(m1), "queued", time.strftime("%H:%M"), flush=True)
pending = dict(later)
while pending:
    for key in list(pending):
        names = [n for n, _ in dense[key]]
        if failed(names):
            print("Dense failed", key, failed(names), flush=True)
            del pending[key]
        elif finished(names):
            append(pending.pop(key))
            print(key, "Dense finished, queued", len(later[key]), time.strftime("%H:%M"), flush=True)
    time.sleep(60)
everything = [n for v in dense.values() for n, _ in v]+[n for n, _ in m1]+[n for v in later.values() for n, _ in v]
while not finished(everything):
    time.sleep(120)
print("M finished", time.strftime("%H:%M"), "failed", failed(everything), flush=True)
