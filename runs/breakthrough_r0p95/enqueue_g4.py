#!/usr/bin/env python3
"""Queue Dense seeds 9/10/11 after CIFAR-100 ResNet34 and before ResNet50, then the frozen G4 runs."""
import os, time
from pathlib import Path

ROOT = Path("runs/breakthrough_r0p95")
Q, LOGS = ROOT/"queue.txt", ROOT/"logs"
PY = "/usr/local/anaconda3/bin/python3"
RUNS = Path("runs").resolve()
SEEDS = (9, 10, 11)


def insert(lines):
    rows = Q.read_text().splitlines()
    at = max(i+1 for i, r in enumerate(rows) if r.startswith(("ds_cifar100_resnet34", "g4_")))
    tmp = Q.with_suffix(".new")
    tmp.write_text("\n".join(rows[:at]+lines+rows[at:])+"\n")
    os.replace(tmp, Q)


dense = [f"g4_dense_r34_{s}\t{PY} resnet34_pruning_experiments.py --dataset cifar10 --models resnet34 --device cuda --amp --amp-dtype float16 "
         f"--epochs 100 --lr 0.1 --lr-scheduler cosine --min-lr 0.0001 --batch-size 128 --num-workers 2 --pin-memory --seed {s} --val-size 5000 "
         f"--split-seed 0 --deterministic --no-cudnn-benchmark --train-dense-only --dense-checkpoint-out runs/fixed_checkpoint_resnet34_new/checkpoints/resnet34_seed{s}_dense.pt "
         f"--save-dir runs/fixed_checkpoint_resnet34_new --run-name dense_resnet34_seed{s}" for s in SEEDS]
insert(dense)
names = [d.split("\t")[0] for d in dense]
while not all((LOGS/f"{n}.rc").exists() for n in names):
    time.sleep(60)
failed = [n for n in names if (LOGS/f"{n}.rc").read_text() != "0"]
if failed:
    raise SystemExit(f"Dense failed: {failed}")
for s in SEEDS:
    (RUNS/f"fixed_checkpoint_resnet34/checkpoints/resnet34_seed{s}_dense.pt").symlink_to(RUNS/f"fixed_checkpoint_resnet34_new/checkpoints/resnet34_seed{s}_dense.pt")
base = f"{PY} scripts/retrain_recovery_worker.py --root runs/confirmation_r0p95 --group-lambda 3e-5 --model resnet34"
protected = "--arm gradual --q-floor --stem-keep"
rows = []
for s in SEEDS:
    for r in (98, 99):
        rows.append(f"g4_rewind_r{r}_{s}\t{base} --seed {s} --ratio .{r} --arm rewind --tag _r34r{r}")
        rows.append(f"g4_fs_r{r}_{s}\t{base} --seed {s} --ratio .{r} {protected} --tag _r34r{r}_fs")
        rows.append(f"g4_fill15_r{r}_{s}\t{base} --seed {s} --ratio .{r} {protected} --q-floor-n 15 --free-fill --tag _r34r{r}_fill15")
        rows.append(f"g4_b15_r{r}_{s}\t{base} --seed {s} --ratio .{r} {protected} --q-floor-n 15 --tag _r34r{r}_b15")
insert(rows)
print(len(rows), "G4 runs queued", flush=True)
while not all((LOGS/f"{r.split(chr(9))[0]}.rc").exists() for r in rows):
    time.sleep(60)
print("G4 finished", time.strftime("%H:%M"), flush=True)
