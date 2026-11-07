#!/usr/bin/env python3
"""Queue ResNet34 Dense seeds 6/7/8 and dataset controls now, then the frozen G3 runs once Dense succeeds."""
import os, time
from pathlib import Path

ROOT = Path("runs/breakthrough_r0p95")
Q, LOGS = ROOT/"queue.txt", ROOT/"logs"
PY = "/usr/local/anaconda3/bin/python3"
RUNS = Path("runs").resolve()
SEEDS = (6, 7, 8)


def begun(row):
    name = row.split("\t")[0]
    return any((LOGS/f"{name}{e}").exists() for e in (".log", ".rc", ".adopted"))


def insert_front(lines):
    rows = Q.read_text().splitlines()
    at = next((i for i, r in enumerate(rows) if not begun(r)), len(rows))
    tmp = Q.with_suffix(".new")
    tmp.write_text("\n".join(rows[:at]+lines+rows[at:])+"\n")
    os.replace(tmp, Q)


dense = [f"g3_dense_r34_{s}\t{PY} resnet34_pruning_experiments.py --dataset cifar10 --models resnet34 --device cuda --amp --amp-dtype float16 "
         f"--epochs 100 --lr 0.1 --lr-scheduler cosine --min-lr 0.0001 --batch-size 128 --num-workers 2 --pin-memory --seed {s} --val-size 5000 "
         f"--split-seed 0 --deterministic --no-cudnn-benchmark --train-dense-only --dense-checkpoint-out runs/fixed_checkpoint_resnet34_new/checkpoints/resnet34_seed{s}_dense.pt "
         f"--save-dir runs/fixed_checkpoint_resnet34_new --run-name dense_resnet34_seed{s}" for s in SEEDS]
control = [f"ctl_{d}_{m}_{s}\t{PY} scripts/dataset_pruning_worker.py --dataset {d} --model {m} --seed {s} --arm rewind --ratio 0"
           for d, m in (("imagenette", "resnet34"), ("imagewoof", "resnet34"), ("cifar100", "resnet34"), ("cifar100", "resnet50")) for s in (0, 1, 2)]
insert_front(dense+control)
names = [line.split("\t")[0] for line in dense]
while not all((LOGS/f"{n}.rc").exists() for n in names):
    time.sleep(60)
failed = [n for n in names if (LOGS/f"{n}.rc").read_text() != "0"]
if failed:
    raise SystemExit(f"Dense failed: {failed}")
for s in SEEDS:
    link = RUNS/f"fixed_checkpoint_resnet34/checkpoints/resnet34_seed{s}_dense.pt"
    link.symlink_to(RUNS/f"fixed_checkpoint_resnet34_new/checkpoints/resnet34_seed{s}_dense.pt")
rows = []
base = f"{PY} scripts/retrain_recovery_worker.py --root runs/confirmation_r0p95 --group-lambda 3e-5 --model resnet34"
for s in SEEDS:
    for r in (95, 98, 99):
        rows.append(f"g3_rewind_r{r}_{s}\t{base} --seed {s} --arm rewind --ratio .{r} --tag _r34r{r}")
        rows.append(f"g3_gradual_r{r}_{s}\t{base} --seed {s} --arm gradual --ratio .{r} --tag _r34r{r}")
        rows.append(f"g3_fs_r{r}_{s}\t{base} --seed {s} --arm gradual --ratio .{r} --q-floor --stem-keep --tag _r34r{r}_fs")
    rows.append(f"g3_floor_r98_{s}\t{base} --seed {s} --arm gradual --ratio .98 --q-floor --tag _r34r98_floor")
insert_front(rows)
print(len(rows), "G3 runs queued", flush=True)
while not all((LOGS/f"{r.split(chr(9))[0]}.rc").exists() for r in rows):
    time.sleep(60)
print("G3 finished", time.strftime("%H:%M"), flush=True)
