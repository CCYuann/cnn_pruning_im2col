#!/usr/bin/env python3
"""Queue plan R: ResNet34 seeds 6/7/8 now; VGG16-BN and ResNet-56 seeds 3/4/5 once their Dense finishes."""
import os, time
from pathlib import Path

ROOT = Path("runs/breakthrough_r0p95")
Q, LOGS = ROOT/"queue.txt", ROOT/"logs"
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
FILL = "--arm gradual --q-floor --q-floor-n 15 --free-fill"


def append(lines):
    rows = Q.read_text().splitlines()
    tmp = Q.with_suffix(".new")
    tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in lines])+"\n")
    os.replace(tmp, Q)


rc = lambda n: (LOGS/f"{n}.rc").read_text().strip() if (LOGS/f"{n}.rc").exists() else None
now = []
for s in (6, 7, 8):
    now += [(f"r_resnet34_floor_r99_{s}", f"{DS} --model resnet34 --seed {s} --arm gradual --ratio .99 --q-floor --discard-model"),
            (f"r_resnet34_fill15_r98_{s}", f"{DS} --model resnet34 --seed {s} {FILL} --ratio .98 --discard-model"),
            (f"r_resnet34_fill15_r99_{s}", f"{DS} --model resnet34 --seed {s} {FILL} --ratio .99 --discard-model"),
            (f"r_resnet34_fill15kd_r98_{s}", f"{DS} --model resnet34 --seed {s} {FILL} --ratio .98 --kd --discard-model")]
dense, later = {}, {}
for m in ("vgg16_bn", "resnet56"):
    dense[m] = [(f"r_{m}_dense_{s}", f"{DS} --model {m} --seed {s} --arm dense") for s in (3, 4, 5)]
    now += dense[m]
    later[m] = []
    for s in (3, 4, 5):
        later[m] += [(f"r_{m}_rewind_r95_{s}", f"{DS} --model {m} --seed {s} --arm rewind --ratio .95 --discard-model")]
        later[m] += [(f"r_{m}_floor_r{r}_{s}", f"{DS} --model {m} --seed {s} --arm gradual --ratio .{r} --q-floor --discard-model") for r in (95, 98, 99)]
        later[m] += [(f"r_{m}_fill15_r{r}_{s}", f"{DS} --model {m} --seed {s} {FILL} --ratio .{r} --discard-model") for r in (98, 99)]
        if m == "vgg16_bn":
            later[m] += [(f"r_{m}_fill15kd_r98_{s}", f"{DS} --model {m} --seed {s} {FILL} --ratio .98 --kd --discard-model")]
append(now)
print(len(now), "queued", time.strftime("%H:%M"), flush=True)
while later:
    for m in list(later):
        codes = [rc(n) for n, _ in dense[m]]
        if any(c not in (None, "0") for c in codes):
            print("Dense failed", m, codes, flush=True); del later[m]
        elif all(c == "0" for c in codes):
            append(later[m]); print(m, "Dense finished, queued", len(later.pop(m)), time.strftime("%H:%M"), flush=True)
    time.sleep(60)
