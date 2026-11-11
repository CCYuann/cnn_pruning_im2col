#!/usr/bin/env python3
"""Queue plan S (90% across eight models): existing-Dense jobs now, ResNet18/ResNet-110 after their Dense."""
import os, time
from pathlib import Path

ROOT = Path("runs/breakthrough_r0p95")
Q, LOGS = ROOT/"queue.txt", ROOT/"logs"
DS = "/usr/local/anaconda3/bin/python3 scripts/dataset_pruning_worker.py --dataset cifar100"
ARMS = {"ft_const": "--arm ft_const", "rewind": "--arm rewind", "floor": "--arm gradual --q-floor"}
EXISTING = {"ft_const": ("resnet34",), "rewind": ("resnet34", "resnet50", "resnet56", "vgg16_bn", "googlenet", "densenet121"),
            "floor": ("resnet34", "resnet50", "resnet56", "vgg16_bn", "googlenet", "densenet121")}


def append(lines):
    rows = Q.read_text().splitlines()
    tmp = Q.with_suffix(".new")
    tmp.write_text("\n".join(rows+[f"{n}\t{c}" for n, c in lines])+"\n")
    os.replace(tmp, Q)


def jobs(model, seeds, skip=True):
    out = []
    for s in seeds:
        for arm, flags in ARMS.items():
            if skip and s < 3 and model in EXISTING[arm]:
                continue
            out.append((f"s_{model}_{arm}_r90_{s}", f"{DS} --model {model} --seed {s} {flags} --ratio .90" + ("" if arm == "floor" else " --discard-model")))
    return out


rc = lambda n: (LOGS/f"{n}.rc").read_text().strip() if (LOGS/f"{n}.rc").exists() else None
dense = {m: [(f"s_{m}_dense_{s}", f"{DS} --model {m} --seed {s} --arm dense") for s in (0, 1, 2)] for m in ("resnet18", "resnet110")}
now = [j for v in dense.values() for j in v]
for m in ("resnet56", "vgg16_bn", "googlenet", "resnet50", "densenet121"):
    now += jobs(m, (0, 1, 2))
now += jobs("resnet56", (3, 4, 5)) + jobs("vgg16_bn", (3, 4, 5))
append(now)
print(len(now), "queued", time.strftime("%H:%M"), flush=True)
later = {m: jobs(m, (0, 1, 2), skip=False) for m in dense}
while later:
    for m in list(later):
        codes = [rc(n) for n, _ in dense[m]]
        if any(c not in (None, "0") for c in codes):
            print("Dense failed", m, codes, flush=True); del later[m]
        elif all(c == "0" for c in codes):
            append(later[m]); print(m, "Dense finished, queued", len(later.pop(m)), time.strftime("%H:%M"), flush=True)
    time.sleep(60)
