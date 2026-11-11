#!/usr/bin/env python3
"""Plan S: 90% across eight CIFAR-100 models. S1 gradual+floor vs ft_const, S2 gradual+floor vs rewind (paired test, >= +0.3pp and 3/3).

Prints a table and writes runs/other_datasets/PLAN_S_RESULTS.json. Groups with missing runs are reported as pending.
"""
import csv, json, subprocess, sys
from pathlib import Path

R = Path("runs/other_datasets")
C = Path(__file__).resolve().parents[2]/"mask_aware_conv_accelerator/scripts/estimate_network_cycles.py"
TMP = R/"plan_s_tmp.csv"
MODELS = ("resnet18", "resnet34", "resnet50", "resnet56", "resnet110", "vgg16_bn", "googlenet", "densenet121")
SUF = {"dense": "dense", "ft_const": "ft_const_r90", "rewind": "rewind_r90", "floor": "gradual_r90_floor"}


def load(model, seed, arm):
    path = R/f"cifar100_{model}_seed{seed}_{SUF[arm]}"
    return (json.loads((path/"result.json").read_text()), path) if (path/"result.json").exists() else (None, path)


def ms(model, path):
    subprocess.run([sys.executable, str(C), "--geometry", str(R/f"geometry_cifar100_{model}.json"), "--masks", f"a={path/'masks.pt'}", "--out", str(TMP)],
                   check=True, stdout=subprocess.DEVNULL)
    rows = {r["label"]: float(r["ms_50mhz"]) for r in csv.DictReader(TMP.open())}
    return rows["a"], rows["dense"]


def verdict(d):
    return "PASS" if len(d) == 3 and sum(d)/3 >= .3 and all(x > 0 for x in d) else "fail"


out = {}
for seeds in ((0, 1, 2), (3, 4, 5)):
    # Seeds 3/4/5 only for the plan R Dense models; ResNet34 seeds 3/4/5 used a stretched Dense LR curve (PLAN_H.md).
    for model in (MODELS if seeds == (0, 1, 2) else ("resnet56", "vgg16_bn")):
        got = {arm: [load(model, s, arm) for s in seeds] for arm in SUF}
        if any(r is None for r, _ in got["floor"]) or all(r is None for r, _ in got["dense"]):
            if seeds == (0, 1, 2):
                print(f"{model:11s} seeds {seeds}: pending")
            continue
        acc = {arm: [r["final_test"]*100 if r else None for r, _ in v] for arm, v in got.items()}
        mean = lambda xs: sum(xs)/len(xs)
        rec = {"dense": mean([a for a in acc["dense"] if a is not None])}
        line = f"{model:11s} seeds {'/'.join(map(str, seeds))}: Dense {rec['dense']:.2f}"
        for arm in ("ft_const", "rewind", "floor"):
            if None in acc[arm]:
                line += f" | {arm} pending"
                continue
            rec[arm] = mean(acc[arm])
            line += f" | {arm} {rec[arm]:.2f}"
            if arm != "ft_const":
                cyc = [ms(model, p) for _, p in got[arm] if (p/"masks.pt").exists()]
                if cyc:
                    rec[f"{arm}_ms"], rec["dense_ms"] = mean([c for c, _ in cyc]), cyc[0][1]
                    line += f" ({rec[f'{arm}_ms']:.2f} ms)"
        for key, ref in (("S1", "ft_const"), ("S2", "rewind")):
            if None not in acc[ref]:
                d = [a-b for a, b in zip(acc["floor"], acc[ref])]
                rec[key] = dict(diffs=d, mean=mean(d), verdict=verdict(d))
                line += f" | {key} floor-{ref} {'/'.join(f'{x:+.2f}' for x in d)} ({mean(d):+.2f}) {verdict(d)}"
        print(line)
        out[f"{model}_seeds{''.join(map(str, seeds))}"] = rec
TMP.unlink(missing_ok=True)
(R/"PLAN_S_RESULTS.json").write_text(json.dumps(out, indent=1))
